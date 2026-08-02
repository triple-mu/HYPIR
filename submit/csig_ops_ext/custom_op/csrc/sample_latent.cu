// sample_latent：替换 runner.py 的 triton_sample_latent，等价 diffusers 的
// DiagonalGaussianDistribution.sample()：z = mean + exp(0.5 * clamp(logvar, -30, 20)) * noise。
//
// 两个输入的物理布局故意不同，照搬 Triton 版的约定：
//   moments (N, 8, H, W) 是 channels_last（NHWC），前 4 通道 mean、后 4 通道 logvar；
//   noise   (N, 4, H, W) 是 contiguous（NCHW）—— 保持 torch.randn 的原始布局，
//           这样无论走不走本算子，消耗的随机数流都与 eager 完全一致（同 seed 可复现）。
//   输出    (N, 4, H, W) channels_last。
// fp32 计算，写回时一次舍入。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include "common.cuh"

namespace {

using namespace custom_op;

constexpr int THREADS = 256;
constexpr int LAT_C   = 4;  // latent 通道数固定 4（SD2.1 VAE）

template<typename T>
__global__ __launch_bounds__(THREADS) void sample_latent_kernel(const T* __restrict__ moments,
                                                                const T* __restrict__ noise,
                                                                T* __restrict__ z,
                                                                int64_t total,
                                                                int64_t hw)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * THREADS;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * THREADS + threadIdx.x; idx < total; idx += stride) {
        const int64_t pix = idx / LAT_C;  // 展平的 (n, hw)
        const int64_t c   = idx % LAT_C;

        // moments 是 NHWC，每像素 8 个通道连续：[0,4) mean、[4,8) logvar
        const float mean = fp_traits<T>::to_float(moments[pix * 2 * LAT_C + c]);
        float       lv   = fp_traits<T>::to_float(moments[pix * 2 * LAT_C + LAT_C + c]);
        lv               = fminf(fmaxf(lv, -30.f), 20.f);
        // noise 是 NCHW：((n * 4 + c) * hw + hw_idx)
        const float nz = fp_traits<T>::to_float(noise[(pix / hw * LAT_C + c) * hw + pix % hw]);

        z[idx] = fp_traits<T>::from_float(mean + __expf(0.5f * lv) * nz);
    }
}

}  // namespace

at::Tensor sample_latent_cuda(const at::Tensor& moments, const at::Tensor& noise)
{
    TORCH_CHECK(moments.is_cuda() && noise.is_cuda(), "sample_latent: inputs must be CUDA");
    TORCH_CHECK(moments.dim() == 4 && noise.dim() == 4, "sample_latent: inputs must be 4D");
    TORCH_CHECK(moments.size(1) == 2 * LAT_C, "sample_latent: moments must have 8 channels");
    TORCH_CHECK(noise.size(1) == LAT_C, "sample_latent: noise must have 4 channels");
    TORCH_CHECK(moments.size(0) == noise.size(0) && moments.size(2) == noise.size(2)
                    && moments.size(3) == noise.size(3),
                "sample_latent: moments / noise spatial shape mismatch");
    TORCH_CHECK(moments.is_contiguous(at::MemoryFormat::ChannelsLast), "sample_latent: moments must be channels_last");
    TORCH_CHECK(noise.is_contiguous(), "sample_latent: noise must be contiguous (NCHW)");
    TORCH_CHECK(moments.scalar_type() == noise.scalar_type(), "sample_latent: dtype mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(moments));

    const int64_t n  = moments.size(0);
    const int64_t h  = moments.size(2);
    const int64_t w  = moments.size(3);
    const int64_t hw = h * w;

    auto z = at::empty({n, LAT_C, h, w}, moments.options().memory_format(at::MemoryFormat::ChannelsLast));
    const int64_t total = z.numel();
    if (total == 0)
        return z;

    auto      stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(std::min<int64_t>((total + THREADS - 1) / THREADS, 4096));

    DISPATCH_FP16_FP32(moments.scalar_type(), "sample_latent", [&] {
        sample_latent_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(moments.data_ptr()),
            reinterpret_cast<const scalar_t*>(noise.data_ptr()),
            reinterpret_cast<scalar_t*>(z.data_ptr()),
            total,
            hw);
    });
    return z;
}

TORCH_LIBRARY_IMPL(custom_op, CUDA, m)
{
    m.impl("sample_latent", &sample_latent_cuda);
}
