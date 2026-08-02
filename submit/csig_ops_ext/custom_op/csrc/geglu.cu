// geglu：替换 runner.py 的 triton_geglu。(..., 2D) -> (..., D)，沿最后一维前后对半切：
//   h = y2[..., :D]，g = y2[..., D:]，out = h * 0.5 * g * (1 + erf(g / sqrt(2)))
// 注意是 erf 精确版 gelu（与 diffusers 的 F.gelu 默认一致），不是 tanh 近似。
// fp32 计算、写回时一次舍入。纯访存瓶颈（读 2D 写 D），故按 VEC=4/8 向量化。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include "common.cuh"

namespace {

using namespace custom_op;

constexpr int THREADS = 256;

// 每行输出 d 个元素，输入行长 2d：h 段与 g 段各自连续，故一次 VEC 宽的 load 不会跨段。
template<typename T, int VEC>
__global__ __launch_bounds__(THREADS) void geglu_kernel(const T* __restrict__ y2,
                                                        T* __restrict__ out,
                                                        int64_t rows,
                                                        int     d)
{
    const int     vec_per_row = d / VEC;
    const int64_t total_vecs  = rows * vec_per_row;
    const int64_t stride      = static_cast<int64_t>(gridDim.x) * THREADS;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * THREADS + threadIdx.x; idx < total_vecs; idx += stride) {
        const int64_t row  = idx / vec_per_row;
        const int64_t col  = (idx % vec_per_row) * VEC;
        const T*      base = y2 + row * 2 * d + col;

        vec_t<T, VEC> hv, gv, ov;
        hv.load(base);
        gv.load(base + d);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const float h = fp_traits<T>::to_float(hv.data[i]);
            const float g = fp_traits<T>::to_float(gv.data[i]);
            ov.data[i]    = fp_traits<T>::from_float(h * 0.5f * g * (1.f + erff(g * 0.70710678118654752f)));
        }
        ov.store(out + row * d + col);
    }
}

}  // namespace

at::Tensor geglu_cuda(const at::Tensor& y2)
{
    TORCH_CHECK(y2.is_cuda(), "geglu: input must be CUDA");
    TORCH_CHECK(y2.dim() >= 1, "geglu: input must have at least 1 dim");
    TORCH_CHECK(y2.is_contiguous(), "geglu: input must be contiguous");
    const int64_t last = y2.size(-1);
    TORCH_CHECK(last % 2 == 0, "geglu: last dim must be even, got ", last);

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(y2));

    const int     d    = static_cast<int>(last / 2);
    const int64_t rows = y2.numel() / last;
    auto          sizes = y2.sizes().vec();
    sizes.back()        = d;
    auto out            = at::empty(sizes, y2.options());
    if (rows == 0 || d == 0)
        return out;

    auto      stream = at::cuda::getCurrentCUDAStream();
    const int vec    = (d % 8 == 0) ? 8 : ((d % 4 == 0) ? 4 : 1);

    DISPATCH_FP16_FP32(y2.scalar_type(), "geglu", [&] {
        const auto* p      = reinterpret_cast<const scalar_t*>(y2.data_ptr());
        auto*       o      = reinterpret_cast<scalar_t*>(out.data_ptr());
        const int64_t nvec = rows * (d / vec);
        const int   blocks = static_cast<int>(std::min<int64_t>((nvec + THREADS - 1) / THREADS, 4096));
        if (vec == 8)
            geglu_kernel<scalar_t, 8><<<blocks, THREADS, 0, stream>>>(p, o, rows, d);
        else if (vec == 4)
            geglu_kernel<scalar_t, 4><<<blocks, THREADS, 0, stream>>>(p, o, rows, d);
        else
            geglu_kernel<scalar_t, 1><<<blocks, THREADS, 0, stream>>>(p, o, rows, d);
    });
    return out;
}

TORCH_LIBRARY_IMPL(custom_op, CUDA, m)
{
    m.impl("geglu", &geglu_cuda);
}
