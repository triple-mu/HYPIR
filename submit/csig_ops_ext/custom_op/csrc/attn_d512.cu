// attn_d512：替换 runner.py 的 triton_attn_d512。单 (batch, head)、非因果、head_dim=512 的
// attention：out = softmax(q @ kᵀ / sqrt(512)) @ v。q/k/v 形如 (1, N, 512) fp16，最后一维连续，
// 行 stride 可为 512 或 1536（融合 QKV 的 chunk 视图）。用在 VAE mid-block，生产形状 N=4096。
//
// 实现选择（与 Triton 版的 split-D flash 不同，这里是刻意的取舍）：
//   Volta(sm70) 没有 cp.async、也没有 mma.16816，只有 HMMA.884；而 flash 要把 BLOCK_M×512 的 fp32
//   累加器留在寄存器里（BLOCK_M=16 时每线程 64 个寄存器仅累加器就占满），sm70 上寄存器压力会直接
//   把 occupancy 压到 1-2 warp/SM。相反，N=4096 时把 N×N 分数矩阵物化只要 33.5 MB，两次 GEMM 交给
//   cuBLAS（Volta 上高度调优的 HMMA 路径），中间只需要一个融合的 scale+softmax kernel。
//   所以这里走「cuBLAS GEMM ×2 + 融合 softmax」；若后续 profile 显示物化的访存成为瓶颈，再补 flash 变体。
//
// 融合 kernel：一个 block 负责一行，fp32 累加求 max 与 sum，行元素留在寄存器里避免二次读全局。
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <math_constants.h>
#include <torch/all.h>
#include <torch/library.h>

#include "common.cuh"

namespace {

using namespace custom_op;

constexpr int SM_THREADS  = 256;
constexpr int MAX_PER_THR = 32;  // 支持到 N = 256 * 32 = 8192 列

// 原地把一行 (scale 后) 做 softmax。行长 n_cols <= SM_THREADS * MAX_PER_THR。
template<typename T>
__global__ __launch_bounds__(SM_THREADS) void softmax_scale_kernel(T* __restrict__ s,
                                                                   int64_t n_cols,
                                                                   float   scale)
{
    __shared__ float smem[SM_THREADS / 32 * 2];

    T* row = s + static_cast<int64_t>(blockIdx.x) * n_cols;

    float vals[MAX_PER_THR];
    int   cnt = 0;
    float m   = -CUDART_INF_F;
    for (int64_t c = threadIdx.x; c < n_cols; c += SM_THREADS) {
        const float v = fp_traits<T>::to_float(row[c]) * scale;
        vals[cnt++]   = v;
        m             = fmaxf(m, v);
    }

    // block max：复用 sum2 的 smem，但归约算子是 max，故单独写一遍
    m = fmaxf(m, __shfl_down_sync(0xffffffff, m, 16));
    m = fmaxf(m, __shfl_down_sync(0xffffffff, m, 8));
    m = fmaxf(m, __shfl_down_sync(0xffffffff, m, 4));
    m = fmaxf(m, __shfl_down_sync(0xffffffff, m, 2));
    m = fmaxf(m, __shfl_down_sync(0xffffffff, m, 1));
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = smem[0];
#pragma unroll
        for (int i = 1; i < SM_THREADS / 32; ++i)
            mm = fmaxf(mm, smem[i]);
        smem[0] = mm;
    }
    __syncthreads();
    const float row_max = smem[0];
    __syncthreads();

    float sum = 0.f;
    for (int i = 0; i < cnt; ++i) {
        vals[i] = __expf(vals[i] - row_max);
        sum += vals[i];
    }
    float dummy = 0.f;
    block_reduce_sum2<SM_THREADS>(sum, dummy, smem);

    const float inv = 1.f / sum;
    int         k   = 0;
    for (int64_t c = threadIdx.x; c < n_cols; c += SM_THREADS)
        row[c] = fp_traits<T>::from_float(vals[k++] * inv);
}

}  // namespace

at::Tensor attn_d512_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "attn_d512: inputs must be CUDA");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "attn_d512: inputs must be (B, N, D)");
    TORCH_CHECK(q.size(0) == 1, "attn_d512: only batch 1 is supported");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "attn_d512: q / k / v shape mismatch");
    TORCH_CHECK(q.size(2) == 512, "attn_d512: head_dim must be 512, got ", q.size(2));
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1,
                "attn_d512: last dim must be contiguous");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "attn_d512: dtype mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

    const int64_t n = q.size(1);
    TORCH_CHECK(n <= SM_THREADS * MAX_PER_THR, "attn_d512: N=", n, " exceeds ", SM_THREADS * MAX_PER_THR);
    auto out = at::empty_like(q, q.options().memory_format(at::MemoryFormat::Contiguous));
    if (n == 0)
        return out;

    // GEMM 交给 cuBLAS：scores (1, N, N) = q @ kᵀ。at::matmul 会按 stride 直接喂 lda，
    // 融合 QKV 的 stride=1536 视图不会触发额外拷贝。
    auto scores = at::matmul(q, k.transpose(-1, -2));
    TORCH_CHECK(scores.is_contiguous(), "attn_d512: scores expected contiguous");

    auto        stream = at::cuda::getCurrentCUDAStream();
    const float scale  = 1.f / std::sqrt(512.f);
    DISPATCH_FP16_FP32(scores.scalar_type(), "attn_d512", [&] {
        softmax_scale_kernel<scalar_t>
            <<<static_cast<int>(n), SM_THREADS, 0, stream>>>(reinterpret_cast<scalar_t*>(scores.data_ptr()), n, scale);
    });

    at::matmul_out(out, scores, v);
    return out;
}

TORCH_LIBRARY_IMPL(custom_op, CUDA, m)
{
    m.impl("attn_d512", &attn_d512_cuda);
}
