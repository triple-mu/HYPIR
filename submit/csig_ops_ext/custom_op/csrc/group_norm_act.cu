// group_norm_act：NHWC 上的 GroupNorm(+可选 SiLU)，替换 runner.py 的 triton_group_norm_act。
// y = (x - mean_g) * rstd_g * weight[c] + bias[c]，act=1 时再过 SiLU；mean/rstd 按 (n, g) 统计，
// 组 g 覆盖通道 [g*cpg, (g+1)*cpg) 与全部 HW 像素。全程 fp32 累加与计算，只在写回时舍入一次。
//
// 布局：x 必须是 channels_last，即物理排布 n -> hw -> c，故一个组的元素是「每个 hw 位置上一段
// 长 cpg 的连续通道」。三段式与 Triton 版同构：
//   ① gn_stats  : grid (N*G, SPLIT)，每块归约自己那段 hw 的 sum/sumsq -> partial
//   ② gn_finalize: grid (N*G)，把 SPLIT 份 partial 加起来（SPLIT==1 时跳过）
//   ③ gn_apply  : grid (HW_tiles, N)，先在 smem 里把 per-group 的 mean/rstd 展开成 per-channel 的
//                 scale/shift（消掉逐元素的整数除法），再逐元素仿射
// 归约顺序固定（warp shuffle + 单线程串行合并 + 固定 SPLIT 顺序），故同输入逐次 bit 一致。
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include "common.cuh"

namespace {

using namespace custom_op;

constexpr int STATS_THREADS = 256;
constexpr int APPLY_THREADS = 256;
constexpr int MAX_SPLIT     = 64;

template<typename T>
__global__ __launch_bounds__(STATS_THREADS) void gn_stats_kernel(const T* __restrict__ x,
                                                                 float* __restrict__ part,
                                                                 int64_t hw,
                                                                 int     c,
                                                                 int     cpg,
                                                                 int     groups,
                                                                 int     split)
{
    __shared__ float smem[STATS_THREADS / 32 * 2];

    const int     ng   = static_cast<int>(blockIdx.x);  // n * G + g
    const int     s    = static_cast<int>(blockIdx.y);
    const int64_t n    = ng / groups;
    const int64_t g    = ng % groups;
    const int64_t base = n * hw * c + g * cpg;

    const int64_t rows_per_split = (hw + split - 1) / split;
    const int64_t hw0            = static_cast<int64_t>(s) * rows_per_split;
    const int64_t hw1            = min(hw0 + rows_per_split, hw);

    // 线程按 (行, 组内通道) 二维铺开：tid -> (row_off = tid / cpg, i = tid % cpg)，
    // 一次迭代覆盖 rows_per_iter 行。若只按 cpg 分线程，cpg=4(C=128) 时 256 线程只有 4 个在干活，
    // 外层还要串行走完 26 万行——实测比 Triton 慢 50 倍，这里靠行方向铺开把并行度补回来。
    const int rows_per_iter = STATS_THREADS / cpg > 0 ? STATS_THREADS / cpg : 1;
    const int r_off         = static_cast<int>(threadIdx.x) / cpg;
    const int i             = static_cast<int>(threadIdx.x) % cpg;

    float sum = 0.f, sqs = 0.f;
    if (r_off < rows_per_iter) {
        for (int64_t row = hw0 + r_off; row < hw1; row += rows_per_iter) {
            const float v = fp_traits<T>::to_float(x[base + row * c + i]);
            sum += v;
            sqs += v * v;
        }
        // cpg > STATS_THREADS 时（本链路不会发生，C<=2560/G=32 -> cpg<=80）补齐剩余通道
        for (int ch = i + STATS_THREADS; ch < cpg; ch += STATS_THREADS) {
            for (int64_t row = hw0; row < hw1; ++row) {
                const float v = fp_traits<T>::to_float(x[base + row * c + ch]);
                sum += v;
                sqs += v * v;
            }
        }
    }
    block_reduce_sum2<STATS_THREADS>(sum, sqs, smem);
    if (threadIdx.x == 0) {
        part[static_cast<int64_t>(ng) * 2 * split + s]         = sum;
        part[static_cast<int64_t>(ng) * 2 * split + split + s] = sqs;
    }
}

__global__ void gn_finalize_kernel(const float* __restrict__ part, float* __restrict__ fin, int split)
{
    const int ng  = static_cast<int>(blockIdx.x);
    float     sum = 0.f, sqs = 0.f;
    for (int s = 0; s < split; ++s) {
        sum += part[static_cast<int64_t>(ng) * 2 * split + s];
        sqs += part[static_cast<int64_t>(ng) * 2 * split + split + s];
    }
    fin[static_cast<int64_t>(ng) * 2]     = sum;
    fin[static_cast<int64_t>(ng) * 2 + 1] = sqs;
}

template<typename T, int ACT>
__global__ __launch_bounds__(APPLY_THREADS) void gn_apply_kernel(const T* __restrict__ x,
                                                                 T* __restrict__ y,
                                                                 const float* __restrict__ fin,
                                                                 const T* __restrict__ weight,
                                                                 const T* __restrict__ bias,
                                                                 int64_t hw,
                                                                 int     c,
                                                                 int     cpg,
                                                                 int     groups,
                                                                 float   inv_cnt,
                                                                 float   eps)
{
    // smem: [0, c) 为 scale = rstd*w，[c, 2c) 为 shift = b - mean*rstd*w。
    // 把 per-group 的 mean/rstd 提前展开到 per-channel，逐元素内循环就没有整数除法了。
    extern __shared__ float smem[];
    const int64_t           n = blockIdx.y;

    for (int ch = threadIdx.x; ch < c; ch += APPLY_THREADS) {
        const int   g    = ch / cpg;
        const float ssum = fin[(n * groups + g) * 2];
        const float ssq  = fin[(n * groups + g) * 2 + 1];
        const float mean = ssum * inv_cnt;
        const float rstd = rsqrtf(ssq * inv_cnt - mean * mean + eps);
        const float w    = fp_traits<T>::to_float(weight[ch]);
        const float b    = fp_traits<T>::to_float(bias[ch]);
        smem[ch]         = rstd * w;
        smem[c + ch]     = b - mean * rstd * w;
    }
    __syncthreads();

    const int64_t base   = n * hw * c;
    const int64_t total  = hw * c;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * APPLY_THREADS;
    const int64_t start  = static_cast<int64_t>(blockIdx.x) * APPLY_THREADS + threadIdx.x;
    // 通道号随固定步长递增，用一次条件减法代替逐元素的整数取模（取模是运行时除法，很贵）。
    int       ch        = static_cast<int>(start % c);
    const int ch_stride = static_cast<int>(stride % c);
    for (int64_t idx = start; idx < total; idx += stride) {
        float v = fp_traits<T>::to_float(x[base + idx]);
        v       = v * smem[ch] + smem[c + ch];
        if (ACT)
            v = v / (1.f + __expf(-v));  // SiLU，fp32 下算完再一次舍入
        y[base + idx] = fp_traits<T>::from_float(v);
        ch += ch_stride;
        if (ch >= c)
            ch -= c;
    }
}

// stats 的块数只有 ng = N*G（G=32，推理时 N=1），远少于 SM 数；靠沿 hw 切 split 补并行度。
// 目标块数取「每块 ELEMS_PER_BLOCK 个元素」与「SM 数的 BLOCKS_PER_SM 倍」的较大值，再向下取 2 的幂
// （非 2 的幂的 rows_per_split 实测会让尾块负载不均、明显掉速）。
// 两个常数是 arch 敏感的：sm86 与 sm70 的最优值不同，故用 env 覆盖以便在目标机上扫参。
//   CSIG_GN_ELEMS_PER_BLOCK / CSIG_GN_BLOCKS_PER_SM，CSIG_GN_SPLIT 则直接锁定 split。
int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (!v || !*v)
        return fallback;
    const int parsed = std::atoi(v);
    return parsed > 0 ? parsed : fallback;
}

int pick_split(int64_t group_elems, int ng, int sm_count, int64_t hw)
{
    static const int forced          = env_int("CSIG_GN_SPLIT", 0);
    static const int elems_per_block = env_int("CSIG_GN_ELEMS_PER_BLOCK", 32768);
    static const int blocks_per_sm   = env_int("CSIG_GN_BLOCKS_PER_SM", 4);
    if (forced > 0)
        return std::min<int64_t>(forced, hw);

    int64_t want = (group_elems + elems_per_block - 1) / elems_per_block;
    want         = std::max<int64_t>(want, static_cast<int64_t>(blocks_per_sm) * sm_count / std::max(ng, 1));
    int s        = 1;
    while (s * 2 <= want && s * 2 <= MAX_SPLIT && s * 2 <= hw)
        s <<= 1;
    return s;
}

}  // namespace

at::Tensor group_norm_act_cuda(const at::Tensor& x,
                               const at::Tensor& weight,
                               const at::Tensor& bias,
                               int64_t           num_groups,
                               double            eps,
                               int64_t           act)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "group_norm_act: all inputs must be CUDA");
    TORCH_CHECK(x.dim() == 4, "group_norm_act: x must be 4D (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "group_norm_act: x must be channels_last");
    TORCH_CHECK(num_groups > 0 && x.size(1) % num_groups == 0, "group_norm_act: C must be divisible by num_groups");
    TORCH_CHECK(weight.numel() == x.size(1) && bias.numel() == x.size(1),
                "group_norm_act: weight / bias must have C elements");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() && bias.scalar_type() == x.scalar_type(),
                "group_norm_act: weight / bias dtype must match x");
    TORCH_CHECK(act == 0 || act == 1, "group_norm_act: act must be 0 (none) or 1 (silu)");

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));

    const int64_t n      = x.size(0);
    const int     c      = static_cast<int>(x.size(1));
    const int64_t hw     = x.size(2) * x.size(3);
    const int     groups = static_cast<int>(num_groups);
    const int     cpg    = c / groups;

    auto y = at::empty_like(x, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    if (n == 0 || hw == 0)
        return y;

    const int   sm_cnt  = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int   split   = pick_split(static_cast<int64_t>(cpg) * hw, static_cast<int>(n) * groups, sm_cnt, hw);
    const int   ng      = static_cast<int>(n) * groups;
    auto        f32     = x.options().dtype(at::kFloat);
    auto        part    = at::empty({static_cast<int64_t>(ng) * 2 * split}, f32);
    auto        fin     = split > 1 ? at::empty({static_cast<int64_t>(ng) * 2}, f32) : part;
    const float inv_cnt = 1.f / static_cast<float>(static_cast<int64_t>(cpg) * hw);

    auto stream = at::cuda::getCurrentCUDAStream();
    // apply 的 grid.x 按元素总量切；上限 4096 块避免小图上启动过多空块。
    const int apply_blocks = static_cast<int>(std::min<int64_t>((hw * c + APPLY_THREADS - 1) / APPLY_THREADS, 4096));
    const size_t smem      = static_cast<size_t>(2 * c) * sizeof(float);

    DISPATCH_FP16_FP32(x.scalar_type(), "group_norm_act", [&] {
        gn_stats_kernel<scalar_t><<<dim3(ng, split), STATS_THREADS, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(x.data_ptr()), part.data_ptr<float>(), hw, c, cpg, groups, split);
        if (split > 1)
            gn_finalize_kernel<<<ng, 1, 0, stream>>>(part.data_ptr<float>(), fin.data_ptr<float>(), split);

        const auto*  xp = reinterpret_cast<const scalar_t*>(x.data_ptr());
        auto*        yp = reinterpret_cast<scalar_t*>(y.data_ptr());
        const auto*  wp = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const auto*  bp = reinterpret_cast<const scalar_t*>(bias.data_ptr());
        const dim3   grid(apply_blocks, static_cast<unsigned>(n));
        const float* fp = fin.data_ptr<float>();
        if (act == 1)
            gn_apply_kernel<scalar_t, 1><<<grid, APPLY_THREADS, smem, stream>>>(
                xp, yp, fp, wp, bp, hw, c, cpg, groups, inv_cnt, static_cast<float>(eps));
        else
            gn_apply_kernel<scalar_t, 0><<<grid, APPLY_THREADS, smem, stream>>>(
                xp, yp, fp, wp, bp, hw, c, cpg, groups, inv_cnt, static_cast<float>(eps));
    });
    return y;
}

TORCH_LIBRARY_IMPL(custom_op, CUDA, m)
{
    m.impl("group_norm_act", &group_norm_act_cuda);
}
