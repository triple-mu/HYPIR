// custom_op 四个算子共享的基础设施：dtype traits、向量化访存、warp/block 归约、dispatch 宏。
// 目标架构是 sm70(V100,服务器)与 sm86(本地功能性测试);故不用 cp.async / mma.16816 / bf16——
// Volta 都不支持。fp16 走 __half2 成对转换,fp32 直通。
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>

namespace custom_op {

// ---- dtype traits：统一 fp16 / fp32 的 float 互转 ----
template<typename T>
struct fp_traits;

template<>
struct fp_traits<__half> {
    __device__ __forceinline__ static float to_float(__half v)
    {
        return __half2float(v);
    }
    __device__ __forceinline__ static __half from_float(float v)
    {
        return __float2half_rn(v);
    }
};

template<>
struct fp_traits<float> {
    __device__ __forceinline__ static float to_float(float v)
    {
        return v;
    }
    __device__ __forceinline__ static float from_float(float v)
    {
        return v;
    }
};

// ---- 向量化访存：VEC 个元素打包成一次 load/store（fp16 用 VEC=8 即 128-bit）----
template<typename T, int VEC>
struct alignas(sizeof(T) * VEC) vec_t {
    T data[VEC];

    __device__ __forceinline__ void load(const T* p)
    {
        *this = *reinterpret_cast<const vec_t*>(p);
    }
    __device__ __forceinline__ void store(T* p) const
    {
        *reinterpret_cast<vec_t*>(p) = *this;
    }
};

// ---- 归约 ----
__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// block 内归约两个量（sum 与 sumsq 一起走，省一次 __syncthreads）。
// 归约顺序固定（warp shuffle + 0 号 warp 串行加），故结果确定性可复现。
template<int THREADS>
__device__ __forceinline__ void block_reduce_sum2(float& a, float& b, float* smem)
{
    constexpr int WARPS = THREADS / 32;
    const int     lane  = threadIdx.x & 31;
    const int     warp  = threadIdx.x >> 5;

    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
        smem[warp]         = a;
        smem[WARPS + warp] = b;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i) {
            sa += smem[i];
            sb += smem[WARPS + i];
        }
        smem[0]     = sa;
        smem[WARPS] = sb;
    }
    __syncthreads();
    a = smem[0];
    b = smem[WARPS];
}

}  // namespace custom_op

// ---- dispatch：只支持 fp16 与 fp32（sm70 无 bf16 tensor core，且推理链路不用 bf16）。
// 用法同 AT_DISPATCH_*：第三个参数传一个泛型 lambda，body 里可用 scalar_t。
#define DISPATCH_FP16_FP32(TYPE, NAME, ...)                                                        \
    [&] {                                                                                          \
        switch (TYPE) {                                                                            \
            case at::ScalarType::Half: {                                                           \
                using scalar_t = __half;                                                           \
                return __VA_ARGS__();                                                              \
            }                                                                                      \
            case at::ScalarType::Float: {                                                          \
                using scalar_t = float;                                                            \
                return __VA_ARGS__();                                                              \
            }                                                                                      \
            default:                                                                               \
                TORCH_CHECK(false, NAME ": unsupported dtype ", TYPE, " (only fp16 / fp32)");       \
        }                                                                                          \
    }()
