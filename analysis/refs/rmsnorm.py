import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
        x_ptr,
        h1_ptr,
        w_ptr,
        eps,
        stride,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        INCLUDE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + cols, mask=cols < N_COLS, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _mean += a * a
    rstd = tl.math.rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)


def _rms_norm_forward(x, attn_norm_weights, eps):
    if not x.is_contiguous():
        raise ValueError("data must be contiguous")
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError("weights must be contiguous")
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        _rms_norm_kernel[(M,)](
            x_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            INCLUDE_WEIGHT=attn_norm_weights is not None,
        )
    return out
