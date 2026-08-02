"""四个融合算子的后端分发层。

同一组逻辑算子有三种实现，由环境变量 ``CSIG_OP_BACKEND`` 选择：

    pytorch   纯 eager PyTorch，无额外依赖
    triton    本文件内的 Triton 融合 kernel（随 torch 分发，运行时 JIT）
    cuda      custom_op_cpp_ext 编译出的 CUDA C++ 扩展（需预先编译好 _C*.so）
    auto      默认。可用的后端各实测一次 512x512 热路径，取最快的（在 Runner 里做）

**内存布局与后端绑定**：triton/cuda 的 kernel 都按 NHWC 寻址，必须 channels_last；
而纯 PyTorch 实测 NCHW 更快（V100 182.70->150.10 ms，3080Ti 224.52->192.50 ms，差距
全在 VAE）。所以布局不能独立于后端选，``CHANNELS_LAST`` 随 ``bind()`` 一起定。

runner.py ``import csig_ops as ops`` 后调用 ``ops.group_norm_act`` 等四个名字，
绑定在 ``bind()`` 时完成。
"""

import os

import torch
from torch.nn import functional as F
from typing import Optional

OP_NAMES = ("group_norm_act", "geglu", "sample_latent", "attn_d512")
BACKENDS = ("pytorch", "triton", "cuda")

# 每个后端的最优内存布局。triton/cuda 是硬要求（kernel 按 NHWC 寻址），pytorch 是实测结论
PREFERS_CHANNELS_LAST = {"pytorch": False, "triton": True, "cuda": True}


# ---------------------------------------------------------------------------
# 后端 1: 纯 PyTorch（同时作为其余后端不适用时的逐算子回退）
# ---------------------------------------------------------------------------

def eager_group_norm_act(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         num_groups: int, eps: float, act: Optional[str]) -> torch.Tensor:
    y = F.group_norm(x, num_groups, weight, bias, eps)
    return F.silu(y) if act == "silu" else y


def eager_geglu(y2: torch.Tensor) -> torch.Tensor:
    h, gate = y2.chunk(2, dim=-1)
    return h * F.gelu(gate)


def eager_sample_latent(moments: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    return mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * noise


def eager_attn_d512(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # (1, N, 512) 单头：补一个 head 维交给 SDPA
    return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]


_EAGER = {
    "group_norm_act": eager_group_norm_act,
    "geglu": eager_geglu,
    "sample_latent": eager_sample_latent,
    "attn_d512": eager_attn_d512,
}

# ---------------------------------------------------------------------------
# 后端 2: Triton 融合 kernel
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    # 这个 try 绝对不能去掉：少数 torch 构建不带 triton，裸 import 会让 csig_ops 连带
    # runner.py 一起 import 失败——评测脚本直接异常退出，没有补救余地。
    # 这里用桩顶住，让下面那些 @triton.jit / tl.constexpr 的**定义**能过（kernel 体不会
    # 被执行），_TRITON 置空后 available() 里不出现 triton 后端，桩函数永远不会被调用。
    HAS_TRITON = False

    class _TritonStub:
        def __getattr__(self, _name):
            return self

        def __call__(self, *args, **kwargs):
            return args[0] if args else self

    triton = tl = _TritonStub()



@triton.jit
def _gn_stats_kernel(X, PART, HW, C, cpg, rows_per_split, G, SPLIT,
                     BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr):
    # NHWC 布局：组 (n, g) 的元素位于 n*HW*C + hw*C + [g*cpg, (g+1)*cpg)
    pid_g = tl.program_id(0).to(tl.int64)  # n*G + g
    pid_s = tl.program_id(1)
    base = (pid_g // G) * HW * C + (pid_g % G) * cpg
    hw0 = pid_s * rows_per_split
    hw1 = tl.minimum(hw0 + rows_per_split, HW)
    cofs = tl.arange(0, BLOCK_C)
    cmask = cofs < cpg
    acc_s = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)
    acc_q = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)
    for hw in tl.range(hw0, hw1, BLOCK_HW):
        rows = hw + tl.arange(0, BLOCK_HW)
        m = (rows[:, None] < hw1) & cmask[None, :]
        x = tl.load(X + base + rows[:, None] * C + cofs[None, :], mask=m, other=0.0).to(tl.float32)
        acc_s += x
        acc_q += x * x
    tl.store(PART + pid_g * 2 * SPLIT + pid_s, tl.sum(acc_s))
    tl.store(PART + pid_g * 2 * SPLIT + SPLIT + pid_s, tl.sum(acc_q))


@triton.jit
def _gn_stats_nhwc_kernel(X, PART, HW, SPLIT, rows_per_split,
                          C: tl.constexpr, G: tl.constexpr, CPG: tl.constexpr,
                          BLOCK_HW: tl.constexpr):
    """cpg 小时用的统计量 kernel：按「整行 C 连续」分块，而不是按 (n,g) 分组。

    _gn_stats_kernel 每行只读 cpg 个 half，cpg=4(C=128) 时每次访存只有 8 字节落在 32B
    sector 里，实测带宽效率 0.35。这里一个 block 吃 BLOCK_HW 行 x 整个 C，行内完全连续，
    组内求和推迟到最后：先按通道累加成 (C,)，再 reshape 成 (G, CPG) 沿 CPG 归约。
    要求 C 是 2 的幂（C=128/256 满足，正好是 VAE 高分辨率段）。
    """
    t = tl.program_id(0)
    n = tl.program_id(1)
    hw0 = t * rows_per_split
    hw1 = tl.minimum(hw0 + rows_per_split, HW)
    cofs = tl.arange(0, C)
    # 累加器保持二维：行是线程内的外层维，跨线程归约只在循环外做一次。
    # 把 tl.sum 放进循环里会让每次迭代都走一遍 shared memory 归约，实测慢 10 倍
    acc_s = tl.zeros((BLOCK_HW, C), dtype=tl.float32)
    acc_q = tl.zeros((BLOCK_HW, C), dtype=tl.float32)
    for hw in tl.range(hw0, hw1, BLOCK_HW):
        rows = hw + tl.arange(0, BLOCK_HW)
        m = rows[:, None] < hw1
        x = tl.load(X + n * HW * C + rows[:, None] * C + cofs[None, :], mask=m,
                    other=0.0).to(tl.float32)
        acc_s += x
        acc_q += x * x
    s = tl.sum(tl.reshape(tl.sum(acc_s, axis=0), (G, CPG)), axis=1)
    q = tl.sum(tl.reshape(tl.sum(acc_q, axis=0), (G, CPG)), axis=1)
    gidx = (n * G + tl.arange(0, G)) * 2 * SPLIT
    tl.store(PART + gidx + t, s)
    tl.store(PART + gidx + SPLIT + t, q)


@triton.jit
def _gn_finalize_kernel(PART, FIN, SPLIT, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    m = offs < SPLIT
    tl.store(FIN + pid * 2, tl.sum(tl.load(PART + pid * 2 * SPLIT + offs, mask=m, other=0.0)))
    tl.store(FIN + pid * 2 + 1,
             tl.sum(tl.load(PART + pid * 2 * SPLIT + SPLIT + offs, mask=m, other=0.0)))


@triton.jit
def _gn_apply_kernel(X, Y, FIN, W, B, HW, C, cpg, G, inv_cnt, eps,
                     ACT: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr):
    # 3D grid（像素块, 通道块, batch）：轴与 NHWC 维度对齐，消掉 1D 平铺的逐元素
    # int64 div/mod 三连，mean/rstd/w/b 降为 per-column 向量 load（实测快 1.1-1.8x）。
    # 全 int32 地址算术（wrapper 保证 numel < 2^31）
    n = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    rmask = rows < HW
    cmask = cols < C
    gi = n * G + cols // cpg
    s = tl.load(FIN + gi * 2, mask=cmask, other=0.0)
    q = tl.load(FIN + gi * 2 + 1, mask=cmask, other=1.0)
    mean = s * inv_cnt
    rstd = tl.rsqrt(q * inv_cnt - mean * mean + eps)
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
    mask = rmask[:, None] & cmask[None, :]
    offs = n * HW * C + rows[:, None] * C + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean[None, :]) * rstd[None, :] * w[None, :] + b[None, :]
    if ACT == 1:  # silu，fp32 计算后一次舍入
        y = y * tl.sigmoid(y)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _geglu_kernel(X, Y, total, D, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // D
    col = offs % D
    h = tl.load(X + row * 2 * D + col, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(X + row * 2 * D + D + col, mask=mask, other=0.0).to(tl.float32)
    y = h * 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))  # erf-exact gelu
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _attn_d512_kernel(Q, K, V, O, N, S, scale,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    # 单 (batch,head)、非因果、D=512 的 split-D flash attention（FA2 online softmax），
    # 4 个 BLOCK_D=128 显式累加器避免物化 N×N 分数矩阵；SDPA 的 flash/cudnn 后端不支持
    # head_dim=512，mem_efficient 在该形状慢 ~1.5x。参考 xlite-dev/ffpa-attn 的 Split-D 思路。
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    q_ptr = Q + offs_m[:, None] * S + offs_d[None, :]
    for n0 in tl.range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_ptr = K + offs_n[:, None] * S + offs_d[None, :]
        s = tl.dot(tl.load(q_ptr), tl.trans(tl.load(k_ptr)))
        s += tl.dot(tl.load(q_ptr + BLOCK_D), tl.trans(tl.load(k_ptr + BLOCK_D)))
        s += tl.dot(tl.load(q_ptr + 2 * BLOCK_D), tl.trans(tl.load(k_ptr + 2 * BLOCK_D)))
        s += tl.dot(tl.load(q_ptr + 3 * BLOCK_D), tl.trans(tl.load(k_ptr + 3 * BLOCK_D)))
        s = s * scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(Q.dtype.element_ty)
        v_ptr = V + offs_n[:, None] * S + offs_d[None, :]
        acc0 = acc0 * alpha[:, None] + tl.dot(p16, tl.load(v_ptr))
        acc1 = acc1 * alpha[:, None] + tl.dot(p16, tl.load(v_ptr + BLOCK_D))
        acc2 = acc2 * alpha[:, None] + tl.dot(p16, tl.load(v_ptr + 2 * BLOCK_D))
        acc3 = acc3 * alpha[:, None] + tl.dot(p16, tl.load(v_ptr + 3 * BLOCK_D))
        m_i = m_new
    o_ptr = O + offs_m[:, None] * 512 + offs_d[None, :]
    tl.store(o_ptr, (acc0 / l_i[:, None]).to(O.dtype.element_ty))
    tl.store(o_ptr + BLOCK_D, (acc1 / l_i[:, None]).to(O.dtype.element_ty))
    tl.store(o_ptr + 2 * BLOCK_D, (acc2 / l_i[:, None]).to(O.dtype.element_ty))
    tl.store(o_ptr + 3 * BLOCK_D, (acc3 / l_i[:, None]).to(O.dtype.element_ty))


@triton.jit
def _sample_latent_kernel(MOM, NZ, Z, total, HW, BLOCK: tl.constexpr):
    # MOM/Z 为 NHWC；noise 为 NCHW（torch.randn 原始布局，保证 RNG 流与 eager 一致）
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    pix = offs // 4
    c = offs % 4
    mean = tl.load(MOM + pix * 8 + c, mask=mask, other=0.0).to(tl.float32)
    lv = tl.load(MOM + pix * 8 + 4 + c, mask=mask, other=0.0).to(tl.float32)
    lv = tl.minimum(tl.maximum(lv, -30.0), 20.0)
    nz = tl.load(NZ + (pix // HW * 4 + c) * HW + pix % HW, mask=mask, other=0.0).to(tl.float32)
    z = mean + tl.exp(0.5 * lv) * nz
    tl.store(Z + offs, z.to(Z.dtype.element_ty), mask=mask)


# RTX 3080 Ti (sm86, triton 3.7.1) 上对生产形状（patch 512/64 tile）的 autotune 结果，
# 作为静态默认；未命中形状回退下方的通用规则。
_GN_STATS_BEST = {  # (HW, C) -> (BLOCK_HW, num_warps, num_stages)
    (262144, 128): (128, 4, 2), (262144, 256): (512, 8, 4),
    (65536, 128): (256, 8, 2), (65536, 256): (32, 4, 4), (65536, 512): (128, 4, 2),
    (16384, 256): (512, 8, 2), (16384, 512): (32, 4, 4),
    (4096, 320): (1024, 8, 2), (4096, 512): (512, 8, 4), (4096, 640): (128, 4, 4),
    (4096, 960): (128, 4, 2),
    (1024, 320): (512, 8, 2), (1024, 640): (512, 8, 2), (1024, 960): (256, 8, 4),
    (1024, 1280): (256, 8, 2), (1024, 1920): (256, 8, 2),
    (256, 640): (256, 8, 4), (256, 1280): (256, 8, 4), (256, 1920): (128, 4, 2),
    (256, 2560): (32, 4, 4),
    (64, 1280): (128, 4, 2), (64, 2560): (32, 4, 2),
}
_GN_APPLY_BEST = {  # (HW, C) -> (BLOCK_HW, BLOCK_C, num_warps)
    (262144, 128): (4, 256, 4), (262144, 256): (8, 256, 8),
    (65536, 128): (8, 128, 8), (65536, 256): (64, 64, 4), (65536, 512): (4, 256, 8),
    (16384, 256): (64, 64, 4), (16384, 512): (32, 64, 8),
    (4096, 320): (64, 64, 4), (4096, 512): (64, 64, 8), (4096, 640): (64, 64, 4),
    (4096, 960): (64, 64, 8),
    (1024, 320): (32, 64, 4), (1024, 640): (64, 64, 8), (1024, 960): (32, 128, 8),
    (1024, 1280): (64, 64, 4), (1024, 1920): (16, 128, 4),
    (256, 640): (8, 128, 8), (256, 1280): (32, 64, 4), (256, 1920): (32, 128, 8),
    (256, 2560): (64, 32, 4),
    (64, 1280): (16, 128, 8), (64, 2560): (8, 256, 8),
}
_GEGLU_BEST = {  # (total, D) -> (BLOCK, num_warps, num_stages)
    (5242880, 1280): (2048, 8, 3), (2621440, 2560): (2048, 8, 3),
    (1310720, 5120): (1024, 4, 3), (327680, 5120): (512, 4, 3),
}
# _gn_stats_nhwc_kernel 的 (SPLIT, BLOCK_HW, num_warps)：单一配置而非逐形状表——
# 评测机型号未知，逐形状调优表只在调它的那张卡上成立。见 csig_bench/gn_stats_ab.py --uniform
_GN_STATS_NHWC = (256, 16, 4)
USE_COALESCED_STATS = True  # 置 False 可退回按 (n,g) 分组的旧 kernel，供基准对照
_SAMPLE_BEST = {16384: (256, 2, 3)}  # total -> (BLOCK, num_warps, num_stages)
_ATTN_BEST = {4096: (16, 32, 4, 1)}  # N -> (BLOCK_M, BLOCK_N, num_warps, num_stages)

_GN_SCRATCH = {}


def _gn_config(C: int, HW: int, cpg: int) -> tuple[int, int, int, int]:
    BLOCK_C = triton.next_power_of_2(cpg)
    BLOCK_HW = max(1, min(4096 // BLOCK_C, triton.next_power_of_2(HW)))
    group_elems = cpg * HW
    # 上限 16 会把大形状压在 512 个块上——V100 实测 (262144,128) 用 split=256 比 16 快 1.46x，
    # 而 HW<=4096 的小形状卡在 0.045ms 的启动地板、split 开大反而更慢，故按 group_elems 缩放
    SPLIT = min(128, max(1, triton.next_power_of_2(-(-group_elems // 8192))))
    num_warps = 8 if BLOCK_HW * BLOCK_C >= 4096 else 4
    return SPLIT, BLOCK_HW, BLOCK_C, num_warps


def _gn_scratch(key: str, numel: int, device: torch.device) -> torch.Tensor:
    buf = _GN_SCRATCH.get((key, device))
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.float32, device=device)
        _GN_SCRATCH[(key, device)] = buf
    return buf


def triton_group_norm_act(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          num_groups: int, eps: float, act: Optional[str]) -> torch.Tensor:
    n, c, h, w = x.shape
    hw = h * w
    cpg = c // num_groups
    assert x.numel() < 2 ** 31
    ng = n * num_groups
    # 每行只有 cpg*2 字节连续，cpg<16 时不足一个 32B sector（sm70 上实测带宽效率掉到
    # 0.32~0.35），改走按整行 C 连续分块的版本
    coalesced = USE_COALESCED_STATS and cpg < 16 and c & (c - 1) == 0
    if coalesced:
        split, bhw, nwc = _GN_STATS_NHWC
        part = _gn_scratch("part", ng * 2 * split, x.device)
        _gn_stats_nhwc_kernel[(split, n)](x, part, hw, split, -(-hw // split),
                                          C=c, G=num_groups, CPG=cpg, BLOCK_HW=bhw,
                                          num_warps=nwc)
    else:
        split, block_hw, block_c, nw = _gn_config(c, hw, cpg)
        part = _gn_scratch("part", ng * 2 * split, x.device)
        block_hw, nw, ns = _GN_STATS_BEST.get((hw, c), (block_hw, nw, 3))
        _gn_stats_kernel[(ng, split)](x, part, hw, c, cpg, -(-hw // split), num_groups, split,
                                      BLOCK_HW=block_hw, BLOCK_C=block_c, num_warps=nw,
                                      num_stages=ns)
    if split > 1:
        fin = _gn_scratch("fin", ng * 2, x.device)
        _gn_finalize_kernel[(ng,)](part, fin, split, BLOCK_S=triton.next_power_of_2(split))
    else:
        fin = part
    y = torch.empty_like(x)
    cfg = _GN_APPLY_BEST.get((hw, c))
    if cfg is None:
        bc = min(256, triton.next_power_of_2(c))
        cfg = (max(1, 2048 // bc), bc, 4)
    bhw, bc, nw2 = cfg
    _gn_apply_kernel[(triton.cdiv(hw, bhw), triton.cdiv(c, bc), n)](
        x, y, fin, weight, bias, hw, c, cpg, num_groups, 1.0 / (hw * cpg), eps,
        ACT=1 if act == "silu" else 0, BLOCK_HW=bhw, BLOCK_C=bc, num_warps=nw2)
    return y


def triton_geglu(y2: torch.Tensor) -> torch.Tensor:
    d = y2.shape[-1] // 2
    out = torch.empty(y2.shape[:-1] + (d,), dtype=y2.dtype, device=y2.device)
    total = out.numel()
    blk, nw, ns = _GEGLU_BEST.get((total, d), (1024, 4, 3))
    _geglu_kernel[(triton.cdiv(total, blk),)](y2, out, total, d, BLOCK=blk,
                                              num_warps=nw, num_stages=ns)
    return out


def triton_sample_latent(moments: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    n, c8, h, w = moments.shape
    z = torch.empty((n, c8 // 2, h, w), dtype=moments.dtype, device=moments.device,
                    memory_format=torch.channels_last)
    total = z.numel()
    blk, nw, ns = _SAMPLE_BEST.get(total, (1024, 4, 3))
    _sample_latent_kernel[(triton.cdiv(total, blk),)](moments, noise, z, total, h * w,
                                                      BLOCK=blk, num_warps=nw, num_stages=ns)
    return z


def triton_attn_d512_is_correct(device: torch.device, dtype: torch.dtype) -> bool:
    """一次性自检 triton_attn_d512 是否可信（约 1 ms）。

    该 kernel 在 Volta(sm70) 上算不对：实测 N=64..4096 时与 fp32 SDPA 参考的最大误差达
    0.13~0.78，而输出本身量级仅 0.03，即结果无效；同时它在 sm70 上还比 SDPA 慢 16%。
    评测机 GPU 型号未知，硬编码架构门槛不可靠，故直接用一个小输入实测。
    """
    with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        q, k, v = (torch.randn(1, 64, 512, device=device, dtype=dtype) for _ in range(3))
        ref = F.scaled_dot_product_attention(q[:, None].float(), k[:, None].float(), v[:, None].float())[:, 0]
        try:
            got = triton_attn_d512(q, k, v)
        except Exception:
            return False
        return bool(torch.isfinite(got).all() and (got.float() - ref).abs().max() < 2e-2)


def triton_attn_d512(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q/k/v: (1, N, 512) 16-bit，最后一维连续；行 stride 可为 512 或 1536（融合 QKV 的 chunk 视图）。
    # N % 64 == 0；fp32 的 tl.dot 走 TF32，精度不适用
    n = q.shape[-2]
    s = q.stride(-2)
    assert q.stride(-1) == 1 and k.stride(-2) == s and v.stride(-2) == s
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    bm, bn, nw, ns = _ATTN_BEST.get(n, (32, 32, 4, 2))
    _attn_d512_kernel[(n // bm,)](q, k, v, o, n, s, 512 ** -0.5, BLOCK_M=bm, BLOCK_N=bn,
                                  BLOCK_D=128, num_warps=nw, num_stages=ns)
    return o


_TRITON = {
    "group_norm_act": triton_group_norm_act,
    "geglu": triton_geglu,
    "sample_latent": triton_sample_latent,
    "attn_d512": triton_attn_d512,
} if HAS_TRITON else {}

# ---------------------------------------------------------------------------
# 后端 3: CUDA C++ 扩展（custom_op_cpp_ext）
# ---------------------------------------------------------------------------

CUDA_LOAD_ERROR = None  # custom_op 加载失败的原因，None 表示成功

try:
    import custom_op as _custom

    _CUDA = {name: getattr(_custom, name) for name in OP_NAMES}
except Exception as exc:
    # 这个 try 绝对不能去掉：预编译的 _C*.so 与评测机的 Python / torch / CUDA / 架构
    # 任一不匹配就会抛（ImportError 或 OSError），裸 import 会让 csig_ops 连带 runner.py
    # 一起 import 失败——评测脚本直接异常退出，没有任何补救余地（PEP 604 那次已经吃过一回）。
    # 失败时 available() 里不出现 cuda 后端，auto 自动退到 triton；原因记在 CUDA_LOAD_ERROR
    # 里而不是静默吞掉，便于在目标机上排查。
    CUDA_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    _CUDA = {}

_IMPLS = {"pytorch": _EAGER, "triton": _TRITON, "cuda": _CUDA}


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------

def _guard_nhwc(fn, fallback):
    """kernel 版只在 4D channels_last 上有效，其余形状/布局退回 eager。"""

    def wrapped(x, *args):
        if x.dim() == 4 and x.is_cuda and x.is_contiguous(memory_format=torch.channels_last):
            return fn(x, *args)
        return fallback(x, *args)

    return wrapped


def _guard_attn(fn):
    """triton 版要求 batch=1、最后一维连续、qkv 行 stride 相同（融合 QKV 的 chunk 视图满足）。

    batch 必须显式挡掉：kernel 的 grid 只有 (N/BLOCK_M,)、寻址里没有 batch 维，batch>1 时
    只会算第 0 个，其余留在 torch.empty 的未初始化内存里——静默出错，比报错危险得多。
    """

    def wrapped(q, k, v):
        if (q.is_cuda and q.shape[0] == 1 and q.stride(-1) == 1
                and k.stride(-2) == q.stride(-2)
                and v.stride(-2) == q.stride(-2) and q.shape[-2] % 64 == 0):
            return fn(q, k, v)
        return eager_attn_d512(q, k, v)

    return wrapped


_attn_checked = {}


def _triton_attn_is_usable(device: torch.device, dtype: torch.dtype) -> bool:
    """triton_attn_d512 在 Volta(sm70) 上算错（误差 0.13~0.78 而输出量级仅 0.03，端到端
    PSNR 42.55->27.32）且比 SDPA 慢 16%。评测机型号未知，硬编码架构门槛不可靠，故实测。"""
    key = (str(device), dtype)
    if key not in _attn_checked:
        _attn_checked[key] = bool(triton_attn_d512_is_correct(device, dtype))
    return _attn_checked[key]


def available(device: Optional[torch.device] = None) -> tuple:
    """当前环境能跑起来的后端，按 BACKENDS 顺序。"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return ("pytorch",)
    return tuple(b for b in BACKENDS if _IMPLS[b])


BACKEND = "pytorch"
CHANNELS_LAST = False
group_norm_act, geglu, sample_latent, attn_d512 = (_EAGER[n] for n in OP_NAMES)


def bind(name: str, device: Optional[torch.device] = None) -> bool:
    """把四个算子绑定到指定后端，返回该后端要用的 channels_last。

    单个算子在该后端缺失或自检不过时退回 eager（目前只有 triton 的 attn_d512 会触发）。
    """
    global BACKEND, CHANNELS_LAST, group_norm_act, geglu, sample_latent, attn_d512
    assert name in BACKENDS, f"未知后端 {name}，可选 {BACKENDS}"
    # 后端不可用时必须报错，不能悄悄绑成 eager：CHANNELS_LAST 会被设成该后端的偏好
    # （triton/cuda 都是 NHWC），而 eager+NHWC 是最差组合（V100 上 182ms vs eager+NCHW 149ms）
    assert name == "pytorch" or _IMPLS[name], (
        f"后端 {name} 不可用" + (f"：{CUDA_LOAD_ERROR}" if name == "cuda" and CUDA_LOAD_ERROR
                                 else f"（HAS_TRITON={HAS_TRITON}）"))
    impls = dict(_EAGER)
    if name != "pytorch":
        for op, fn in _IMPLS[name].items():
            impls[op] = _guard_attn(fn) if op == "attn_d512" else _guard_nhwc(fn, _EAGER[op])
        if name == "triton" and device is not None and device.type == "cuda":
            if not _triton_attn_is_usable(device, torch.float16):
                impls["attn_d512"] = eager_attn_d512
    group_norm_act = impls["group_norm_act"]
    geglu = impls["geglu"]
    sample_latent = impls["sample_latent"]
    attn_d512 = impls["attn_d512"]
    BACKEND = name
    CHANNELS_LAST = PREFERS_CHANNELS_LAST[name]
    return CHANNELS_LAST


def requested(default: str = "auto") -> str:
    """后端选择：环境变量 CSIG_OP_BACKEND 优先，其次 config.yaml 的 op_backend。

    auto 表示由调用方实测选优。
    """
    val = os.environ.get("CSIG_OP_BACKEND", str(default)).strip().lower()
    assert val in BACKENDS + ("auto",), f"CSIG_OP_BACKEND={val} 无效，可选 {BACKENDS + ('auto',)}"
    return val
