"""残差 add 融进 group_norm_act 的收益基准。

nsys 显示 V100 上有 101 个 add/iter 紧跟着 `gn_stats`（合计 5.34 ms），全部来自
`ResnetBlock2D` 的 `return x + h` 之后下一块的 `gn_act(norm1, ·)`。融合前后的访存：

    分开   add(读x,读r,写s) + stats(读s) + apply(读s,写y)  = 6 遍
    融合   fused_stats(读x,读r,写s)      + apply(读s,写y)  = 5 遍

即省 1/6。按 add 的 5.34 ms 反推 1 遍约 1.78 ms —— 但这是模型推算，实测可能因寄存器压力
或 cache 命中而偏离，所以在这里用一个真的融合 kernel 量。

形状取 csig_ops._GN_STATS_BEST 里 HW 最大的几组（收益集中在 VAE 高分辨率段）。

用法：
    python csig_bench/fusion_bench.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
# 允许用 --model-dir 指到别处（服务器上 csig_ops.py 在 exp_new/）
_MD = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--model-dir"),
           str(REPO_ROOT / "model_dir"))
sys.path.insert(0, str(Path(_MD).resolve()))

import csig_ops as ops  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _fused_stats_kernel(X, R, S, PART, HW, C, cpg, rows_per_split, G, SPLIT,
                        BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr):
    """在算 GroupNorm 统计量的同时完成 s = x + r 并写出 s。

    与 csig_ops._gn_stats_kernel 的唯一差别：多读一个 R、多写一个 S。累加用融合后的 s。
    """
    pid_g = tl.program_id(0).to(tl.int64)
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
        off = base + rows[:, None] * C + cofs[None, :]
        x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R + off, mask=m, other=0.0).to(tl.float32)
        s = x + r
        tl.store(S + off, s.to(X.dtype.element_ty), mask=m)
        acc_s += s
        acc_q += s * s
    tl.store(PART + pid_g * 2 * SPLIT + pid_s, tl.sum(acc_s))
    tl.store(PART + pid_g * 2 * SPLIT + SPLIT + pid_s, tl.sum(acc_q))


FUSED_BEST = {}  # (HW, C) -> (BLOCK_HW, num_warps, num_stages)，--tune 扫出来的


def fused_group_norm_act(x, r, weight, bias, num_groups, eps, act, cfg_stats=None):
    """返回 (s, y)：s = x + r，y = SiLU(GroupNorm(s))。逐 kernel 复用 csig_ops 的实现。"""
    n, c, h, w = x.shape
    hw = h * w
    cpg = c // num_groups
    split, block_hw, block_c, nw = ops._gn_config(c, hw, cpg)
    ng = n * num_groups
    part = ops._gn_scratch("part", ng * 2 * split, x.device)
    # 融合版多一次读 + 一次写，最优配置与非融合版不同，不能直接套 _GN_STATS_BEST
    block_hw, nw, ns = (cfg_stats or FUSED_BEST.get((hw, c))
                        or ops._GN_STATS_BEST.get((hw, c), (block_hw, nw, 3)))
    s = torch.empty_like(x)
    _fused_stats_kernel[(ng, split)](x, r, s, part, hw, c, cpg, -(-hw // split), num_groups, split,
                                     BLOCK_HW=block_hw, BLOCK_C=block_c, num_warps=nw, num_stages=ns)
    if split > 1:
        fin = ops._gn_scratch("fin", ng * 2, x.device)
        ops._gn_finalize_kernel[(ng,)](part, fin, split, BLOCK_S=triton.next_power_of_2(split))
    else:
        fin = part
    y = torch.empty_like(x)
    cfg = ops._GN_APPLY_BEST.get((hw, c))
    if cfg is None:
        bc = min(256, triton.next_power_of_2(c))
        cfg = (max(1, 2048 // bc), bc, 4)
    bhw, bc, nw2 = cfg
    ops._gn_apply_kernel[(triton.cdiv(hw, bhw), triton.cdiv(c, bc), n)](
        s, y, fin, weight, bias, hw, c, cpg, num_groups, 1.0 / (hw * cpg), eps,
        ACT=1 if act == "silu" else 0, BLOCK_HW=bhw, BLOCK_C=bc, num_warps=nw2)
    return s, y


def timeit(fn, n=50, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description="add + group_norm_act 融合基准")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tune", action="store_true", help="为融合 kernel 单独扫配置")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    # 取 HW 最大的几组生产形状：可融合的 add 里 77% 的耗时集中在最大的 20 个上
    shapes = sorted(ops._GN_STATS_BEST.keys(), reverse=True)[:8]
    print(f"{torch.cuda.get_device_name(0)}  fp16 NHWC  GroupNorm(32)+SiLU\n")
    print(f"{'HW x C':16s}{'分开 ms':>10s}{'融合 ms':>10s}{'省':>9s}{'最大误差':>11s}")
    tot_a = tot_b = 0.0
    for hw, c in shapes:
        side = int(hw ** 0.5)
        while side > 1 and hw % side:
            side -= 1
        x = torch.randn(1, c, side, hw // side, device="cuda", dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        r = torch.randn_like(x)
        wt = torch.randn(c, device="cuda", dtype=torch.float16)
        bs = torch.randn(c, device="cuda", dtype=torch.float16)

        def sep():
            s = x + r
            return s, ops.triton_group_norm_act(s, wt, bs, 32, 1e-5, "silu")

        if args.tune:
            best, best_ms = None, float("inf")
            for bhw in (32, 64, 128, 256, 512, 1024, 2048):
                for nwarp in (2, 4, 8):
                    for nstage in (2, 3, 4):
                        try:
                            ms = timeit(lambda: fused_group_norm_act(
                                x, r, wt, bs, 32, 1e-5, "silu", (bhw, nwarp, nstage)), 20, 5)
                        except Exception:
                            continue
                        if ms < best_ms:
                            best, best_ms = (bhw, nwarp, nstage), ms
            FUSED_BEST[(hw, c)] = best
            print(f"  [tune] {hw}x{c} -> BLOCK_HW={best[0]} warps={best[1]} stages={best[2]}")

        fus = lambda: fused_group_norm_act(x, r, wt, bs, 32, 1e-5, "silu")
        (s0, y0), (s1, y1) = sep(), fus()
        err = max(float((s0 - s1).abs().max()), float((y0 - y1).abs().max()))
        a, b = timeit(sep, args.iters), timeit(fus, args.iters)
        tot_a += a
        tot_b += b
        print(f"{f'{hw} x {c}':16s}{a:10.3f}{b:10.3f}{(a - b) / a * 100:8.1f}%{err:11.2e}")
    print(f"{'合计':16s}{tot_a:10.3f}{tot_b:10.3f}{(tot_a - tot_b) / tot_a * 100:8.1f}%")


if __name__ == "__main__":
    main()
