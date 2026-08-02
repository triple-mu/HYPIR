"""两个 GroupNorm 统计量 kernel 的直接对拍与计时。

    old   _gn_stats_kernel        按 (n,g) 分组，每行只读 cpg 个 half
    new   _gn_stats_nhwc_kernel   按整行 C 连续分块，组内求和推迟到最后

用法：
    python csig_bench/gn_stats_ab.py [--model-dir DIR] [--tune]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
_MD = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--model-dir"),
           str(REPO_ROOT / "model_dir"))
sys.path.insert(0, str(Path(_MD).resolve()))

import csig_ops as ops  # noqa: E402
import triton  # noqa: E402


def run_old(x, num_groups=32, force_split=None):
    n, c, h, w = x.shape
    hw = h * w
    cpg = c // num_groups
    split, block_hw, block_c, nw = ops._gn_config(c, hw, cpg)
    if force_split:
        split = force_split
    part = torch.empty(n * num_groups * 2 * split, dtype=torch.float32, device=x.device)
    block_hw, nw, ns = ops._GN_STATS_BEST.get((hw, c), (block_hw, nw, 3))
    ops._gn_stats_kernel[(n * num_groups, split)](
        x, part, hw, c, cpg, -(-hw // split), num_groups, split,
        BLOCK_HW=block_hw, BLOCK_C=block_c, num_warps=nw, num_stages=ns)
    return part, split


def run_new(x, split, bhw, nw, num_groups=32):
    n, c, h, w = x.shape
    hw = h * w
    part = torch.empty(n * num_groups * 2 * split, dtype=torch.float32, device=x.device)
    ops._gn_stats_nhwc_kernel[(split, n)](
        x, part, hw, split, -(-hw // split),
        C=c, G=num_groups, CPG=c // num_groups, BLOCK_HW=bhw, num_warps=nw)
    return part


def reduce_part(part, ng, split):
    """(ng, 2, split) -> 每组的 (sum, sumsq)，供两条路径的结果比对。"""
    v = part.view(ng, 2, split).float()
    return v.sum(-1)


def timeit(fn, n=100, warm=25):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def sweep_uniform(shapes, iters):
    """评测机型号未知，逐形状调优表不可移植；这里找一个在所有形状上总耗时最小的单配置。"""
    xs = []
    for hw, c in shapes:
        side = int(hw ** 0.5)
        while side > 1 and hw % side:
            side -= 1
        xs.append((hw, c, torch.randn(1, c, side, hw // side, device="cuda",
                                      dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)))
    base = sum(timeit(lambda x=x: run_old(x), iters) for _, _, x in xs)
    print(f"旧 kernel（_gn_config 的 split，<=16）合计 {base:.3f} ms")
    print("旧 kernel 换更大的 split —— 检验收益到底来自访存合并还是占用率：")
    for sp in (16, 32, 64, 128, 256, 512):
        t = sum(timeit(lambda x=x, s=sp: run_old(x, force_split=s), 30, 10) for _, _, x in xs)
        print(f"    split={sp:4d}  {t:7.3f} ms  ({base / t:.2f}x)")
    print()
    rows = []
    for sp in (16, 32, 64, 128, 256):
        for bhw in (2, 4, 8, 16):
            for nw in (4, 8):
                try:
                    tot = sum(timeit(lambda x=x: run_new(x, sp, bhw, nw), 30, 10) for _, _, x in xs)
                except Exception:
                    continue
                rows.append((tot, (sp, bhw, nw)))
    rows.sort()
    print(f"{'配置':22s}{'合计 ms':>10s}{'相对旧':>9s}")
    for tot, cfg in rows[:8]:
        print(f"{str(cfg):22s}{tot:10.3f}{base / tot:8.2f}x")
    print(f"\n最优单配置 _GN_STATS_NHWC = {rows[0][1]}  ({base / rows[0][0]:.2f}x)")


def main() -> None:
    ap = argparse.ArgumentParser(description="gn_stats 新旧对拍")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tune", action="store_true", help="逐形状调优")
    ap.add_argument("--uniform", action="store_true", help="扫单一配置在所有形状上的总耗时")
    args = ap.parse_args()

    # 只测 cpg<16 且 C 是 2 的幂的形状——新 kernel 的适用范围
    shapes = [(hw, c) for hw, c in sorted(ops._GN_STATS_BEST.keys(), reverse=True)
              if c // 32 < 16 and c & (c - 1) == 0]
    print(f"{torch.cuda.get_device_name(0)}  fp16 NHWC\n")
    print(f"{'HW x C':15s}{'cpg':>4s}{'copy':>8s}{'old':>8s}{'new':>8s}{'加速':>8s}"
          f"{'新效率':>9s}{'误差':>11s}  cfg")
    if args.uniform:
        return sweep_uniform(shapes, args.iters)
    best_tbl = {}
    tot_old = tot_new = 0.0
    for hw, c in shapes:
        side = int(hw ** 0.5)
        while side > 1 and hw % side:
            side -= 1
        x = torch.randn(1, c, side, hw // side, device="cuda", dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        copy_ms = timeit(lambda: x.clone(), args.iters)   # 读1写1
        p_old, split_old = run_old(x)
        t_old = timeit(lambda: run_old(x), args.iters)

        cands = ([(sp, bhw, nw) for sp in (8, 16, 32, 64, 128) for bhw in (2, 4, 8, 16)
                  for nw in (4, 8)] if args.tune else
                 [(32, 4, 4), (32, 8, 4), (64, 4, 8), (16, 8, 4)])
        best, best_ms = None, float("inf")
        for sp, bhw, nw in cands:
            try:
                ms = timeit(lambda: run_new(x, sp, bhw, nw), 30, 10)
            except Exception:
                continue
            if ms < best_ms:
                best, best_ms = (sp, bhw, nw), ms
        best_tbl[(hw, c)] = best
        p_new = run_new(x, *best)
        ref = reduce_part(p_old, 32, split_old)
        got = reduce_part(p_new, 32, best[0])
        err = float(((got - ref).abs() / ref.abs().clamp_min(1e-6)).max())
        eff = (x.numel() * 2 / (best_ms / 1e3)) / (2 * x.numel() * 2 / (copy_ms / 1e3))
        tot_old += t_old
        tot_new += best_ms
        print(f"{f'{hw} x {c}':15s}{c // 32:4d}{copy_ms:8.3f}{t_old:8.3f}{best_ms:8.3f}"
              f"{t_old / best_ms:7.2f}x{eff:9.2f}{err:11.2e}  {best}")
    print(f"\n{'合计':15s}{'':12s}{tot_old:8.3f}{tot_new:8.3f}{tot_old / tot_new:7.2f}x")
    if args.tune:
        print("\n_GN_STATS_NHWC_BEST = {")
        for k, v in best_tbl.items():
            print(f"    {k}: {v},")
        print("}")


if __name__ == "__main__":
    main()
