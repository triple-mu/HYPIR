"""GroupNorm 三个 kernel 的实测带宽 vs roofline。

诊断用：`_gn_stats_kernel` 按 (n, g) 分组遍历，NHWC 下每行只读 cpg 个 half。cpg = C/32，
所以 C=128 时每行只有 8 字节连续 —— 32B sector 利用率 25%，访存效率被打到 1/4。而 C=128/256
恰好是 VAE 高分辨率段（512x512、256x256）的形状，也是 GroupNorm 耗时的大头。

对每个生产形状量四条：
    copy      纯拷贝，作为该形状的 roofline 参考（读 1 写 1）
    add       s = x + r（读 2 写 1），完全合并访存
    stats     _gn_stats_kernel + finalize（读 1）
    apply     _gn_apply_kernel（读 1 写 1，已经是 2D 分块、合并的）

「有效字节 / 实测耗时」除以 copy 的带宽即为效率，低于 1 的部分就是 tiling 浪费掉的。

用法：
    python csig_bench/gn_bandwidth.py [--model-dir DIR]
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


def timeit(fn, n=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def stats_only(x, num_groups=32):
    """只跑统计量那两个 kernel，不做 apply。"""
    n, c, h, w = x.shape
    hw = h * w
    cpg = c // num_groups
    split, block_hw, block_c, nw = ops._gn_config(c, hw, cpg)
    ng = n * num_groups
    part = ops._gn_scratch("part", ng * 2 * split, x.device)
    block_hw, nw, ns = ops._GN_STATS_BEST.get((hw, c), (block_hw, nw, 3))
    ops._gn_stats_kernel[(ng, split)](x, part, hw, c, cpg, -(-hw // split), num_groups, split,
                                      BLOCK_HW=block_hw, BLOCK_C=block_c, num_warps=nw,
                                      num_stages=ns)
    if split > 1:
        fin = ops._gn_scratch("fin", ng * 2, x.device)
        ops._gn_finalize_kernel[(ng,)](part, fin, split, BLOCK_S=triton.next_power_of_2(split))
        return fin
    return part


def apply_only(x, fin, weight, bias, num_groups=32, eps=1e-5):
    n, c, h, w = x.shape
    hw = h * w
    cpg = c // num_groups
    y = torch.empty_like(x)
    cfg = ops._GN_APPLY_BEST.get((hw, c))
    if cfg is None:
        bc = min(256, triton.next_power_of_2(c))
        cfg = (max(1, 2048 // bc), bc, 4)
    bhw, bc, nw2 = cfg
    ops._gn_apply_kernel[(triton.cdiv(hw, bhw), triton.cdiv(c, bc), n)](
        x, y, fin, weight, bias, hw, c, cpg, num_groups, 1.0 / (hw * cpg), eps,
        ACT=1, BLOCK_HW=bhw, BLOCK_C=bc, num_warps=nw2)
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description="GroupNorm kernel 带宽诊断")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    shapes = sorted(ops._GN_STATS_BEST.keys(), reverse=True)[:10]
    print(f"{torch.cuda.get_device_name(0)}  fp16 NHWC  GroupNorm(32)\n")
    print(f"{'HW x C':15s}{'cpg':>4s}{'MB':>7s}{'copy':>8s}{'add':>8s}{'stats':>8s}{'apply':>8s}"
          f"{'  | 效率(相对 copy 带宽)':s}")
    tot = {"copy": 0.0, "add": 0.0, "stats": 0.0, "apply": 0.0}
    for hw, c in shapes:
        side = int(hw ** 0.5)
        while side > 1 and hw % side:
            side -= 1
        x = torch.randn(1, c, side, hw // side, device="cuda", dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)
        r = torch.randn_like(x)
        wt = torch.randn(c, device="cuda", dtype=torch.float16)
        bs = torch.randn(c, device="cuda", dtype=torch.float16)
        mb = x.numel() * 2 / 1024 ** 2

        ms = {"copy": timeit(lambda: x.clone(), args.iters),
              "add": timeit(lambda: x + r, args.iters),
              "stats": timeit(lambda: stats_only(x), args.iters)}
        fin = stats_only(x)
        ms["apply"] = timeit(lambda: apply_only(x, fin, wt, bs), args.iters)
        for k, v in ms.items():
            tot[k] += v

        # 有效字节数：copy 读1写1、add 读2写1、stats 读1、apply 读1写1
        passes = {"copy": 2, "add": 3, "stats": 1, "apply": 2}
        bw = {k: passes[k] * x.numel() * 2 / (ms[k] / 1e3) / 1e9 for k in ms}
        eff = {k: bw[k] / bw["copy"] for k in ms}
        print(f"{f'{hw} x {c}':15s}{c // 32:4d}{mb:7.1f}"
              + "".join(f"{ms[k]:8.3f}" for k in ("copy", "add", "stats", "apply"))
              + "  | " + "  ".join(f"{k} {eff[k]:.2f}" for k in ("add", "stats", "apply")))
    print(f"\n{'合计 ms':15s}{'':11s}"
          + "".join(f"{tot[k]:8.3f}" for k in ("copy", "add", "stats", "apply")))


if __name__ == "__main__":
    main()
