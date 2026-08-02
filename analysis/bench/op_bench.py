"""逐算子逐形状基准：pytorch / triton / cuda 三个后端的耗时与加速比。

用法：
    python csig_bench/op_bench.py              # 四个算子的全部生产形状
    python csig_bench/op_bench.py --op geglu   # 只跑一个
    python csig_bench/op_bench.py --e2e        # 额外跑 512x512 端到端注入对比
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model_dir"))

import csig_ops as ops  # noqa: E402


def timeit(fn, n=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def make_nhwc(c, hw, dtype):
    side = int(hw ** 0.5)
    while side > 1 and hw % side:
        side -= 1
    return torch.randn(1, c, side, hw // side, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last
    )


def cases(op):
    """(标签, 入参) 列表，形状取自 csig_ops.py 的静态调优表。"""
    dt = torch.float16
    if op == "group_norm_act":
        for hw, c in sorted(ops._GN_STATS_BEST.keys()):
            x = make_nhwc(c, hw, dt)
            w = torch.randn(c, device="cuda", dtype=dt)
            b = torch.randn(c, device="cuda", dtype=dt)
            yield f"C={c} HW={hw}", (x, w, b, 32, 1e-5, "silu")
    elif op == "geglu":
        for total, d in sorted(ops._GEGLU_BEST.keys()):
            yield f"rows={total // d} D={d}", (torch.randn(1, total // d, 2 * d, device="cuda", dtype=dt),)
    elif op == "sample_latent":
        for side in (32, 64, 128):
            mom = torch.randn(1, 8, side, side, device="cuda", dtype=dt).contiguous(
                memory_format=torch.channels_last
            )
            yield f"{side}x{side}", (mom, torch.randn(1, 4, side, side, device="cuda", dtype=dt))
    elif op == "attn_d512":
        for n in (256, 1024, 4096):
            yield f"N={n}", tuple(torch.randn(1, n, 512, device="cuda", dtype=dt) for _ in range(3))


def run_op(op):
    impls = {
        "pytorch": ops._EAGER[op],
        "triton": ops._TRITON.get(op),
        "cuda": ops._CUDA.get(op),
    }
    have = {k: v for k, v in impls.items() if v is not None}
    print(f"\n=== {op} ===")
    print(f"{'shape':<22s}" + "".join(f"{k + ' ms':>12s}" for k in have) + f"{'triton/cuda':>15s}")
    for label, args in cases(op):
        ms = {}
        for k, fn in have.items():
            try:
                ms[k] = timeit(lambda f=fn: f(*args))
            except Exception as e:  # 某后端在该形状不支持时不影响其余对比
                ms[k] = float("nan")
                print(f"  [{k}] {label} 失败: {type(e).__name__}: {str(e)[:60]}")
        row = f"{label:<22s}" + "".join(f"{ms[k]:12.3f}" for k in have)
        if "triton" in ms and "cuda" in ms and ms["cuda"] > 0:
            row += f"{ms['triton'] / ms['cuda']:14.2f}x"
        print(row)


def run_e2e():
    """512x512 端到端：三个后端各自最优布局下的时延与输出差异。"""
    import math

    model_dir = REPO_ROOT / "model_dir"
    if not (model_dir / "hypir_weights.pth").exists():
        print("\n[e2e] 跳过：找不到 model_dir/hypir_weights.pth")
        return
    print("\n=== e2e 512x512 ===")
    out = {}
    for backend in ops.available():
        os.environ["CSIG_OP_BACKEND"] = backend
        from runner import Runner  # 每次重新构造：布局与后端绑定，在 __init__ 里定
        r = Runner(str(model_dir))
        x = torch.rand(1, 3, 512, 512, device=r.device) * 2 - 1
        ms = timeit(lambda: r.infer(x), n=30, warmup=5)
        out[backend] = r.infer(x).float().cpu()
        print(f"  {backend:<8s} channels_last={str(r.channels_last):5s} {ms:8.2f} ms")
        del r
        torch.cuda.empty_cache()
    ref = out.get("pytorch")
    for backend, v in out.items():
        if ref is None or backend == "pytorch":
            continue
        mse = float(((v - ref) ** 2).mean())
        print(f"  {backend} vs pytorch: max={float((v - ref).abs().max()):.3e} "
              f"PSNR={10 * math.log10(4.0 / max(mse, 1e-20)):.2f} dB")


def main():
    ap = argparse.ArgumentParser(description="pytorch / triton / cuda 逐算子基准")
    ap.add_argument("--op", default="all", choices=["all", *ops.OP_NAMES])
    ap.add_argument("--e2e", action="store_true", help="额外跑 512x512 端到端三后端对比")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}  可用后端: {ops.available()}")

    for op in ops.OP_NAMES if args.op == "all" else [args.op]:
        run_op(op)
    if args.e2e:
        run_e2e()


if __name__ == "__main__":
    main()
