"""512x512 评测路径的旋钮扫描（在评测机同型号 GPU 上跑，本地结果不作数）。

扫的都是会直接改变评测时延、进而改变综合分的选项：
    - 算子后端 pytorch / triton / cuda（各自绑定最优内存布局）
    - SDPA 后端优先级（V100 上 FLASH 不可用，实际会落到哪个）
    - cudnn.benchmark 开/关
再加两项诊断：分段耗时、CPU 下发 vs e2e（判断是否 launch-bound，决定 CUDA Graph 是否值得做）。

用法：
    python csig_bench/knob_sweep.py --model-dir model_dir
    python csig_bench/knob_sweep.py --model-dir model_dir --only sdpa
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parent.parent

SDPA_CHOICES = {
    "flash+eff": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
    "eff": [SDPBackend.EFFICIENT_ATTENTION],
    "math": [SDPBackend.MATH],
}
if hasattr(SDPBackend, "CUDNN_ATTENTION"):  # torch >= 2.5
    SDPA_CHOICES["cudnn+flash+eff"] = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                                       SDPBackend.EFFICIENT_ATTENTION]


def measure(fn, n: int = 30, warm: int = 10):
    """返回 (CPU 下发 ms, e2e ms)。每次迭代都从 idle GPU 起测。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    cpu = e2e = 0.0
    for _ in range(n):
        t = time.perf_counter()
        fn()
        cpu += time.perf_counter() - t
        torch.cuda.synchronize()
        e2e += time.perf_counter() - t
    return cpu / n * 1e3, e2e / n * 1e3


def build(model_dir: str, **cfg_override):
    """构造 Runner，可覆盖 config.yaml 里的键。算子后端由 CSIG_OP_BACKEND 决定。"""
    import runner as R
    base = R._load_config
    R._load_config = lambda d: {**base(d), **cfg_override}
    try:
        return R.Runner(model_dir)
    finally:
        R._load_config = base


def main() -> None:
    parser = argparse.ArgumentParser(description="512x512 旋钮扫描")
    parser.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--only", type=str, default="", help="只跑某一节: backend|sdpa|cudnn|segments")
    args = parser.parse_args()
    sel = lambda name: not args.only or args.only == name

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import csig_ops
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")

    r = build(args.model_dir)
    size = r.patch_size
    x = torch.randn(1, 3, size, size, device=r.device).clamp(-1, 1)

    if sel("backend"):
        print(f"== 算子后端（布局与后端绑定）==  可用: {csig_ops.available()}")
        for name in csig_ops.available():
            os.environ["CSIG_OP_BACKEND"] = name
            rr = build(args.model_dir)
            c, e = measure(lambda: rr.infer(x), n=args.iters)
            print(f"  {name:8s} channels_last={str(rr.channels_last):5s} "
                  f"CPU下发 {c:8.3f}  e2e {e:8.3f} ms")
            del rr
            torch.cuda.empty_cache()
        os.environ["CSIG_OP_BACKEND"] = csig_ops.BACKEND

    if sel("sdpa"):
        print("\n== SDPA 后端优先级 ==")
        xf = x.to(r.weight_dtype)
        if r._mf:
            xf = xf.contiguous(memory_format=r._mf)
        for name, backends in SDPA_CHOICES.items():
            try:
                with sdpa_kernel(backends):
                    c, e = measure(lambda: r._infer_tile(xf), n=args.iters)
                print(f"  {name:16s}  CPU下发 {c:8.3f}  e2e {e:8.3f} ms")
            except RuntimeError as exc:
                print(f"  {name:16s}  不可用: {str(exc).splitlines()[0][:70]}")

    if sel("cudnn"):
        print("\n== cudnn.benchmark ==")
        for bm in (True, False):
            torch.backends.cudnn.benchmark = bm
            c, e = measure(lambda: r.infer(x), n=args.iters, warm=20)
            print(f"  benchmark={str(bm):5s}      CPU下发 {c:8.3f}  e2e {e:8.3f} ms")
        torch.backends.cudnn.benchmark = True

    if sel("segments"):
        # 整条 infer 的下发时间总是接近 e2e（发射队列填满后的反压），判据看各段下发之和
        print("\n== 分段 ==")
        c, e2e = measure(lambda: r.infer(x), n=args.iters)
        print(f"  {'infer 整体':22s}  CPU下发 {c:8.3f}  e2e {e2e:8.3f} ms")
        xf = x.to(r.weight_dtype)
        if r._mf:
            xf = xf.contiguous(memory_format=r._mf)
        with torch.no_grad(), sdpa_kernel(SDPA_CHOICES["flash+eff"]):
            enc = lambda: csig_ops.sample_latent(r.vae.encode_moments(xf), r._tile_noise)
            z_lq = enc()
            unet = lambda: r._forward_generator(z_lq.to(r.weight_dtype), r._text_embed)
            z = unet().to(r.weight_dtype)
            dec = lambda: r.vae.decode(z).float()
            import runner as R
            wav = lambda: R.wavelet_reconstruction(dec().contiguous(), x).clamp(-1, 1)
            tot = 0.0
            for label, fn in (("VAE encode+sample", enc), ("UNet+ddpm", unet),
                              ("VAE decode", dec), ("decode+wavelet", wav)):
                c, e = measure(fn, n=args.iters)
                if label != "decode+wavelet":
                    tot += c
                print(f"  {label:22s}  CPU下发 {c:8.3f}  e2e {e:8.3f} ms")
        verdict = "launch-bound（CUDA Graph 有收益）" if tot > 0.8 * e2e else "GPU-bound（收益有限）"
        print(f"  分段下发合计 {tot:.2f} ms vs e2e {e2e:.2f} ms -> {verdict}")

    print(f"\n峰值显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
