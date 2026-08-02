"""512x512 评测路径的时延探针（不进提交包）。

除 e2e 墙钟外，还量 **CPU 下发时间**：每次调用前 GPU 已 idle，测函数返回所需的时间。

判据要看**分段**：整条 infer 的下发时间总是接近 e2e，那是发射队列填满后的反压，不代表
主机侧是瓶颈。真正的判据是「各段下发时间之和」对 e2e——之和远小于 e2e 才是 GPU-bound，
接近才是 launch-bound（此时 CUDA Graph 捕获才有收益）。

用法：
    python csig_bench/latency_probe.py --model-dir model_dir
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def main() -> None:
    parser = argparse.ArgumentParser(description="512x512 时延探针")
    parser.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import csig_ops
    from runner import Runner

    t0 = time.time()
    runner = Runner(args.model_dir)
    size = runner.patch_size
    print(f"{torch.cuda.get_device_name(0)} | 加载 {time.time() - t0:.1f}s | patch {size} "
          f"| 后端 {csig_ops.BACKEND} channels_last={runner.channels_last}")

    x = torch.randn(1, 3, size, size, device=runner.device).clamp(-1, 1)
    cpu, e2e = measure(lambda: runner.infer(x), n=args.iters)
    print(f"\ninfer({size})            CPU下发 {cpu:8.3f}  e2e {e2e:8.3f} ms")

    # 分段：全部走热路径的输入，避免 tiled 包装干扰
    xf = x.to(runner.weight_dtype)
    if runner._mf:
        xf = xf.contiguous(memory_format=runner._mf)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        enc = lambda: csig_ops.sample_latent(runner.vae.encode_moments(xf), runner._tile_noise)
        z_lq = enc()
        unet = lambda: runner._forward_generator(z_lq.to(runner.weight_dtype), runner._text_embed)
        z = unet().to(runner.weight_dtype)
        dec = lambda: runner.vae.decode(z).float()
        tot = 0.0
        for label, fn in (("VAE encode+sample", enc), ("UNet+ddpm", unet), ("VAE decode", dec)):
            c, e = measure(fn, n=args.iters)
            tot += c
            print(f"  {label:20s}  CPU下发 {c:8.3f}  e2e {e:8.3f} ms")
    verdict = "launch-bound（CUDA Graph 有收益）" if tot > 0.8 * e2e else "GPU-bound（CUDA Graph 收益有限）"
    print(f"\n分段下发合计 {tot:.2f} ms vs e2e {e2e:.2f} ms -> {verdict}")

    print(f"\n峰值显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
