"""把 512x512 拆成更小的分块跑，看总耗时是否更低。

动机：VAE mid-block 与 UNet 的 self-attention 是 O(N^2)，N 是 latent token 数。
512x512 -> latent 64x64 -> N=4096；切成 4 块 256x256 后每块 N=1024，注意力总量降到 1/4。
卷积是线性的，总量不变；但小张量的 kernel 效率更低、launch 更多。净效应要实测。

分块必须按 batch 维打包（(4,3,256,256)）而不是串行跑 4 次，否则 launch 数翻 4 倍。
串行那一路也一起测，用来量这个差别。

**这不是提交方案**：非重叠分块会有接缝，且模型是按 512 patch 训/调的，画质必然变化。
这里只回答「速度上有没有搞头」。

用法：
    python csig_bench/tile_size_bench.py [--model-dir DIR] [--backend cuda]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parent.parent


def build(runner, ops, batch: int, size: int, seed: int):
    """返回处理 (batch,3,size,size) 的闭包：噪声与 cross-attn 的 K/V 都按 batch 铺好。"""
    lat = size // 8
    with torch.random.fork_rng(devices=[runner.device]):
        torch.manual_seed(seed)
        noise = torch.randn((batch, 4, lat, lat), device=runner.device,
                            dtype=runner.weight_dtype)
    text = runner._text_embed.expand(batch, -1, -1).contiguous()
    import runner as R
    with torch.no_grad():
        for m in runner.unet.modules():
            if isinstance(m, R.CrossAttention) and m.to_k.in_features != m.to_q.in_features:
                m._kv_cache = (m.to_k(text), m.to_v(text))

    @torch.no_grad()
    def run(x):
        z = ops.sample_latent(runner.vae.encode_moments(x), noise)
        z = runner._forward_generator(z.to(runner.weight_dtype), text)
        return runner.vae.decode(z.to(runner.weight_dtype)).float()

    return run


def timeit(fn, n=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description="分块尺寸对总耗时的影响")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--backend", type=str, default="")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    import os
    if args.backend:
        os.environ["CSIG_OP_BACKEND"] = args.backend
    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import csig_ops as ops
    from runner import Runner

    r = Runner(args.model_dir)
    print(f"{torch.cuda.get_device_name(0)} | 后端 {ops.BACKEND} "
          f"channels_last={r.channels_last}\n")
    mf = r._mf or torch.contiguous_format

    print(f"{'配置':22s}{'latent N':>10s}{'注意力相对量':>13s}{'ms':>9s}{'相对 512':>10s}")
    base = None
    for batch, size in ((1, 512), (4, 256), (16, 128), (64, 64)):
        x = torch.randn(batch, 3, size, size, device=r.device,
                        dtype=r.weight_dtype).contiguous(memory_format=mf)
        n_tok = (size // 8) ** 2
        attn_rel = batch * n_tok ** 2 / 4096 ** 2
        try:
            run = build(r, ops, batch, size, r.seed)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                ms = timeit(lambda: run(x), args.iters)
        except Exception as exc:
            print(f"{f'({batch},3,{size},{size})':22s}{n_tok:10d}  失败: {type(exc).__name__}: "
                  f"{str(exc)[:50]}")
            continue
        base = base or ms
        print(f"{f'({batch},3,{size},{size})':22s}{n_tok:10d}{attn_rel:13.3f}{ms:9.2f}"
              f"{base / ms:9.2f}x")

    # 串行 vs 打包：量 launch 开销的代价
    print("\n串行跑 4 块 256（不打 batch）作对照：")
    run1 = build(r, ops, 1, 256, r.seed)
    x1 = torch.randn(1, 3, 256, 256, device=r.device,
                     dtype=r.weight_dtype).contiguous(memory_format=mf)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        ms1 = timeit(lambda: [run1(x1) for _ in range(4)], args.iters)
    print(f"  4 x (1,3,256,256) 串行  {ms1:.2f} ms")


if __name__ == "__main__":
    main()
