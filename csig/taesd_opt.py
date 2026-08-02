"""TAESD 那条路能优化多少（V100）。

换完 VAE 后 UNet 占 80%，TAESD 只占 22%——这里量的是那 22% 里还剩多少肉，
顺便确认几个「以为已经生效、其实没有」的点：

  1. channels_last：runner 的 latent 是 channels_last，而 AutoencoderTiny 是
     默认 contiguous 加载的，中间可能白挨一次布局转换
  2. 上采样折叠：TAESD 解码器里是 Upsample(2) + conv(64,64,bias=False)，
     与 SD VAE 里那个 nearest+3x3conv 完全同构，runner 已有的 fold_upsample_conv
     可以直接套（乘加降到 4/9）
  3. CUDA Graph：SD 那条路实测只省 0.4 ms（算力受限）；TAESD kernel 小而多，
     启动开销占比可能完全不同

    python csig/taesd_opt.py --model-dir model_dir
"""
import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn as nn


def bench(fn, n=100, w=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def fold_upsample(dec, mf):
    """把 Upsample(2) + conv(k3,s1,p1,bias=False) 折成 ConvTranspose(k4,s2,p1)。

    nearest 上采样等价于把每个输入像素复制到 2x2；随后的 3x3 卷积可以吸收进
    一个 4x4 步长 2 的转置卷积，数学等价而乘加降到 4/9。runner 里给 SD VAE 用的
    是同一套变换。
    """
    mods = list(dec)
    out, i = [], 0
    n_fold = 0
    while i < len(mods):
        if (isinstance(mods[i], nn.Upsample) and i + 1 < len(mods)
                and isinstance(mods[i + 1], nn.Conv2d) and mods[i + 1].kernel_size == (3, 3)):
            c = mods[i + 1]
            ct = nn.ConvTranspose2d(c.in_channels, c.out_channels, 4, 2, 1,
                                    bias=c.bias is not None).to(c.weight.device, c.weight.dtype)
            with torch.no_grad():
                # nearest 复制 2x2 后做 3x3 卷积 = 4x4 转置卷积，权重按 2x2 平铺累加
                w = c.weight                       # (o,i,3,3)
                W = torch.zeros(c.in_channels, c.out_channels, 4, 4,
                                device=w.device, dtype=w.dtype)
                for dy in range(2):
                    for dx in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                oy, ox = ky + dy, kx + dx
                                if oy < 4 and ox < 4:
                                    W[:, :, oy, ox] += w[:, :, ky, kx].t()
                ct.weight.copy_(W)
                if c.bias is not None:
                    ct.bias.copy_(c.bias)
            out.append(ct.to(memory_format=mf) if mf else ct)
            n_fold += 1
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nn.Sequential(*out), n_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_dir")
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    spec = importlib.util.spec_from_file_location("pkgrunner", os.path.join(a.model_dir, "runner.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["pkgrunner"] = m
    spec.loader.exec_module(m)
    import csig_ops as ops
    r = m.Runner(a.model_dir)
    dev, dt = r.device, r.weight_dtype
    mf = torch.channels_last if ops.CHANNELS_LAST else None

    from diffusers import AutoencoderTiny
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()

    x = torch.zeros(1, 3, 512, 512, dtype=dt, device=dev)
    z = torch.zeros(1, 4, 64, 64, dtype=dt, device=dev)
    if mf:
        x = x.contiguous(memory_format=mf)
        z = z.contiguous(memory_format=mf)

    print("GPU:", torch.cuda.get_device_name(0), "| runner 的 channels_last:", ops.CHANNELS_LAST, "\n")
    rows = []
    with torch.no_grad():
        rows.append(("decode 原样(contiguous 权重)", bench(lambda: tae.decoder(z))))
        rows.append(("encode 原样", bench(lambda: tae.encoder(x))))

        tae_cl = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()
        if mf:
            tae_cl.to(memory_format=mf)
        rows.append(("decode + channels_last", bench(lambda: tae_cl.decoder(z))))
        rows.append(("encode + channels_last", bench(lambda: tae_cl.encoder(x))))

        folded, nf = fold_upsample(tae_cl.decoder, mf)
        ref = tae_cl.decoder(z).float()
        got = folded(z).float()
        err = float((ref - got).abs().max())
        rows.append((f"decode + cl + 上采样折叠({nf}处)", bench(lambda: folded(z))))
        print(f"上采样折叠的数值误差(最大绝对差): {err:.2e}\n")

        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                folded(z)
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g):
            _ = folded(z)
        rows.append(("decode + cl + 折叠 + CUDAGraph", bench(lambda: g.replay())))

    print("%-34s %10s" % ("变体", "毫秒"))
    print("-" * 46)
    for k, v in rows:
        print("%-34s %9.3f" % (k, v))


if __name__ == "__main__":
    main()
