"""纯 PyTorch 的 TAESD（不依赖 diffusers），叠满优化后在 V100 上测速。

提交包只能依赖 torch，所以 TAESD 必须像 runner 里的 SD VAE 一样内联实现、
自己解析权重。权重文件 taesd_{encoder,decoder}.safetensors 用的就是原始
nn.Sequential 索引命名，不需要 key 映射。

结构（全程只有 conv + ReLU，没有任何归一化层，对融合极友好）：
    解码器 Clamp -> conv(4,64) -> ReLU -> B*3 -> [Up(2) -> conv(64,64,无bias) -> B*3]*3
           -> B -> conv(64,3)
    编码器 conv(3,64) -> B -> [conv(s=2,无bias) -> B*3]*3 -> conv(64,4)
    B = conv-ReLU-conv-ReLU-conv + 恒等跳连 + ReLU（全是 64->64，skip 无参数）

叠加的优化：
    1. 纯 torch，去掉 diffusers 的包装开销
    2. fp16 + channels_last（runner 的 latent 本来就是 channels_last，
       原先用 AutoencoderTiny 时是 contiguous 加载的，中间白挨一次布局转换）
    3. 上采样折叠：Up(2)+conv(k3) -> ConvTranspose(k4,s2,p1)，乘加降到 4/9，
       复用 runner 里已验证的 fold_upsample_conv（float64 相对误差 2e-16）
    4. cudnn.benchmark
    5. CUDA Graph（kernel 小而多，启动开销占比与 SD VAE 那条路完全不同）

    python csig/taesd_pure.py --taesd-dir <含 taesd_*.safetensors 的目录>
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
              "BF16": torch.bfloat16}
# nearest 上采样后 3x3 卷积的抽头映射，与 runner 的 _CONVT_TAPS 一致
_CONVT_TAPS = {0: [2], 1: [1, 2], 2: [0, 1], 3: [0]}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, info in head.items():
            if k == "__metadata__":
                continue
            b, e = info["data_offsets"]
            f.seek(base + b)
            buf = bytearray(f.read(e - b))
            out[k] = torch.frombuffer(buf, dtype=_ST_DTYPES[info["dtype"]]).view(info["shape"])
    return out


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    """conv-ReLU-conv-ReLU-conv + 恒等跳连 + ReLU。in==out 时 skip 无参数。"""

    def __init__(self, n=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(n, n, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(n, n, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(n, n, 3, padding=1))
        self.fuse = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fuse(self.conv(x) + x)


def build_decoder():
    return nn.Sequential(
        Clamp(), nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(inplace=True),
        Block(), Block(), Block(), nn.Upsample(scale_factor=2), nn.Conv2d(64, 64, 3, padding=1, bias=False),
        Block(), Block(), Block(), nn.Upsample(scale_factor=2), nn.Conv2d(64, 64, 3, padding=1, bias=False),
        Block(), Block(), Block(), nn.Upsample(scale_factor=2), nn.Conv2d(64, 64, 3, padding=1, bias=False),
        Block(), nn.Conv2d(64, 3, 3, padding=1))


def build_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), Block(),
        nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False), Block(), Block(), Block(),
        nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False), Block(), Block(), Block(),
        nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False), Block(), Block(), Block(),
        nn.Conv2d(64, 4, 3, padding=1))


def load_into(seq, sd):
    """权重文件的 key 就是 Sequential 索引，逐条严格匹配。"""
    got = seq.state_dict()
    assert set(got) == set(sd), (
        "key 不匹配\n  只在模型里: %s\n  只在权重里: %s"
        % (sorted(set(got) - set(sd))[:5], sorted(set(sd) - set(got))[:5]))
    seq.load_state_dict({k: v.to(got[k].dtype) for k, v in sd.items()})
    return seq


class FoldedUp(nn.Module):
    """Upsample(nearest,2) + Conv2d(k3,p1) 折成 ConvTranspose2d(k4,s2,p1)。"""

    def __init__(self, conv, channels_last):
        super().__init__()
        W = conv.weight.detach()
        o, c = W.shape[:2]
        v = torch.zeros(o, c, 4, 4, dtype=W.dtype, device=W.device)
        for kp, sp in _CONVT_TAPS.items():
            for kq, sq in _CONVT_TAPS.items():
                v[:, :, kp, kq] = W[:, :, sp, :][:, :, :, sq].sum((2, 3))
        v = v.transpose(0, 1).contiguous()
        if channels_last:
            v = v.contiguous(memory_format=torch.channels_last)
        self.weight = nn.Parameter(v, requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach(), requires_grad=False)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, stride=2, padding=1)


def fold_decoder(dec, channels_last):
    mods, out, n = list(dec), [], 0
    i = 0
    while i < len(mods):
        if (isinstance(mods[i], nn.Upsample) and i + 1 < len(mods)
                and isinstance(mods[i + 1], nn.Conv2d)):
            out.append(FoldedUp(mods[i + 1], channels_last))
            n += 1
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nn.Sequential(*out), n


def find_taesd(d):
    e = glob.glob(os.path.join(d, "**", "taesd_encoder.safetensors"), recursive=True)
    dd = glob.glob(os.path.join(d, "**", "taesd_decoder.safetensors"), recursive=True)
    assert e and dd, "找不到 taesd_{encoder,decoder}.safetensors，dir=%s" % d
    return e[0], dd[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taesd-dir", required=True)
    ap.add_argument("--compare-diffusers", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    dt = torch.float16
    torch.backends.cudnn.benchmark = True
    ep, dp = find_taesd(a.taesd_dir)

    enc = load_into(build_encoder(), load_safetensors(ep)).to(dev, dt).eval()
    dec = load_into(build_decoder(), load_safetensors(dp)).to(dev, dt).eval()
    print("权重加载完成（key 逐条严格匹配）")

    x = torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1
    z = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)

    def bench(fn, n=100, w=20):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    rows = []
    with torch.no_grad():
        if a.compare_diffusers:
            from diffusers import AutoencoderTiny
            tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()
            de = (tae.encoder(x).float() - enc(x).float()).abs().max()
            dd_ = (tae.decoder(z).float() - dec(z).float()).abs().max()
            print("与 diffusers 的最大绝对差: encoder %.2e  decoder %.2e" % (de, dd_))
            rows.append(("diffusers AutoencoderTiny encode", bench(lambda: tae.encoder(x))))
            rows.append(("diffusers AutoencoderTiny decode", bench(lambda: tae.decoder(z))))

        rows.append(("纯torch encode (contiguous)", bench(lambda: enc(x))))
        rows.append(("纯torch decode (contiguous)", bench(lambda: dec(z))))

        mf = torch.channels_last
        enc_cl = enc.to(memory_format=mf)
        dec_cl = dec.to(memory_format=mf)
        xc = x.contiguous(memory_format=mf)
        zc = z.contiguous(memory_format=mf)
        rows.append(("  + channels_last encode", bench(lambda: enc_cl(xc))))
        rows.append(("  + channels_last decode", bench(lambda: dec_cl(zc))))

        folded, nf = fold_decoder(dec_cl, True)
        folded = folded.to(dev, dt)
        err = (dec_cl(zc).float() - folded(zc).float()).abs().max()
        print("上采样折叠 %d 处，与折叠前最大绝对差 %.2e" % (nf, err))
        rows.append(("  + 上采样折叠 decode", bench(lambda: folded(zc))))

        for tag, mod, inp in (("encode", enc_cl, xc), ("decode", folded, zc)):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    mod(inp)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mod(inp)
            rows.append(("  + CUDA Graph %s" % tag, bench(lambda: g.replay())))

    print("\n%-36s %10s" % ("变体", "毫秒"))
    print("-" * 48)
    for k, v in rows:
        print("%-36s %9.3f" % (k, v))


if __name__ == "__main__":
    main()
