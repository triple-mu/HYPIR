"""TAESD 的 cuDNN 融合版：conv+relu 与 conv+add+relu 走 cudnn 融合原语。

TAESD 全程只有 conv 和 ReLU，没有归一化层，正好对上 PyTorch 暴露的两个 cudnn 融合原语：
    torch.cudnn_convolution_relu(x, w, b, stride, padding, dilation, groups)
    torch.cudnn_convolution_add_relu(x, w, z, alpha, b, stride, padding, dilation, groups)

结构里 conv->ReLU 有 28 处，Block 末尾的 conv->add(skip)->ReLU 有 13 处，全部可融。
微基准（64ch @512x512）实测 0.524 -> 0.309 ms，1.70x。

**数值**：微基准里融合与分开的最大绝对差 1.25e-01，但那是拿 N(0,1) 随机权重测的——
576 项累加后输出标准差约 24，折算相对误差 0.5%，是 fp16 累加顺序差异而非错误。
真实权重下的判据必须是端到端输出的 PSNR，本脚本就是量那个。

    python csig/taesd_fused.py --taesd-dir <含 taesd_*.safetensors 的目录>
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taesd_pure import (Clamp, load_safetensors, find_taesd, build_decoder_layers,
                        build_encoder_layers, load_into, FoldedUp, _CONVT_TAPS)

_P = ((1, 1), (1, 1), (1, 1), 1)      # stride, padding, dilation, groups


class FusedConvReLU(nn.Module):
    """conv + bias + ReLU 一次 cudnn 调用。"""

    def __init__(self, conv):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach(), requires_grad=False)
        b = conv.bias
        # cudnn 融合原语要求 bias 非空；无 bias 的层补零向量，代价可忽略
        self.bias = nn.Parameter(b.detach() if b is not None else
                                 torch.zeros(conv.out_channels, dtype=conv.weight.dtype,
                                             device=conv.weight.device), requires_grad=False)
        self.stride, self.padding = conv.stride, conv.padding

    def forward(self, x):
        return torch.cudnn_convolution_relu(x, self.weight, self.bias,
                                            self.stride, self.padding, (1, 1), 1)


class FusedBlock(nn.Module):
    """conv-ReLU-conv-ReLU-conv + skip + ReLU  ->  两次 conv_relu + 一次 conv_add_relu。"""

    def __init__(self, blk):
        super().__init__()
        c0, c2, c4 = blk.conv[0], blk.conv[2], blk.conv[4]
        self.a = FusedConvReLU(c0)
        self.b = FusedConvReLU(c2)
        self.w = nn.Parameter(c4.weight.detach(), requires_grad=False)
        self.bi = nn.Parameter(c4.bias.detach(), requires_grad=False)

    def forward(self, x):
        h = self.b(self.a(x))
        # z=x 即残差；alpha=1
        return torch.cudnn_convolution_add_relu(h, self.w, x, 1, self.bi, (1, 1), (1, 1), (1, 1), 1)


def fuse(seq, mf):
    """把 Sequential 里的 Block、conv+ReLU 对、Upsample+conv 对全部替换掉。"""
    from taesd_pure import Block
    mods, out, i = list(seq), [], 0
    n_blk = n_cr = n_up = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, Block):
            out.append(FusedBlock(m)); n_blk += 1; i += 1
        elif isinstance(m, nn.Upsample) and i + 1 < len(mods) and isinstance(mods[i + 1], nn.Conv2d):
            out.append(FoldedUp(mods[i + 1], mf is not None)); n_up += 1; i += 2
        elif isinstance(m, nn.Conv2d) and i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU):
            out.append(FusedConvReLU(m)); n_cr += 1; i += 2
        else:
            out.append(m); i += 1
    return nn.Sequential(*out), (n_blk, n_cr, n_up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taesd-dir", required=True)
    a = ap.parse_args()
    dev, dt, mf = torch.device("cuda"), torch.float16, torch.channels_last
    torch.backends.cudnn.benchmark = True
    ep, dp = find_taesd(a.taesd_dir)

    from taesd_pure import TAESDEncoder, TAESDDecoder
    enc, dec = TAESDEncoder(), TAESDDecoder()
    load_into(enc.layers, load_safetensors(ep))
    load_into(dec.layers, load_safetensors(dp))
    enc = enc.to(dev, dt).eval().to(memory_format=mf)
    dec = dec.to(dev, dt).eval().to(memory_format=mf)

    fenc, fdec = TAESDEncoder(), TAESDDecoder()
    fenc.layers, se = fuse(enc.layers, mf)
    fdec.layers, sd_ = fuse(dec.layers, mf)
    fenc, fdec = fenc.to(dev, dt).eval(), fdec.to(dev, dt).eval()
    print("编码器融合: %d 个 Block, %d 个 conv+relu, %d 个上采样折叠" % se)
    print("解码器融合: %d 个 Block, %d 个 conv+relu, %d 个上采样折叠\n" % sd_)

    x = (torch.rand(1, 3, 512, 512, device=dev, dtype=dt) * 2 - 1).contiguous(memory_format=mf)
    z = (torch.randn(1, 4, 64, 64, device=dev, dtype=dt) * 3).contiguous(memory_format=mf)

    def psnr(u, v, rng=2.0):
        return 10 * torch.log10(rng ** 2 / ((u.float() - v.float()) ** 2).mean()).item()

    def bench(fn, n=100, w=20):
        for _ in range(w):
            fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

    with torch.no_grad():
        ze, zf = enc(x), fenc(x)
        oe, of = dec(z), fdec(z)
        print("端到端数值（真实权重）：")
        print("  encoder  PSNR %6.2f dB   最大绝对差 %.3e" % (psnr(ze, zf, 8.0), (ze - zf).abs().max()))
        print("  decoder  PSNR %6.2f dB   最大绝对差 %.3e" % (psnr(oe, of), (oe - of).abs().max()))
        t_e0, t_e1 = bench(lambda: enc(x)), bench(lambda: fenc(x))
        t_d0, t_d1 = bench(lambda: dec(z)), bench(lambda: fdec(z))
    print("\n%-22s %10s %10s %8s" % ("", "未融合", "cudnn融合", "加速"))
    print("-" * 54)
    print("%-22s %9.3f %10.3f %7.2fx" % ("encoder", t_e0, t_e1, t_e0 / t_e1))
    print("%-22s %9.3f %10.3f %7.2fx" % ("decoder", t_d0, t_d1, t_d0 / t_d1))
    print("%-22s %9.3f %10.3f %7.2fx" % ("合计", t_e0 + t_d0, t_e1 + t_d1,
                                          (t_e0 + t_d0) / (t_e1 + t_d1)))
    unet = 33.76
    print("\n代入整条链路（UNet %.2f ms 不变）：%.2f -> %.2f ms"
          % (unet, unet + t_e0 + t_d0, unet + t_e1 + t_d1))
    base = 85.78
    for tag, tot in (("融合前", unet + t_e0 + t_d0), ("融合后", unet + t_e1 + t_d1)):
        sp = base / tot
        print("  %s: 相对 SD/SD 的 %.2f ms  提速 %.2fx  速度分 %+.1f%%"
              % (tag, base, sp, 100 * (sp ** 0.204 - 1)))


if __name__ == "__main__":
    main()
