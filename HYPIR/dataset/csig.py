"""CSIG-2026 赛道二的数据管线：nvJPEG GPU 解码 + 实测退化配方。

替换原版的 RealESRGANDataset / RealESRGANBatchTransform。两处根本差异：

**1. 退化配方按实测重建**（见 csig/README.md）。原版一半样本带 sigma<=30 的高斯噪 + 泊松噪，
而真实 LQ 的噪声实测只有 sigma 0.014-0.090 灰阶（不到 1/10 个 LSB）—— 模型学到的激进降噪
在完全无噪的输入上只会吃掉本就稀缺的细节。真实退化的感知强度相当于 ~7x 重采样 / sigma~3.6 高斯，
而不是此前认为的 2x。

**2. 解码搬到 GPU**。原来每个样本要在 worker 里全解一张 4096x3072 的 JPEG（实测 100 ms/核，
4 个 worker 只有 ~27 样本/秒），是硬瓶颈。现在 worker 只读原始字节（0.01 ms/张），
解码与随机裁剪在 GPU 上批量做。

实测（NGC pytorch:26.07 / H200 / torchvision 0.28 nvJPEG）：

| batch | 解码+裁剪 | 吞吐 | 每张 |
|---:|---:|---:|---:|
| 8 | 51.8 ms | 154/秒 | 6.5 ms |
| 24 | 83.4 ms | 288/秒 | 3.5 ms |
| 32 | 107.4 ms | 298/秒 | 3.4 ms |

（对照：CPU 全解码 100 ms/张 = 10 张/秒/核；DCT 域裁剪实测只有 1.3x，因为仍要
Huffman 解码整条熵流才能定位 MCU，已弃用。）
"""

import os
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

# 退化参数，按感知分标定到中位 p_rel 0.989（真实 LQ 为 1.000）。
# 标定过程：pole=1、各向同性、无空间变化时 sigma=1.8/2.4/2.8/3.2 分别给出
# p_rel 2.419/1.492/1.091/0.803，即 sigma~2.9 对应真实强度；下面区间以此为中心展开。
SIGMA_RANGE = (1.85, 3.45)
ANISO_RANGE = (1.0, 1.45)     # 长短轴比，实测 1.2-1.4
POLE_RANGE = (0.9, 2.1)       # 极点阶数；纯高斯已被实测拒绝（重尾更贴）
SPATIAL_VAR = 0.4             # 图内 sigma 变化，实测 ±40%
AFFINE_A = (1.00, 1.10)
AFFINE_B = (-14.0 / 255.0, 0.0)
JPEG_QUALITY = 95             # 字节级确认：128 个量化系数与 IJG q95 全等
P_AFFINE = 0.6


class CSIGDataset(Dataset):
    """只读原始 JPEG 字节，解码留给 GPU。

    worker 侧成本实测 0.01 ms/张（8 万张/秒/核），所以 2 个 worker 足够，
    不需要为解码堆 CPU。
    """

    def __init__(self, file_list: str, out_size: int = 512, prompt: str = "",
                 image_path_prefix: str = "") -> None:
        with open(file_list, "r") as f:
            self.paths = [l.strip() for l in f if l.strip()]
        assert self.paths, "空的 file_list: %s" % file_list
        self.prefix = image_path_prefix
        self.out_size = out_size
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        for _ in range(10):     # 少量坏文件直接换一张，不让训练挂掉
            try:
                with open(self.prefix + self.paths[idx], "rb") as f:
                    buf = f.read()
                if len(buf) > 1024:
                    return {"jpeg": torch.frombuffer(bytearray(buf), dtype=torch.uint8),
                            "txt": self.prompt}
            except Exception:
                pass
            idx = random.randrange(len(self.paths))
        raise RuntimeError("连续 10 次读图失败，检查 file_list")


def csig_collate(batch: List[Dict]) -> Dict:
    """字节长度不一，不能 stack，保持 list 交给 nvJPEG 批量解码。"""
    return {"jpeg": [b["jpeg"] for b in batch], "txt": [b["txt"] for b in batch]}


class CSIGBatchTransform:
    """GPU 上：nvJPEG 批量解码 -> 随机裁剪 -> 退化。逐样本独立采参数。"""

    def __init__(self, hq_key: str = "jpeg", extra_keys: Optional[List[str]] = None,
                 out_size: int = 512, jpeg_quality: int = JPEG_QUALITY,
                 use_hflip: bool = True) -> None:
        self.hq_key = hq_key
        self.extra_keys = extra_keys or ["txt"]
        self.out_size = out_size
        self.jpeg_quality = jpeg_quality
        self.use_hflip = use_hflip
        self.jpeger = None

    # ---------- 解码与裁剪 ----------
    def _decode_crop(self, bufs: List[torch.Tensor], device) -> torch.Tensor:
        S = self.out_size
        imgs = decode_jpeg(bufs, device=device, mode=ImageReadMode.RGB)
        crops = []
        for im in imgs:
            _, h, w = im.shape
            if h < S or w < S:      # 兜底：太小的图放大到至少 S
                im = F.interpolate(im[None].float(), size=(max(h, S), max(w, S)),
                                   mode="bicubic", align_corners=False)[0].clamp(0, 255).to(torch.uint8)
                _, h, w = im.shape
            y = random.randint(0, h - S)
            x = random.randint(0, w - S)
            c = im[:, y:y + S, x:x + S]
            if self.use_hflip and random.random() < 0.5:
                c = torch.flip(c, dims=[2])
            crops.append(c)
        return torch.stack(crops).float().div_(255.0)

    # ---------- 退化 ----------
    def _lowpass(self, x, sigma, aniso, theta, pole):
        """各向异性重尾低通，逐样本参数。H(f) = 1/(1+(f'/f0)^2)^pole。

        f0 = 0.1325/sigma 是高斯 -6dB 点的换算。重尾（单极点~四次）比纯高斯更贴实测，
        logRMSE 0.078-0.74 vs 1.33-5.47。
        """
        b, _, h, w = x.shape
        fy = torch.fft.fftfreq(h, device=x.device).view(1, h, 1)
        fx = torch.fft.fftfreq(w, device=x.device).view(1, 1, w)
        c = torch.cos(theta).view(b, 1, 1)
        s = torch.sin(theta).view(b, 1, 1)
        u = (fx * c + fy * s) / aniso.view(b, 1, 1)
        v = -fx * s + fy * c
        f0 = (0.1325 / sigma).view(b, 1, 1)
        resp = 1.0 / (1.0 + (u * u + v * v) / (f0 * f0)) ** pole.view(b, 1, 1)
        return torch.fft.ifft2(torch.fft.fft2(x) * resp.unsqueeze(1)).real

    @torch.no_grad()
    def __call__(self, batch: Dict) -> Dict:
        bufs = batch[self.hq_key]
        device = torch.device("cuda", torch.cuda.current_device())
        hq = self._decode_crop(bufs, device)
        b, _, h, w = hq.shape

        def U(lo, hi):
            return torch.empty(b, device=device).uniform_(lo, hi)

        # 1) 低通（含图内空间变化：两档模糊 + 平滑随机掩码混合）
        sigma = U(*SIGMA_RANGE)
        lo = self._lowpass(hq, sigma * (1 - SPATIAL_VAR), U(*ANISO_RANGE),
                           U(0.0, float(np.pi)), U(*POLE_RANGE))
        hi = self._lowpass(hq, sigma * (1 + SPATIAL_VAR), U(*ANISO_RANGE),
                           U(0.0, float(np.pi)), U(*POLE_RANGE))
        m = torch.rand(b, 1, 8, 8, device=device)
        m = F.interpolate(m, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        out = lo * (1 - m) + hi * m

        # 2) 逐通道仿射 + clip（clip 到 0 产生黑位压死，实测真实数据有 4.5-7.8% 像素死在 0）
        do = (torch.rand(b, device=device) < P_AFFINE).float().view(b, 1, 1, 1)
        ca = torch.empty(b, 3, 1, 1, device=device).uniform_(*AFFINE_A)
        cb = torch.empty(b, 3, 1, 1, device=device).uniform_(*AFFINE_B)
        out = (out * (1 - do + do * ca) + do * cb).clamp(0, 1)

        # 3) JPEG q95（DiffJPEG 与 libjpeg 的 4:2:0 一致，两者 q95 差 2 dB，远小于模糊项）
        if self.jpeger is None:
            from HYPIR.dataset.diffjpeg import DiffJPEG
            self.jpeger = DiffJPEG(differentiable=False).to(device)
        self.jpeger.to(out)
        out = self.jpeger(out, quality=out.new_full((b,), float(self.jpeg_quality)))

        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        return {"GT": hq, "LQ": lq, **{k: batch[k] for k in self.extra_keys}}
