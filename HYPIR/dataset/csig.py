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

# 退化参数。
#
# v1（窄）曾按感知分标定到中位 p_rel 0.989，实测 sigma~2.9 对应真实强度，区间以此展开：
#   SIGMA_RANGE=(1.85,3.45) ANISO=(1.0,1.45) POLE=(0.9,2.1) JPEG 钉死 q95 无噪声
# 结果是灾难：模型在这套退化上从 19.0% 涨到 45.6%，线上却从 44.04 掉到 20.76。
#
# 判决实验（csig/domain_split.py，赛题验证对，同一批图只换退化方式）：
#   赛题内容 + 合成退化   基线 16.3%  ->  微调 44.7%
#   赛题内容 + 真实退化   基线 13.5%  ->  微调 -0.1%
# 内容域没问题，问题就是退化。而配对实验（csig/degrade_ablate.py）显示合成与真实的
# ΔFR +0.012 / ΔNR +0.002，两个感知指标都匹配得很好。
#
# **匹配 FR/NR 两个标量 != 匹配退化算子。** 两者落在同一感知分位置，却是不同的变换；
# 模型学会的是求逆我那一个特定算子。真实 LQ 的 FR 0.384 经微调模型只到 0.391——几乎没动。
#
# v2 的目标因此改成：让真实退化落进合成分布的**支撑集内部**，而不是让均值相等。
# 具体两条改动：
#   1. 加入真实重采样链路。v1 是纯频域低通（给频率响应乘衰减），永远不产生**混叠**；
#      而真实的降采样-升采样必然带混叠与振铃。这是「感知分相同、算子不同」的关键缺口。
#   2. 全面放宽区间，并加回轻度噪声与 JPEG 质量范围。v1 为了"精确匹配实测中心"而收窄，
#      恰恰是这个收窄让模型失去泛化——原版 HYPIR 用 RealESRGAN 那套很宽的分布，
#      基线在真实退化上能拿 13.5% 正是因为它不专精于任何单一算子。
SIGMA_RANGE = (0.6, 6.2)      # v1 (1.85,3.45)，中心不变、两端大幅放开。
                              # 上界按覆盖实测定：真实最重的 crop FR 0.290，
                              # 4.6 时合成最重只到 0.313，盖不住重尾
ANISO_RANGE = (1.0, 2.0)      # v1 (1.0,1.45)
POLE_RANGE = (0.7, 3.0)       # v1 (0.9,2.1)；极点阶数，纯高斯已被实测拒绝
SPATIAL_VAR = 0.4             # 图内 sigma 变化，实测 ±40%
AFFINE_A = (1.00, 1.10)
AFFINE_B = (-14.0 / 255.0, 0.0)
JPEG_RANGE = (70, 98)         # v1 钉死 95（字节级确认真实 LQ 是 IJG q95）。
                              # 仍以 95 为常见值，但让模型见过别的质量，别把 q95 的
                              # 块效应当成唯一先验
JPEG_QUALITY = 95             # 兼容旧引用；单值路径仍用它
P_AFFINE = 0.6

# --- v2 新增 ---
P_RESAMPLE = 0.65             # 走真实重采样链路（而非纯频域低通）的概率
SCALE_RANGE = (1.6, 7.0)      # 降采样倍数；实测真实退化的感知强度约等于 7x 重采样，
                              # 但那是频域标定值，这里给一个覆盖它的宽区间
NOISE_SIGMA = (0.0, 2.5 / 255.0)   # 真实 LQ 实测噪声 0.014-0.090 灰阶（几乎没有）。
                              # 加回一点点不是为了教降噪，是为了别让"零噪声"成为
                              # 模型赖以识别输入的特征
P_CLEAN = 0.04                # 少量近乎不退化的样本：教会模型"输入已经够好就别动"。
                              # v1 里模型永远面对重退化输入，于是无条件大改


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

    def __getitem__(self, idx: int) -> Dict[str, Union[bytes, str]]:
        # 返回原始 bytes 而不是 tensor：accelerate 的 prepare(dataloader) 会把 batch 里的
        # tensor 自动搬到 GPU，而 decode_jpeg 要求输入字节在 CPU 上。返回 bytes 可以让
        # accelerate 直接跳过（它只递归处理 tensor），也省掉 GPU->CPU->GPU 的往返。
        for _ in range(10):     # 少量坏文件直接换一张，不让训练挂掉
            try:
                with open(self.prefix + self.paths[idx], "rb") as f:
                    buf = f.read()
                if len(buf) > 1024:
                    return {"jpeg": buf, "txt": self.prompt}
            except Exception:
                pass
            idx = random.randrange(len(self.paths))
        raise RuntimeError("连续 10 次读图失败，检查 file_list")


def csig_collate(batch: List[Dict]) -> Dict:
    """bytes 无法 stack，保持 list；也让 accelerate 的自动设备搬运跳过它。"""
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
    def _decode_crop(self, bufs: List[bytes], device) -> torch.Tensor:
        S = self.out_size
        # decode_jpeg 要求输入张量在 CPU 上（它内部自己拷到 GPU 解码）
        cpu_bufs = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in bufs]
        imgs = decode_jpeg(cpu_bufs, device=device, mode=ImageReadMode.RGB)
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

    def _resample_chain(self, x):
        """降采样再升回原尺寸。与 _lowpass 的关键差别是**会产生混叠**。

        频域低通只是把高频乘小，信息干净地消失；而真实的降采样把高频折叠到低频，
        升采样再引入插值核的振铃。这两类失真的可逆性完全不同，模型学会求逆前者
        并不代表会求逆后者——v1 只有前者，这是它在真实数据上失效的直接原因。

        下采样与上采样各自独立抽插值方式：真实 ISP 链路里这两步未必用同一个核。
        逐样本抽倍率，所以按样本循环（batch 8，开销可忽略）。
        """
        b, _, h, w = x.shape
        modes = ("bilinear", "bicubic", "area")
        outs = []
        for i in range(b):
            s = float(torch.empty(1).uniform_(*SCALE_RANGE))
            dm = modes[int(torch.randint(len(modes), (1,)))]
            um = modes[int(torch.randint(len(modes) - 1, (1,)))]   # 上采样不用 area
            kw_d = {} if dm == "area" else {"align_corners": False}
            small = F.interpolate(x[i:i + 1], size=(max(int(h / s), 8), max(int(w / s), 8)),
                                  mode=dm, **kw_d)
            outs.append(F.interpolate(small, size=(h, w), mode=um, align_corners=False))
        return torch.cat(outs).clamp(0, 1)

    @torch.no_grad()
    def __call__(self, batch: Dict) -> Dict:
        bufs = batch[self.hq_key]
        device = torch.device("cuda", torch.cuda.current_device())
        hq = self._decode_crop(bufs, device)
        b, _, h, w = hq.shape

        def U(lo, hi):
            return torch.empty(b, device=device).uniform_(lo, hi)

        # 1) 模糊。两条支路二选一，逐样本抽：
        #    a) 频域重尾低通（v1 唯一的那条）——干净的频率衰减，不产生混叠
        #    b) 真实重采样链路——降采样再升回来，必然带混叠与振铃
        #    真实退化里两种成分都有，v1 只有 a，模型于是只会求逆 a。
        use_rs = torch.rand(b, device=device) < P_RESAMPLE

        sigma = U(*SIGMA_RANGE)
        lo = self._lowpass(hq, sigma * (1 - SPATIAL_VAR), U(*ANISO_RANGE),
                           U(0.0, float(np.pi)), U(*POLE_RANGE))
        hi = self._lowpass(hq, sigma * (1 + SPATIAL_VAR), U(*ANISO_RANGE),
                           U(0.0, float(np.pi)), U(*POLE_RANGE))
        m = torch.rand(b, 1, 8, 8, device=device)
        m = F.interpolate(m, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        out = lo * (1 - m) + hi * m

        if bool(use_rs.any()):
            # 逐样本抽倍率与插值方式；下采样与上采样各自独立选，因为真实 ISP 链路
            # 里这两步未必用同一个核
            rs = self._resample_chain(hq)
            sel = use_rs.view(b, 1, 1, 1).float()
            out = out * (1 - sel) + rs * sel

        # 少量近乎不退化的样本：v1 里模型永远面对重退化输入，于是无条件大改；
        # 真实测试集里有干净图（取证查出 9 张几乎未退化），必须教会它"别动"
        keep = (torch.rand(b, device=device) < P_CLEAN).view(b, 1, 1, 1).float()
        out = out * (1 - keep) + hq * keep

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
        # v1 钉死 q95。真实 LQ 确实是 IJG q95（字节级确认），但把它当成唯一先验会让
        # 模型把 q95 的块效应当作输入的识别特征。给个范围，95 仍是常见值。
        q = torch.empty(b, device=device).uniform_(*JPEG_RANGE)
        out = self.jpeger(out, quality=q)

        # 轻度噪声：真实 LQ 实测几乎无噪，加一点不是为了教降噪，
        # 是为了别让"零噪声"成为模型赖以判断的特征
        ns = torch.empty(b, 1, 1, 1, device=device).uniform_(*NOISE_SIGMA)
        out = (out + torch.randn_like(out) * ns).clamp(0, 1)

        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        return {"GT": hq, "LQ": lq, **{k: batch[k] for k in self.extra_keys}}
