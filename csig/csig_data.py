"""CSIG-2026 赛道二的训练数据管线 —— 替换 HYPIR 的 RealESRGAN 退化。

接口契约与 HYPIR 原件一致，可直接在 configs/sd2_train.yaml 里换 target：
    dataset:         -> CSIGDataset          返回 {"hq": [3,H,W] [0,1] RGB, "txt": str}
    batch_transform: -> CSIGBatchTransform   返回 {"GT": [B,3,H,W], "LQ": [B,3,H,W], "txt": [...]}

与 RealESRGAN 那套的关键差别（全部来自实测，见 ../FINDINGS.md §5 与 §9.5）：

| 维度       | RealESRGAN（HYPIR 原配置）        | 真实退化（本文件实现）              |
|-----------|----------------------------------|----------------------------------|
| 噪声       | 50% 概率高斯 σ∈[1,30]+泊松+灰噪   | **完全没有**（实测 σ<0.1 灰阶）    |
| JPEG      | q ∈ [30, 95] 随机                 | **恒为 q95**（字节级确认）         |
| 模糊核     | iso/aniso/plateau/sinc 混合       | 重尾单极点~四次、各向异性、空间变化 |
| 训练目标   | `use_sharpener=True` USM 锐化后的 GT | **原始 GT**（USM 把天花板从 8.340 压到 7.965） |

**噪声那条最要命**：原配置有一半样本带强噪声，模型学会了激进降噪；喂给它完全无噪的输入，
那套降噪行为只会吃掉本就稀缺的细节。

⚠️ **crop 必须在原生分辨率上裁**。退化的 σ≈2-4 px 是在 4K 栅格上测的；
先把 4K 缩到 512 再加退化，相对模糊强度会差 8 倍。用 `screen_corpus.py` 筛数据源。
"""

import random
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

# 参数范围与 degrade.py 保持一致（那边是单图 numpy 版，这里是 GPU 批量版）
SIGMA_RANGE = (1.85, 3.45)
ANISO_RANGE = (1.0, 1.45)
POLE_RANGE = (0.9, 2.1)
SPATIAL_VAR = 0.4
AFFINE_A = (1.00, 1.10)
AFFINE_B = (-14.0 / 255.0, 0.0)   # 这里工作在 [0,1] 域，故除以 255
JPEG_QUALITY = 95
P_AFFINE = 0.6


def hf_max(gray: np.ndarray, s: int = 512, grid: int = 4, nb: int = 64) -> float:
    """原生锐度判据：全图最锐 patch 的高频能量占比。**越高越锐。**

    取 grid x grid 个 512 patch（跳过 std<=8 的纯色/天空），每个做 Hanning 窗 + |FFT|^2 +
    径向平均，报 `P(r>0.5·Nyq) / P(r>0)`，整图取 max。取 max 而非中位是为了避开
    虚化背景/大片天空造成的误杀。

    实测标定（n=16 真手机原生图，每张同时构造三个变体）：
        原生                        hf_max 中位 0.01717，p10 0.00567
        2x 下采样+上采样（假高清）    中位 0.00266，p10 0.00075
        高斯 σ=2.5（≈赛题退化强度）   中位 0.00006
    阈值取舍：0.005 保留 ~92% 原生、误收 ~15% 假高清、**误收 0% 模糊图**。
    数据完全不缺，宁可多杀不可放过。

    注：先前用「2x 自往返 PSNR」做判据已废弃——它内容依赖太强，实测把原生的
    case1.jpg(47.82 dB) 误拒、把已退化的 case99.jpg(41.56 dB) 误收。
    """
    h, w = gray.shape[:2]
    g = gray.astype(np.float64)
    win = np.hanning(s)[:, None] * np.hanning(s)[None, :]
    best = 0.0
    for iy in range(grid):
        for ix in range(grid):
            y = int((h - s) * (iy + 0.5) / grid)
            x = int((w - s) * (ix + 0.5) / grid)
            c = g[y:y + s, x:x + s]
            if c.shape != (s, s) or c.std() <= 8:
                continue
            F = np.fft.fftshift(np.abs(np.fft.fft2((c - c.mean()) * win)) ** 2)
            cy = cx = s // 2
            yy, xx = np.indices(F.shape)
            r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
            idx = np.clip((r * nb).astype(int), 0, nb - 1)
            cnt = np.bincount(idx.ravel(), minlength=nb)
            p = np.bincount(idx.ravel(), F.ravel(), minlength=nb) / np.maximum(cnt, 1)
            p /= p[1]
            best = max(best, p[nb // 2:].sum() / p[1:].sum())
    return float(best)


_ANNEX_K = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def ijg_quality(table) -> Optional[int]:
    """把量化表反推成 IJG quality。**厂商自研表返回 None**（表示该判据不适用）。

    这一步必须做：华为原生 JPEG 的亮度表和是 915，按 IJG 的「表和↔质量」映射会被当成
    低质量图误杀 —— 而那正是我们最宝贵的域内 GT（测试集里 9 张未退化的相机原生图）。
    """
    t = list(table)[:64]
    best, best_err = None, 1e9
    for q in range(1, 101):
        s = 5000 // q if q < 50 else 200 - 2 * q
        ref = [min(255, max(1, (b * s + 50) // 100)) for b in _ANNEX_K]
        err = sum(abs(a - b) for a, b in zip(t, ref)) / 64.0
        if err < best_err:
            best, best_err = q, err
    return best if best_err <= 3.0 else None   # 拟合不上就是厂商表


def screen(path: str) -> bool:
    """三道闸门。任何一道不过就别拿来当训练 GT。"""
    import os
    from PIL import Image
    im = Image.open(path)
    # 闸门 1：谱截止——剔除插值放大 / 本身就软。这是主判据
    if hf_max(np.asarray(im.convert("L"))) < 0.005:
        return False
    # 闸门 2：仅对 IJG 族表生效，剔除重编码图（RCTW-17 中位 Q75，直接出局）
    q = getattr(im, "quantization", None)
    if q:
        iq = ijg_quality(q[0])
        if iq is not None and iq < 93:
            return False
    # 闸门 3：bpp 交叉验证未被过度压缩
    # 标尺：赛题原生 4.56-5.81 | HDR+ 4.17 | PD12M 3.11 | 赛题GT 2.20 | 4KLSDB 1.6 | Pexels 0.83
    return os.path.getsize(path) * 8 / (im.width * im.height) >= 2.0


class CSIGDataset(Dataset):
    """从原生高分辨率图上随机裁 512x512。只负责出 HQ，退化在 batch transform 里做。"""

    def __init__(self, file_list: str, out_size: int = 512, prompt: str = "",
                 use_hflip: bool = True, image_path_prefix: str = "") -> None:
        with open(file_list, "r") as f:
            self.paths = [l.strip() for l in f if l.strip()]
        self.prefix = image_path_prefix
        self.out_size = out_size
        self.prompt = prompt
        self.use_hflip = use_hflip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        s = self.out_size
        for _ in range(10):  # 少量坏文件直接换一张，不让训练挂掉
            path = self.prefix + self.paths[idx]
            img = cv2.imread(path)
            if img is not None and img.shape[0] >= s and img.shape[1] >= s:
                break
            idx = random.randrange(len(self.paths))
        else:
            raise RuntimeError("连续 10 次读图失败，检查 file_list")

        y = random.randint(0, img.shape[0] - s)
        x = random.randint(0, img.shape[1] - s)
        crop = img[y:y + s, x:x + s]
        if self.use_hflip and random.random() < 0.5:
            crop = crop[:, ::-1]
        hq = torch.from_numpy(crop[..., ::-1].transpose(2, 0, 1).copy()).float() / 255.0
        return {"hq": hq, "txt": self.prompt}


class CSIGBatchTransform:
    """GPU 批量退化。逐样本独立采参数，与 degrade.py 的单图版数学一致。"""

    def __init__(self, hq_key: str = "hq", extra_keys: Optional[List[str]] = None,
                 jpeg_quality: int = JPEG_QUALITY) -> None:
        self.hq_key = hq_key
        self.extra_keys = extra_keys or ["txt"]
        self.jpeg_quality = jpeg_quality
        self.jpeger = None  # 延迟到第一次调用（要拿 device）

    def _lowpass(self, x: torch.Tensor, sigma: torch.Tensor, aniso: torch.Tensor,
                 theta: torch.Tensor, pole: torch.Tensor) -> torch.Tensor:
        """各向异性重尾低通，逐样本参数。x: [B,3,H,W]。

        H(f) = 1 / (1 + (f'/f0)^2)^pole，f' 是按 (aniso, theta) 椭圆缩放后的频率。
        f0 = 0.1325/sigma 是高斯的 -6dB 点换算。纯高斯已被实测拒绝（重尾更贴）。
        """
        b, _, h, w = x.shape
        fy = torch.fft.fftfreq(h, device=x.device).view(1, h, 1)
        fx = torch.fft.fftfreq(w, device=x.device).view(1, 1, w)
        c = torch.cos(theta).view(b, 1, 1)
        s = torch.sin(theta).view(b, 1, 1)
        a = aniso.view(b, 1, 1)
        u = (fx * c + fy * s) / a
        v = -fx * s + fy * c
        f0 = (0.1325 / sigma).view(b, 1, 1)
        resp = 1.0 / (1.0 + (u * u + v * v) / (f0 * f0)) ** pole.view(b, 1, 1)
        return torch.fft.ifft2(torch.fft.fft2(x) * resp.unsqueeze(1)).real

    @torch.no_grad()
    def __call__(self, batch: Dict) -> Dict:
        hq = batch[self.hq_key]
        b, _, h, w = hq.shape
        dev = hq.device

        def U(lo, hi):
            return torch.empty(b, device=dev).uniform_(lo, hi)

        # 1) 低通（含图内空间变化：两档模糊 + 平滑随机掩码混合）
        sigma = U(*SIGMA_RANGE)
        aniso = U(*ANISO_RANGE)
        theta = U(0.0, float(np.pi))
        pole = U(*POLE_RANGE)
        lo = self._lowpass(hq, sigma * (1 - SPATIAL_VAR), aniso, theta, pole)
        hi = self._lowpass(hq, sigma * (1 + SPATIAL_VAR), aniso, theta, pole)
        m = torch.rand(b, 1, 8, 8, device=dev)
        m = F.interpolate(m, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        out = lo * (1 - m) + hi * m

        # 2) 逐通道仿射 + clip（clip 到 0 产生黑位压死，实测真实数据有 4.5-7.8% 像素死在 0）
        do = (torch.rand(b, device=dev) < P_AFFINE).float().view(b, 1, 1, 1)
        ca = torch.empty(b, 3, 1, 1, device=dev).uniform_(*AFFINE_A)
        cb = torch.empty(b, 3, 1, 1, device=dev).uniform_(*AFFINE_B)
        out = out * (1 - do + do * ca) + do * cb
        out = out.clamp(0, 1)

        # 3) JPEG q95（DiffJPEG 与 libjpeg 的 4:2:0 一致）
        if self.jpeger is None:
            from HYPIR.dataset.diffjpeg import DiffJPEG
            self.jpeger = DiffJPEG(differentiable=False).to(dev)
        self.jpeger.to(out)
        q = out.new_full((b,), float(self.jpeg_quality))
        out = self.jpeger(out, quality=q)

        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        return {"GT": hq, "LQ": lq, **{k: batch[k] for k in self.extra_keys}}
