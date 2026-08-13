"""HQ corpus screening helpers plus compatibility re-exports.

The old copy of the training degradation lived here and drifted away from the
actual training module.  ``HYPIR.dataset.csig`` is now the single source of truth;
this file keeps only the corpus-quality gates used by preparation scripts.
"""

import os
import sys
from typing import Optional

import numpy as np


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


def source_bpp(path: str, image=None) -> tuple[float, bool]:
    """Return comparable primary-image bpp and whether this is ISO gain-map MPO.

    Huawei MPO files contain roughly 5.8--6.1 MB of unindexed private tail data.
    Counting that tail as JPEG payload inflated the old 4.56--5.81 bpp estimate.
    For a valid ISO 21496 gain-map MPO we report frame-0 bpp and mark it so the
    single-frame corpus threshold can be skipped; its vendor qtable and container
    are not comparable to an ordinary IJG JPEG.
    """

    from PIL import Image

    im = image or Image.open(path)
    is_gain_map_mpo = False
    payload_size = os.path.getsize(path)
    if getattr(im, "format", None) == "MPO":
        try:
            from csig.mpo_hdr import inspect_file

            info = inspect_file(path)
            if info.frames:
                payload_size = info.frames[0].size
            is_gain_map_mpo = info.gain_map_frame_index is not None
        except (OSError, ValueError):
            # Screening is a heuristic.  A malformed MPO will still face the
            # conservative whole-file threshold and later decoder validation.
            pass
    return payload_size * 8.0 / (im.width * im.height), is_gain_map_mpo


def screen(path: str) -> bool:
    """三道闸门。任何一道不过就别拿来当训练 GT。"""
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
    # 闸门 3：单帧普通图用 bpp 交叉验证未被过度压缩。ISO gain-map MPO
    # 使用厂商量化表且带独立辅助帧，不能套同一阈值。
    bpp, is_gain_map_mpo = source_bpp(path, im)
    return is_gain_map_mpo or bpp >= 2.0


# Keep direct ``python csig/<tool>.py`` entry points working: for those commands
# Python puts ``csig/`` rather than the repository root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep old import sites working without maintaining a second degradation copy.
from HYPIR.dataset.csig import CSIGBatchTransform, CSIGDataset, csig_collate  # noqa: E402,F401
from HYPIR.dataset.csig_degradation import *  # noqa: E402,F401,F403


__all__ = [
    "CSIGBatchTransform", "CSIGDataset", "csig_collate", "hf_max", "ijg_quality", "screen",
    "source_bpp",
]
