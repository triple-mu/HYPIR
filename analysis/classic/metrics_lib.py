"""赛题二评测指标：PSNR / SSIM / LPIPS / NIQE，统一在 uint8 BGR 上取值。"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BasicSR"))

_lpips_net = None
_niqe_params = None


def psnr(a, b):
    """a, b: uint8 BGR 同尺寸。"""
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def ssim(a, b):
    """Wang 标准参数（gaussian 11x11 sigma1.5），三通道均值。"""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        a, b, channel_axis=2, data_range=255,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def lpips_score(a, b, tile=1024, device="cuda"):
    """分块 LPIPS（AlexNet），块面积加权平均。"""
    global _lpips_net
    if _lpips_net is None:
        import lpips as lpips_mod
        _lpips_net = lpips_mod.LPIPS(net="alex").to(device).eval()
    h, w = a.shape[:2]
    tot, area = 0.0, 0
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                pa, pb = a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]
                if min(pa.shape[:2]) < 64:
                    continue
                ta = torch.from_numpy(pa[:, :, ::-1].copy()).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
                tb = torch.from_numpy(pb[:, :, ::-1].copy()).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
                v = float(_lpips_net(ta, tb).item())
                n = pa.shape[0] * pa.shape[1]
                tot += v * n
                area += n
    return tot / area


def niqe(img):
    """BasicSR NIQE，输入 uint8 BGR。"""
    from basicsr.metrics.niqe import calculate_niqe
    return float(calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y"))
