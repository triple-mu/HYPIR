"""PSNR / SSIM / LPIPS / NIQE 评测工具。输入统一为 uint8 BGR。"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/BasicSR")

_lpips_net = None
_dev = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d * d).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def ssim(a, b):
    """BasicSR 口径：逐通道 8bit SSIM，均值（float32 + 可分离滤波加速）。"""
    import cv2
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5).astype(np.float32)
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    f = lambda z: cv2.sepFilter2D(z, -1, k, k)[5:-5, 5:-5]
    mu1, mu2 = f(x), f(y)
    m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = f(x * x) - m1s
    s2 = f(y * y) - m2s
    s12 = f(x * y) - m12
    m = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2))
    return float(m.mean())


def lpips_score(a, b):
    global _lpips_net
    import lpips as L
    if _lpips_net is None:
        _lpips_net = L.LPIPS(net="alex").to(_dev).eval()
    def t(x):
        x = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1])).permute(2, 0, 1)[None]
        return (x.float() / 127.5 - 1.0).to(_dev)
    with torch.no_grad():
        return float(_lpips_net(t(a), t(b)).item())


def niqe(img):
    from basicsr.metrics.niqe import calculate_niqe
    return float(calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y"))
