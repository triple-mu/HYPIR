"""量真实退化的 MTF 并用参数族拟合，判定它属于哪一类模糊。

反解空域 PSF 对重模糊不可靠（高频被削光 -> 病态 -> 解出又宽又不衰减的假核，
case1/case3 就是这样）。MTF |K(f)| = |FFT(LQ)|/|FFT(GT)| 的径向平均是良态的，
且能直接和参数族比：

    高斯       exp(-(f/f0)^2 / 2)
    重尾极点   1 / (1 + (f/f0)^2)^p          v1 标定时选的这个
    重采样     |sinc| 型，在 1/s 处有零点与旁瓣
    盒式平均   |sinc(pi f n)|

同时量合成 v2 的两条支路，看它们和真实的 MTF 差在哪。

    python csig/mtf_fit.py --val <验证集目录>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def radial_mtf(gt, lq, nbins=96):
    """|FFT(LQ)|/|FFT(GT)| 的径向平均。用能量加权，低能量频点不参与。"""
    win_y = torch.hann_window(gt.shape[-2], periodic=False, dtype=torch.float64, device=gt.device)
    win_x = torch.hann_window(gt.shape[-1], periodic=False, dtype=torch.float64, device=gt.device)
    win = win_y[:, None] * win_x[None, :]
    g = (gt.mean(1)[0].double() - gt.mean()) * win
    l = (lq.mean(1)[0].double() - lq.mean()) * win
    G = torch.fft.fftshift(torch.fft.fft2(g)).abs()
    L = torch.fft.fftshift(torch.fft.fft2(l)).abs()
    H, W = G.shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=G.device) - cy,
                          torch.arange(W, device=G.device) - cx, indexing="ij")
    # 归一化到奈奎斯特 = 1
    r = torch.sqrt((y.double() / cy) ** 2 + (x.double() / cx) ** 2) / np.sqrt(2)
    rb = (r * nbins).clamp(0, nbins - 1).long().flatten()
    num = torch.zeros(nbins, dtype=torch.float64, device=G.device)
    den = torch.zeros(nbins, dtype=torch.float64, device=G.device)
    num.scatter_add_(0, rb, (L * G).flatten())      # 能量加权：sum(L*G)/sum(G^2)
    den.scatter_add_(0, rb, (G * G).flatten())
    mtf = (num / den.clamp(min=1e-30)).cpu().numpy()
    f = (np.arange(nbins) + 0.5) / nbins
    return f, mtf


def fit_families(f, mtf, band):
    """在 band 内用三个参数族拟合，返回 {名字: (参数, RMSE)}。"""
    ff, mm = f[band], np.clip(mtf[band], 1e-6, None)
    out = {}
    # 高斯
    best = (None, 1e18)
    for f0 in np.linspace(0.02, 1.0, 200):
        pred = np.exp(-(ff / f0) ** 2 / 2)
        e = float(np.sqrt(np.mean((pred - mm) ** 2)))
        if e < best[1]:
            best = (f0, e)
    out["高斯"] = ("f0=%.3f" % best[0], best[1])
    # 重尾极点
    best = (None, None, 1e18)
    for f0 in np.linspace(0.02, 1.0, 120):
        for p in np.linspace(0.3, 4.0, 60):
            pred = 1.0 / (1.0 + (ff / f0) ** 2) ** p
            e = float(np.sqrt(np.mean((pred - mm) ** 2)))
            if e < best[2]:
                best = (f0, p, e)
    out["重尾极点"] = ("f0=%.3f p=%.2f" % (best[0], best[1]), best[2])
    # 重采样（理想带限：截止在 1/s，之后为 0；实际有旁瓣，这里用 sinc 包络近似）
    best = (None, 1e18)
    for s in np.linspace(1.2, 12.0, 200):
        fc = 1.0 / s
        pred = np.abs(np.sinc(ff / fc / 2)) * (ff < fc * 2)
        e = float(np.sqrt(np.mean((pred - mm) ** 2)))
        if e < best[1]:
            best = (s, e)
    out["重采样sinc"] = ("s=%.2f" % best[0], best[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    def to_t(x):
        return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255.

    show = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]
    for lqp in sorted(glob.glob(os.path.join(a.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
        gt = to_t(cv2.imread(gtp)[:, :, ::-1].copy())
        lq = to_t(cv2.imread(lqp)[:, :, ::-1].copy())
        f, mtf = radial_mtf(gt, lq)
        idx = [int(np.argmin(np.abs(f - v))) for v in show]
        print("\n=== %s ===" % base)
        print("  频率(奈奎斯特=1) " + " ".join("%6.2f" % f[i] for i in idx))
        print("  真实 MTF         " + " ".join("%6.3f" % mtf[i] for i in idx))
        # -3dB / -6dB 截止
        def cutoff(th):
            below = np.where(mtf < th)[0]
            return f[below[0]] if len(below) else 1.0
        print("  截止频率: -3dB %.3f   -6dB %.3f   -20dB %.3f"
              % (cutoff(0.708), cutoff(0.5), cutoff(0.1)))
        band = (f > 0.03) & (f < 0.9)
        fits = fit_families(f, mtf, band)
        print("  参数族拟合（RMSE 越小越贴）:")
        for k, (par, e) in sorted(fits.items(), key=lambda kv: kv[1][1]):
            print("     %-10s %-18s RMSE %.4f" % (k, par, e))

        # 对照：合成两条支路
        for tag, syn in (("重采样4.3x", None), ("重尾低通", None)):
            pass
        h, w = gt.shape[-2:]
        for s in (2.0, 4.3, 7.0):
            sm = F.interpolate(gt, size=(int(h/s), int(w/s)), mode="bicubic", align_corners=False)
            up = F.interpolate(sm, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
            _, m2 = radial_mtf(gt, up)
            print("  合成重采样%.1fx    " % s + " ".join("%6.3f" % m2[i] for i in idx))


if __name__ == "__main__":
    main()
