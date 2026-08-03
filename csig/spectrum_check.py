"""用径向平均功率谱定位真实 LQ 的退化强度，看它落在合成分布的哪里。

目视发现真实 LQ 比大多数合成样本更糊，而 FR/NR 的均值却"匹配良好"——两者必有一个
误导。功率谱是直接量"高频被削掉多少"的工具，不像 FR/NR 那样把很多因素揉成一个标量。

做法：对每张 GT 施加一系列**已知强度**的纯重采样，量各自的谱；再量真实 LQ 的谱，
用最小二乘找出与它最接近的那个强度。得到的「等效倍率」直接说明真实退化的位置。

    python csig/spectrum_check.py --val <验证集目录>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def radial_psd(img, nbins=64):
    """灰度图的径向平均功率谱（对数），返回 (freq, logpsd)。"""
    g = img.mean(1, keepdim=True)                       # (1,1,H,W)
    G = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
    p = (G.real ** 2 + G.imag ** 2)[0, 0]
    H, W = p.shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=p.device) - cy,
                          torch.arange(W, device=p.device) - cx, indexing="ij")
    r = torch.sqrt((y.float() / cy) ** 2 + (x.float() / cx) ** 2)     # 归一化到 [0, ~1.41]
    rb = (r / 1.0 * nbins).clamp(0, nbins - 1).long()
    out = torch.zeros(nbins, device=p.device)
    cnt = torch.zeros(nbins, device=p.device)
    out.scatter_add_(0, rb.flatten(), p.flatten())
    cnt.scatter_add_(0, rb.flatten(), torch.ones_like(p.flatten()))
    psd = out / cnt.clamp(min=1)
    f = (torch.arange(nbins, device=p.device).float() + 0.5) / nbins
    return f.cpu().numpy(), torch.log10(psd.clamp(min=1e-12)).cpu().numpy()


def resample(x, s, dm="bicubic", um="bicubic"):
    h, w = x.shape[-2:]
    kw = {} if dm == "area" else {"align_corners": False}
    small = F.interpolate(x, size=(max(int(h / s), 8), max(int(w / s), 8)), mode=dm, **kw)
    return F.interpolate(small, size=(h, w), mode=um, align_corners=False).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device("cuda")
    S = a.crop
    # 只比中高频段：低频被内容主导，对退化不敏感
    scales = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0]

    print("%-8s %12s %12s %10s" % ("图", "真实LQ等效倍率", "v2 分布范围", "落点"))
    print("-" * 50)
    allfit = []
    for lqp in sorted(glob.glob(os.path.join(a.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
        gt_f = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
        H, W = gt_f.shape[:2]
        y, x = (H - S) // 2, (W - S) // 2
        gt = torch.from_numpy(gt_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
        lq = torch.from_numpy(lq_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.

        f, p_gt = radial_psd(gt)
        _, p_lq = radial_psd(lq)
        band = (f > 0.15) & (f < 0.8)
        # 以 GT 为基准看衰减量，去掉内容差异
        d_real = (p_lq - p_gt)[band]
        best, berr = None, 1e18
        for s in scales:
            _, p_s = radial_psd(resample(gt, s))
            e = float(np.mean((p_s - p_gt)[band] - d_real) ** 2 +
                      np.mean(((p_s - p_gt)[band] - d_real) ** 2))
            if e < berr:
                best, berr = s, e
        allfit.append(best)
        inside = "分布内" if 1.6 <= best <= 7.0 else ("**超出上界 7.0**" if best > 7.0 else "低于下界")
        print("%-8s %12.1fx %12s %10s" % (base, best, "1.6 - 7.0", inside))

    print("\n真实 LQ 的等效倍率: %s，中位 %.1fx" % (allfit, float(np.median(allfit))))
    print("v2 的 SCALE_RANGE = (1.6, 7.0)，均匀采样 -> 期望 4.3x")
    print("\n各频段衰减（以 case1 为例，负值=高频被削掉多少个数量级）:")
    lqp = sorted(glob.glob(os.path.join(a.val, "*_lq.jpg")))[0]
    base = os.path.basename(lqp).replace("_lq.jpg", "")
    gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
    gt_f = cv2.imread(gtp)[:, :, ::-1].copy(); lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
    H, W = gt_f.shape[:2]; y, x = (H-S)//2, (W-S)//2
    gt = torch.from_numpy(gt_f[y:y+S,x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
    lq = torch.from_numpy(lq_f[y:y+S,x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
    f, p_gt = radial_psd(gt); _, p_lq = radial_psd(lq)
    rows = {"真实LQ": p_lq - p_gt}
    for s in (2.0, 4.3, 7.0, 11.0):
        _, ps = radial_psd(resample(gt, s)); rows["合成%.1fx" % s] = ps - p_gt
    idx = [int(nb * len(f)) for nb in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    print("   %-10s %s" % ("归一化频率", "  ".join("%6.2f" % f[i] for i in idx)))
    for k, v in rows.items():
        print("   %-10s %s" % (k, "  ".join("%6.2f" % v[i] for i in idx)))


if __name__ == "__main__":
    main()
