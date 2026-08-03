"""合成退化分布是否**逐频段**覆盖住真实退化。

这是替代 coverage_check.py 的正确工具。之前用 FR/NR 两个标量判覆盖，
把不同频段的差异揉成一个数，得出了「匹配良好」的错误结论；实际用 1D MTF 一量，
真实退化的等效倍率是 5.0-8.0，而当时的 SCALE_RANGE 是 (1.6, 7.0)——
过半样本比真实轻，最重的真实样本还在分布之外。

判据：对每个频率，看真实 MTF 落在合成 MTF 分布的第几百分位。
落在 10-90 之间才算覆盖住；系统性地贴近 0 或 100 说明分布整体偏了。

    python csig/mtf_coverage.py --val <验证集目录> [--draws 60]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from degrade_vis import degrade
from HYPIR.dataset.csig import (CSIGBatchTransform, SCALE_RANGE, POLE_RANGE,
                                SIGMA_RANGE, P_RESAMPLE)


def mtf1d(gt, lq, axis=1):
    g = gt.mean(1)[0].double()
    l = lq.mean(1)[0].double()
    if axis == 0:
        g, l = g.T, l.T
    n = g.shape[-1]
    w = torch.hann_window(n, periodic=False, dtype=torch.float64, device=g.device)
    G = torch.fft.rfft((g - g.mean(-1, keepdim=True)) * w, dim=-1)
    L = torch.fft.rfft((l - l.mean(-1, keepdim=True)) * w, dim=-1)
    return ((L * G.conj()).real.sum(0) / (G.abs() ** 2).sum(0).clamp(min=1e-30)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--crop", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tf = CSIGBatchTransform()
    S = a.crop
    probe = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print("当前配方: SCALE_RANGE=%s  POLE_RANGE=%s  SIGMA_RANGE=%s  P_RESAMPLE=%.2f\n"
          % (SCALE_RANGE, POLE_RANGE, SIGMA_RANGE, P_RESAMPLE))
    all_pct = []
    for lqp in sorted(glob.glob(os.path.join(a.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
        gt_f = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
        H, W = gt_f.shape[:2]
        y, x = (H - S) // 2, (W - S) // 2
        gt = torch.from_numpy(gt_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.
        lq = torch.from_numpy(lq_f[y:y+S, x:x+S]).permute(2,0,1)[None].float().to(dev)/255.

        m_real = mtf1d(gt, lq)
        f = np.arange(len(m_real)) / (len(m_real) - 1)
        syn = np.stack([mtf1d(gt, degrade(gt, tf, seed=7000 + k)[0]) for k in range(a.draws)])

        idx = [int(np.argmin(np.abs(f - v))) for v in probe]
        pct = [float((syn[:, i] < m_real[i]).mean() * 100) for i in idx]
        all_pct.append(pct)
        print("=== %s ===" % base)
        print("  频率      " + " ".join("%6.2f" % f[i] for i in idx))
        print("  真实 MTF  " + " ".join("%6.3f" % m_real[i] for i in idx))
        print("  合成 中位 " + " ".join("%6.3f" % np.median(syn[:, i]) for i in idx))
        print("  合成 P10  " + " ".join("%6.3f" % np.percentile(syn[:, i], 10) for i in idx))
        print("  合成 P90  " + " ".join("%6.3f" % np.percentile(syn[:, i], 90) for i in idx))
        print("  真实落点  " + " ".join("%5.0f%%" % p for p in pct))
        bad = sum(1 for p in pct if p < 10 or p > 90)
        print("  -> %d/%d 个频点落在 10-90 分位之外 %s\n"
              % (bad, len(pct), "（覆盖不足）" if bad > len(pct) // 3 else "（可接受）"))

    m = np.array(all_pct).mean(0)
    print("三张平均的落点百分位: " + " ".join("%5.0f%%" % v for v in m))
    lo = (m < 10).sum(); hi = (m > 90).sum()
    if lo > len(m) // 3:
        print("=> 真实 MTF 系统性低于合成分布：合成退化**太轻**，需调强（加大 SCALE/SIGMA）")
    elif hi > len(m) // 3:
        print("=> 真实 MTF 系统性高于合成分布：合成退化**太重**，需调弱")
    else:
        print("=> 分布覆盖住了真实退化")


if __name__ == "__main__":
    main()
