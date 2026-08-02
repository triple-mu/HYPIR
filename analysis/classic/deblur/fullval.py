"""全图 12MP 验证：候选算子在 3 张完整验证图上的 dPSNR/dSSIM/dLPIPS/dNIQE 与耗时。
LPIPS 用 512x512 分块平均（全图显存放不下），NIQE 用 BasicSR 全图。
"""
import numpy as np
import cv2
import time
import os
import sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ops
import metrics

cv2.setNumThreads(0)
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集/"
PAIRS = [("case1_lq.jpg", "case1_gt.png"), ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]


def load(p):
    return np.array(Image.open(VAL + p).convert("RGB"))[:, :, ::-1].copy()


def lpips_tiled(a, b, t=512):
    H, W = a.shape[:2]
    vs = []
    for y in range(0, H - t + 1, t):
        for x in range(0, W - t + 1, t):
            vs.append(metrics.lpips_score(a[y:y + t, x:x + t], b[y:y + t, x:x + t]))
    return float(np.mean(vs))


CANDS = [
    ("identity", {}),
    ("chroma_blur", dict(s=4.0)),
    ("chroma_blur_luma_blur", dict(sc=4.0, sy=0.5)),
    ("bilat_chroma", dict(d=5, sc=20, ss=5, cs=4.0)),
    ("gblur", dict(s=0.8)),
    ("usm", dict(s=2.0, amt=0.35)),
    ("wiener", dict(s=1.0, nsr=0.01)),
    ("rl", dict(s=1.0, it=3)),
]


def main():
    data = [(load(l), load(g)) for l, g in PAIRS]
    base = []
    for lq, gt in data:
        base.append((metrics.psnr(lq, gt), metrics.ssim(lq, gt),
                     lpips_tiled(lq, gt), metrics.niqe(lq)))
    print("LQ 基线(全图):")
    for (l, _), b in zip(PAIRS, base):
        print(f"  {l}: PSNR={b[0]:.4f} SSIM={b[1]:.5f} LPIPS={b[2]:.4f} NIQE={b[3]:.4f}")
    print(f"  MEAN: PSNR={np.mean([b[0] for b in base]):.4f} SSIM={np.mean([b[1] for b in base]):.5f} "
          f"LPIPS={np.mean([b[2] for b in base]):.4f} NIQE={np.mean([b[3] for b in base]):.4f}\n")

    for spec in CANDS:
        d = []
        t0 = time.perf_counter()
        outs = [ops.apply(spec, lq) for lq, _ in data]
        cost = (time.perf_counter() - t0) / len(data) * 1000
        for (lq, gt), o, b in zip(data, outs, base):
            d.append((metrics.psnr(o, gt) - b[0], metrics.ssim(o, gt) - b[1],
                      lpips_tiled(o, gt) - b[2], metrics.niqe(o) - b[3]))
        m = np.mean(d, 0)
        ps = ",".join(f"{k}={v}" for k, v in spec[1].items())
        print(f"{spec[0]:<22s} {ps:<32s} dPSNR={m[0]:+.4f} dSSIM={m[1]:+.5f} "
              f"dLPIPS={m[2]:+.4f} dNIQE={m[3]:+.3f} cost={cost:.0f}ms")
        print("   per-image dPSNR:", [round(x[0], 4) for x in d],
              " dSSIM:", [round(x[1], 5) for x in d])


if __name__ == "__main__":
    main()
