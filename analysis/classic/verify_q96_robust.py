"""稳健性检查：3x3 网格多 crop 上的符号一致性 + JPEG quality 扫描。"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"


def params(q):
    return [cv2.IMWRITE_JPEG_QUALITY, q,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]


def load_bgr(p):
    return np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy()


def gt_path(c):
    for e in (".png", ".jpg"):
        p = os.path.join(VAL, f"{c}_gt{e}")
        if os.path.exists(p):
            return p


def grid_crops(img, s=1024, n=3):
    h, w = img.shape[:2]
    out = []
    for i in range(n):
        for j in range(n):
            y = int((h - s) * (i + 0.5) / n) // 8 * 8
            x = int((w - s) * (j + 0.5) / n) // 8 * 8
            out.append(((i, j), (y, x)))
    return out


def main():
    data = []
    for c in ["case1", "case2", "case3"]:
        data.append((c, load_bgr(os.path.join(VAL, f"{c}_lq.jpg")), load_bgr(gt_path(c))))

    print("=== A. 3x3 网格 crop 符号一致性 (q=96) ===")
    tot_p, tot_s, neg_p, neg_s, npos = [], [], 0, 0, 0
    for c, lq, gt in data:
        dps, dss = [], []
        for (i, j), (y, x) in grid_crops(lq):
            a = lq[y:y + 1024, x:x + 1024]
            g = gt[y:y + 1024, x:x + 1024]
            o = cv2.imdecode(cv2.imencode(".jpg", a, params(96))[1], cv2.IMREAD_COLOR)
            dp = sk_psnr(g, o, data_range=255) - sk_psnr(g, a, data_range=255)
            ds = (sk_ssim(g, o, data_range=255, channel_axis=2)
                  - sk_ssim(g, a, data_range=255, channel_axis=2))
            dps.append(dp); dss.append(ds)
            neg_p += dp < 0; neg_s += ds < 0; npos += 1
        tot_p += dps; tot_s += dss
        print(f"{c}: dPSNR mean {np.mean(dps):+.4f} min {np.min(dps):+.4f} max {np.max(dps):+.4f} "
              f"neg {sum(1 for v in dps if v < 0)}/9 | "
              f"dSSIM mean {np.mean(dss):+.6f} min {np.min(dss):+.6f} neg {sum(1 for v in dss if v < 0)}/9")
    print(f"ALL 27 crops: dPSNR<0 in {neg_p}/{npos}, dSSIM<0 in {neg_s}/{npos}")

    print("\n=== B. quality 扫描（中心 1024 crop，三图平均） ===")
    print(f"{'q':>4} {'meanDPSNR':>11} {'meanDSSIM':>12}   per-image dPSNR")
    for q in [88, 90, 92, 94, 95, 96, 97, 98, 99, 100]:
        dps, dss = [], []
        for c, lq, gt in data:
            h, w = lq.shape[:2]
            y = ((h - 1024) // 2) // 8 * 8; x = ((w - 1024) // 2) // 8 * 8
            a = lq[y:y + 1024, x:x + 1024]; g = gt[y:y + 1024, x:x + 1024]
            o = cv2.imdecode(cv2.imencode(".jpg", a, params(q))[1], cv2.IMREAD_COLOR)
            dps.append(sk_psnr(g, o, data_range=255) - sk_psnr(g, a, data_range=255))
            dss.append(sk_ssim(g, o, data_range=255, channel_axis=2)
                       - sk_ssim(g, a, data_range=255, channel_axis=2))
        print(f"{q:>4} {np.mean(dps):+11.4f} {np.mean(dss):+12.6f}   "
              + " ".join(f"{v:+.4f}" for v in dps))

    print("\n=== C. 4:4:4 vs 4:2:0 (q=96, 中心 crop) ===")
    for name, sf in [("444", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
                     ("420", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)]:
        dps, dss = [], []
        for c, lq, gt in data:
            h, w = lq.shape[:2]
            y = ((h - 1024) // 2) // 8 * 8; x = ((w - 1024) // 2) // 8 * 8
            a = lq[y:y + 1024, x:x + 1024]; g = gt[y:y + 1024, x:x + 1024]
            p = [cv2.IMWRITE_JPEG_QUALITY, 96, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sf,
                 cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
            o = cv2.imdecode(cv2.imencode(".jpg", a, p)[1], cv2.IMREAD_COLOR)
            dps.append(sk_psnr(g, o, data_range=255) - sk_psnr(g, a, data_range=255))
            dss.append(sk_ssim(g, o, data_range=255, channel_axis=2)
                       - sk_ssim(g, a, data_range=255, channel_axis=2))
        print(f"{name}: meanDPSNR {np.mean(dps):+.4f} meanDSSIM {np.mean(dss):+.6f}  "
              + " ".join(f"{v:+.4f}" for v in dps))


if __name__ == "__main__":
    main()
