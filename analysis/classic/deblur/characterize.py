"""刻画退化：对齐检查、最优高斯 sigma 拟合、残差噪声估计、色调偏移。"""
import numpy as np
import cv2
import glob
import os

CROPS = "/home/ubuntu/workspace/contest/CSIG-2026/deblur_search/crops/"


def shift_est(lq, gt):
    a = cv2.cvtColor(lq, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * w, b * w)
    return dx, dy, resp


def fit_sigma(lq, gt):
    """找使 blur(gt,sigma) 最接近 lq 的 sigma。"""
    g = gt.astype(np.float32)
    l = lq.astype(np.float32)
    best = None
    for s in np.arange(0.2, 3.01, 0.1):
        k = int(2 * round(3 * s) + 1)
        bl = cv2.GaussianBlur(g, (k, k), s)
        e = float(((bl - l) ** 2).mean())
        if best is None or e < best[1]:
            best = (s, e)
    return best


def main():
    for case in ["case1", "case2", "case3"]:
        for i in range(5):
            lq = np.load(f"{CROPS}{case}_{i}_lq.npy")
            gt = np.load(f"{CROPS}{case}_{i}_gt.npy")
            dx, dy, r = shift_est(lq, gt)
            s, e = fit_sigma(lq, gt)
            # 残差（blur(gt,s) vs lq）的高频能量 ~ 噪声
            k = int(2 * round(3 * s) + 1)
            res = cv2.GaussianBlur(gt.astype(np.float32), (k, k), s) - lq.astype(np.float32)
            noise = res.std()
            dmean = (lq.astype(np.float64).mean((0, 1)) - gt.astype(np.float64).mean((0, 1)))
            print(f"{case}_{i}: shift=({dx:+.2f},{dy:+.2f}) r={r:.2f}  sigma*={s:.2f} rmse={e**0.5:.2f} "
                  f"noise_std={noise:.2f}  dBGR={dmean.round(2)}")


if __name__ == "__main__":
    main()
