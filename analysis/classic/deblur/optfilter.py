"""线性滤波上界：最小二乘拟合一个全局 k x k FIR 滤波器 LQ->GT。
这是任意线性移不变算子（含一切反卷积）的理论最优，用来判定该族是否有正收益。
支持留一交叉验证，避免只在拟合集上自欺。
"""
import numpy as np
import cv2
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CROPS = HERE + "/crops/"
CASES = [f"{c}_{i}" for c in ("case1", "case2", "case3") for i in range(5)]


def accum(lq, gt, k, step=2, per_ch=True):
    """返回 (A, b) 或 每通道的列表。A:(k*k+1)^2, b:(k*k+1)"""
    r = k // 2
    n = k * k + 1
    chans = range(3) if per_ch else [None]
    out = []
    for c in chans:
        L = lq[:, :, c].astype(np.float32) if c is not None else lq.astype(np.float32)
        G = gt[:, :, c].astype(np.float32) if c is not None else gt.astype(np.float32)
        L = L / 255.0
        G = G / 255.0
        H, W = L.shape[:2]
        cols = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cols.append(L[r + dy:H - r + dy:step, r + dx:W - r + dx:step].ravel())
        cols.append(np.ones_like(cols[0]))
        X = np.stack(cols, 1)
        y = G[r:H - r:step, r:W - r:step].ravel()
        out.append((X.T @ X, X.T @ y, len(y)))
    return out


def fit(cases, k, step=2, ridge=1e-6):
    n = k * k + 1
    A = [np.zeros((n, n), np.float64) for _ in range(3)]
    B = [np.zeros(n, np.float64) for _ in range(3)]
    for name in cases:
        lq = np.load(f"{CROPS}{name}_lq.npy")
        gt = np.load(f"{CROPS}{name}_gt.npy")
        for c, (a, b, _) in enumerate(accum(lq, gt, k, step)):
            A[c] += a
            B[c] += b
    ws = []
    for c in range(3):
        M = A[c] + ridge * np.trace(A[c]) / n * np.eye(n)
        ws.append(np.linalg.solve(M, B[c]))
    return ws  # 每通道 (k*k+1,)


def apply_ws(x, ws, k):
    xf = x.astype(np.float32) / 255.0
    out = np.empty_like(xf)
    for c in range(3):
        ker = ws[c][:k * k].reshape(k, k).astype(np.float32)
        out[:, :, c] = cv2.filter2D(xf[:, :, c], -1, cv2.flip(ker, -1),
                                    borderType=cv2.BORDER_REFLECT) + ws[c][k * k]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import metrics
    import json
    base = json.load(open(HERE + "/baseline.json"))
    for k in [3, 5, 7, 9, 13]:
        # 留一 case 交叉验证：在另外两个 case 上拟合，测第三个
        dps, dss = [], []
        for hold in ["case1", "case2", "case3"]:
            tr = [n for n in CASES if not n.startswith(hold)]
            te = [n for n in CASES if n.startswith(hold)]
            ws = fit(tr, k)
            for n in te:
                lq = np.load(f"{CROPS}{n}_lq.npy")
                gt = np.load(f"{CROPS}{n}_gt.npy")
                o = apply_ws(lq, ws, k)
                dps.append(metrics.psnr(o, gt) - base[n]["psnr"])
                dss.append(metrics.ssim(o, gt) - base[n]["ssim"])
        # 全量拟合（乐观上界）
        ws_all = fit(CASES, k)
        dpi, dsi = [], []
        for n in CASES:
            lq = np.load(f"{CROPS}{n}_lq.npy")
            gt = np.load(f"{CROPS}{n}_gt.npy")
            o = apply_ws(lq, ws_all, k)
            dpi.append(metrics.psnr(o, gt) - base[n]["psnr"])
            dsi.append(metrics.ssim(o, gt) - base[n]["ssim"])
        ker = ws_all[1][:k * k].reshape(k, k)
        print(f"k={k:2d}  LOCV dPSNR={np.mean(dps):+.4f} dSSIM={np.mean(dss):+.5f} | "
              f"in-fit dPSNR={np.mean(dpi):+.4f} dSSIM={np.mean(dsi):+.5f} | "
              f"kernel sum={ker.sum():.4f} center={ker[k//2,k//2]:.4f}")
        np.save(f"{HERE}/ws_k{k}.npy", np.array(ws_all))
