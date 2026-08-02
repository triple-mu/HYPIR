"""在验证集 crop 上最小二乘拟合一个全局 KxK FIR 核（+bias），并做留一交叉验证。"""
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E


def accum(lq, gt, r):
    """累加正规方程 (A,b,n)：特征 = LQ 的 (2r+1)^2 邻域 + 常数 1。"""
    K = 2 * r + 1
    n = K * K
    L = lq.astype(np.float64)
    G = gt.astype(np.float64)
    H, W = L.shape[:2]
    # 有效区域去掉边界 r
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for c in range(L.shape[2]):
        Lc, Gc = L[:, :, c], G[:, :, c]
        cols = np.stack(
            [Lc[r + dy:H - r + dy, r + dx:W - r + dx].ravel()
             for dy in range(-r, r + 1) for dx in range(-r, r + 1)] + [np.ones((H - 2 * r) * (W - 2 * r))],
            axis=1,
        )
        y = Gc[r:H - r, r:W - r].ravel()
        A += cols.T @ cols
        b += cols.T @ y
    return A, b


def solve(A, b, lam=1e-3):
    n = A.shape[0]
    reg = lam * np.trace(A) / n * np.eye(n)
    reg[-1, -1] = 0
    return np.linalg.solve(A + reg, b)


def make_fn(w, r):
    import cv2
    K = 2 * r + 1
    k = w[:-1].reshape(K, K).astype(np.float32)
    bias = float(w[-1])

    def fn(img):
        # filter2D 做的是相关，与拟合时 pred(x)=sum_d w[d]*L(x+d) 的约定一致，不需翻转
        o = cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT) + bias
        return np.clip(o + 0.5, 0, 255).astype(np.uint8)

    return fn


if __name__ == "__main__":
    crops = E.crop_set(1024, 3)
    for r in (1, 2, 3, 4):
        per = {}
        for tag, lq, gt in crops:
            per[tag] = accum(lq, gt, r)
        # 全量拟合
        A = sum(p[0] for p in per.values())
        b = sum(p[1] for p in per.values())
        w = solve(A, b)
        fn = make_fn(w, r)
        m, base = E.evaluate(fn, crops, with_lpips=False)
        print(f"[r={r} full-fit] dPSNR {m['d_psnr']:+.3f} dSSIM {m['d_ssim']:+.4f}  bias={w[-1]:+.2f} ksum={w[:-1].sum():.3f}")
        # 留一案例交叉验证
        for hold in E.CASES:
            Ah = sum(p[0] for t, p in per.items() if not t.startswith(hold))
            bh = sum(p[1] for t, p in per.items() if not t.startswith(hold))
            wh = solve(Ah, bh)
            sub = [c for c in crops if c[0].startswith(hold)]
            mh, _ = E.evaluate(make_fn(wh, r), sub, with_lpips=False)
            print(f"    hold {hold}: dPSNR {mh['d_psnr']:+.3f} dSSIM {mh['d_ssim']:+.4f}")
        np.save(f"/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE/fir_r{r}.npy", w)
