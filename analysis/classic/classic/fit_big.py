"""用 FFT 累积正规方程，拟合大尺寸全局线性滤波器（含 bias），测线性反卷积的真实上限。"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E


def _corr(F1, F2, r, N):
    """返回 t in [-r,r]^2 的 sum_x a(x) b(x+t)，F1=conj(FFT(a)) 已取共轭。"""
    c = np.fft.irfft2(F1 * F2, s=N)
    out = np.zeros((2 * r + 1, 2 * r + 1))
    for i, dy in enumerate(range(-r, r + 1)):
        for j, dx in enumerate(range(-r, r + 1)):
            out[i, j] = c[dy % N[0], dx % N[1]]
    return out


def accum_fft(lq, gt, r):
    """特征 = LQ 的 (2r+1)^2 邻域 + 常数 1。返回 (A, b)。"""
    K = 2 * r + 1
    n = K * K
    H, W = lq.shape[:2]
    N = (H + 4 * r + 2, W + 4 * r + 2)
    A = np.zeros((n + 1, n + 1))
    bb = np.zeros(n + 1)
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    for c in range(3):
        L = lq[:, :, c].astype(np.float64)
        G = gt[:, :, c].astype(np.float64)
        FL = np.fft.rfft2(L, s=N)
        FG = np.fft.rfft2(G, s=N)
        RLL = _corr(np.conj(FL), FL, 2 * r, N)     # R_LL(t), t in [-2r,2r]
        C = _corr(np.conj(FG), FL, r, N)           # C(t) = sum G(x) L(x+t)
        for i, (dy1, dx1) in enumerate(offs):      # A[d',d] = R_LL(d-d')
            for j, (dy2, dx2) in enumerate(offs):
                A[i, j] += RLL[(dy2 - dy1) + 2 * r, (dx2 - dx1) + 2 * r]
        A[:n, n] += L.sum()
        A[n, :n] += L.sum()
        A[n, n] += L.size
        bb[:n] += C.reshape(-1)
        bb[n] += G.sum()
    return A, bb


def solve(A, b, lam):
    n = A.shape[0]
    reg = lam * np.trace(A[:-1, :-1]) / (n - 1) * np.eye(n)
    reg[-1, -1] = 0
    return np.linalg.solve(A + reg, b)


def make_fn(w, r, use_bias=True):
    K = 2 * r + 1
    k = np.ascontiguousarray(w[:-1].reshape(K, K).astype(np.float32))
    bias = float(w[-1]) if use_bias else 0.0

    def fn(img):
        o = cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT) + bias
        return np.clip(o + 0.5, 0, 255).astype(np.uint8)
    return fn


if __name__ == "__main__":
    crops = E.crop_set(1024, 3)
    R = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "6,10").split(",")]
    for r in R:
        per = {t: accum_fft(l, g, r) for t, l, g in crops}
        for lam in (1e-4, 1e-2):
            A = sum(p[0] for p in per.values())
            b = sum(p[1] for p in per.values())
            w = solve(A, b, lam)
            m, _ = E.evaluate(make_fn(w, r), crops, with_lpips=False)
            m0, _ = E.evaluate(make_fn(w, r, False), crops, with_lpips=False)
            print(f"[r={r} lam={lam}] 全量拟合 dPSNR {m['d_psnr']:+.3f} dSSIM {m['d_ssim']:+.4f} | 去bias {m0['d_psnr']:+.3f}/{m0['d_ssim']:+.4f}  ksum={w[:-1].sum():.3f} bias={w[-1]:+.2f}")
            for hold in E.CASES:
                Ah = sum(p[0] for t, p in per.items() if not t.startswith(hold))
                bh = sum(p[1] for t, p in per.items() if not t.startswith(hold))
                wh = solve(Ah, bh, lam)
                sub = [c for c in crops if c[0].startswith(hold)]
                mh, _ = E.evaluate(make_fn(wh, r), sub, with_lpips=False)
                print(f"     hold {hold}: dPSNR {mh['d_psnr']:+.3f} dSSIM {mh['d_ssim']:+.4f}")
            # 单案例自拟合上限
            for c in E.CASES:
                Ac = sum(p[0] for t, p in per.items() if t.startswith(c))
                bc = sum(p[1] for t, p in per.items() if t.startswith(c))
                wc = solve(Ac, bc, lam)
                sub = [x for x in crops if x[0].startswith(c)]
                mc, _ = E.evaluate(make_fn(wc, r), sub, with_lpips=False)
                print(f"     self {c}(上限): dPSNR {mc['d_psnr']:+.3f} dSSIM {mc['d_ssim']:+.4f}")
            np.save(f"/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE/big_r{r}_l{lam}.npy", w)
