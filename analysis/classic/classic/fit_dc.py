"""DC 约束（sum(w)=1, 无 bias）的全局 FIR 拟合 + 留一交叉验证。纯空间形状，不含色调。"""
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E
from fit_big import accum_fft, make_fn


def solve_dc(A, b, lam):
    """在 sum(w)=1 约束下解 min ||Aw-b||，A/b 已含 bias 行列，这里丢弃 bias 维。"""
    A, b = A[:-1, :-1].copy(), b[:-1].copy()
    n = A.shape[0]
    A = A + lam * np.trace(A) / n * np.eye(n)
    one = np.ones(n)
    KKT = np.zeros((n + 1, n + 1))
    KKT[:n, :n] = A
    KKT[:n, n] = one
    KKT[n, :n] = one
    rhs = np.concatenate([b, [1.0]])
    w = np.linalg.solve(KKT, rhs)[:n]
    return np.concatenate([w, [0.0]])


if __name__ == "__main__":
    crops = E.crop_set(1024, 3)
    for r in (2, 4, 6, 10, 14):
        per = {t: accum_fft(l, g, r) for t, l, g in crops}
        for lam in (1e-3, 1e-1):
            A = sum(p[0] for p in per.values())
            b = sum(p[1] for p in per.values())
            w = solve_dc(A, b, lam)
            fn = make_fn(w, r)
            m, _ = E.evaluate(fn, crops, with_lpips=False)
            pc = E.per_case(fn, crops)
            print(f"[r={r} lam={lam}] 全量 dPSNR {m['d_psnr']:+.4f} dSSIM {m['d_ssim']:+.5f} "
                  f"| worst {min(x[1] for x in pc):+.3f}/{min(x[2] for x in pc):+.5f} center={w[len(w)//2]:.4f}")
            loo = []
            for hold in E.CASES:
                Ah = sum(p[0] for t, p in per.items() if not t.startswith(hold))
                bh = sum(p[1] for t, p in per.items() if not t.startswith(hold))
                wh = solve_dc(Ah, bh, lam)
                sub = [c for c in crops if c[0].startswith(hold)]
                mh, _ = E.evaluate(make_fn(wh, r), sub, with_lpips=False)
                loo.append((hold, mh["d_psnr"], mh["d_ssim"]))
            print("      LOO: " + "  ".join(f"{h} {p:+.4f}/{s:+.5f}" for h, p, s in loo))
            np.save(f"/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE/dc_r{r}_l{lam}.npy", w)
