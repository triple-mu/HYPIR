"""在 3 张完整 12MP 验证图上跑最终候选：PSNR/SSIM/LPIPS/NIQE + 真实耗时。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E
import ops as O
from fit_big import make_fn

W10 = np.load("/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE/dc_r10_l0.001.npy")

CANDS = [
    ("identity", None),
    ("A creup(2,16)+guided(1,4)", O.chain(O.chroma_reup(2, 16), O.guided(1, 4))),
    ("B creup(2,16)+guided(1,8)", O.chain(O.chroma_reup(2, 16), O.guided(1, 8))),
    ("C creup(2,64)+guided(1,12)", O.chain(O.chroma_reup(2, 64), O.guided(1, 12))),
    ("D cguided(4,16)+guided(1,4)", O.chain(O.chroma_guided(4, 16), O.guided(1, 4))),
    ("E creup(2,16) only", O.chroma_reup(2, 16)),
    ("ref gblur(0.8)", O.gblur(0.8)),
    ("ref usm(0.2,1.5)", O.usm(0.2, 1.5)),
    ("ref fitted-lowpass r10", make_fn(W10, 10)),
]

if __name__ == "__main__":
    pairs = [(c, E.load_full(c, "lq"), E.load_full(c, "gt")) for c in E.CASES]
    res = {}
    for name, fn in CANDS:
        rows, cost = [], 0.0
        for c, lq, gt in pairs:
            t = time.perf_counter()
            o = lq if fn is None else fn(lq)
            cost += (time.perf_counter() - t) * 1000 * 12e6 / (lq.shape[0] * lq.shape[1])
            rows.append((E.psnr(o, gt), E.ssim(o, gt), E.lpips_d(o, gt), E.niqe(o)))
        m = np.array(rows).mean(0)
        res[name] = (m, np.array(rows), cost / len(pairs))
        b = res["identity"][0]
        pc = np.array(rows) - res["identity"][1]
        print(f"{name:<30} dPSNR {m[0]-b[0]:+.4f} dSSIM {m[1]-b[1]:+.5f} dLPIPS {m[2]-b[2]:+.5f} "
              f"dNIQE {m[3]-b[3]:+.4f} | {cost/len(pairs):6.0f}ms | abs {m[0]:.3f}/{m[1]:.4f}/{m[2]:.4f}/{m[3]:.3f}")
        if fn is not None:
            print("      逐图: " + "  ".join(f"{c} {pc[i,0]:+.3f}/{pc[i,1]:+.5f}" for i, c in enumerate(E.CASES)))
        sys.stdout.flush()
