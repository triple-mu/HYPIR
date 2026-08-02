"""在 crop 上扫参数，只报 PSNR/SSIM（快），筛出双不降的候选。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E
import ops as O


def run(cands, crops, base=None, tag=""):
    if base is None:
        base = [(E.psnr(l, g), E.ssim(l, g)) for _, l, g in crops]
    bp, bs = np.mean([x[0] for x in base]), np.mean([x[1] for x in base])
    res = []
    for name, fn in cands:
        t = time.perf_counter()
        ps, ss = [], []
        for _, l, g in crops:
            o = fn(l)
            ps.append(E.psnr(o, g))
            ss.append(E.ssim(o, g))
        dt = (time.perf_counter() - t) / len(crops) / (1024 * 1024) * 12e6 * 1000
        dp, ds = np.mean(ps) - bp, np.mean(ss) - bs
        # 逐 crop 最差值
        wp = min(np.array(ps) - np.array([x[0] for x in base]))
        ws = min(np.array(ss) - np.array([x[1] for x in base]))
        res.append((name, dp, ds, wp, ws, dt))
        print(f"{tag}{name:<38} dPSNR {dp:+.4f}  dSSIM {ds:+.5f}   worst {wp:+.3f}/{ws:+.5f}  {dt:7.0f}ms")
    return res, base


if __name__ == "__main__":
    crops = E.crop_set(1024, 3)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = None

    if which in ("all", "usm"):
        print("=== USM 幅度扫描（含负值=模糊）===")
        c = [(f"usm a={a} r={r}", O.usm(a, r)) for r in (0.8, 1.5, 3.0) for a in (-0.3, -0.15, 0.1, 0.2, 0.35, 0.5)]
        _, base = run(c, crops, base)

    if which in ("all", "blur"):
        print("=== 纯高斯模糊 ===")
        c = [(f"gblur s={s}", O.gblur(s)) for s in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)]
        _, base = run(c, crops, base)

    if which in ("all", "guided"):
        print("=== 导向滤波去噪 ===")
        c = [(f"guided r={r} eps={e}", O.guided(r, e)) for r in (2, 4, 8) for e in (1, 4, 16, 64, 256)]
        _, base = run(c, crops, base)
        print("=== 导向滤波细节增益 ===")
        c = [(f"gsharp r={r} eps={e} b={b}", O.guided_sharp(r, e, b)) for r in (4, 8) for e in (16, 64, 256)
             for b in (0.6, 0.8, 1.2, 1.5)]
        _, base = run(c, crops, base)

    if which in ("all", "bilat"):
        print("=== 双边 ===")
        c = [(f"bilat d=5 sc={sc} ss={ss}", O.bilat(5, sc, ss)) for sc in (5, 10, 20, 40) for ss in (3, 6)]
        _, base = run(c, crops, base)

    if which in ("all", "misc"):
        print("=== CLAHE / 增益偏置 / 中值 ===")
        c = [(f"clahe c={cl} g={g}", O.clahe(cl, g)) for cl in (1.0, 2.0) for g in (8, 16)]
        c += [(f"gain g={g} b={b}", O.gain_bias(g, b)) for g, b in
              ((1.0, 2.0), (1.0, -2.0), (0.97, 4.0), (1.03, -4.0), (1.05, -6.0))]
        c += [("median 3", O.median(3))]
        _, base = run(c, crops, base)

    if which in ("all", "ms"):
        print("=== 多尺度（中频/高频分别增益）===")
        c = [(f"msharp mid={a1} hi={a2}", O.msharp(a1, a2)) for a1 in (0.0, 0.2, 0.4, 0.6) for a2 in (-0.3, -0.15, 0.0, 0.2)]
        _, base = run(c, crops, base)

    if which in ("all", "adapt"):
        print("=== 自适应锐化 / 平坦区去噪 ===")
        c = [(f"ausm a={a} lo={lo} hi={hi}", O.adaptive_usm(a, 1.5, lo, hi))
             for a in (0.3, 0.5) for lo, hi in ((2, 8), (4, 12), (6, 20))]
        c += [(f"dflat r={r} eps={e} se={se}", O.denoise_flat(r, e, se)) for r in (4,) for e in (16, 64) for se in (4, 8)]
        _, base = run(c, crops, base)

    if which in ("all", "wav"):
        print("=== 小波 / TV ===")
        c = [(f"wavelet s={s}", O.wavelet(s)) for s in (1, 2, 4)]
        c += [(f"tv w={w}", O.tv(w)) for w in (0.005, 0.01, 0.02)]
        _, base = run(c, crops, base)
