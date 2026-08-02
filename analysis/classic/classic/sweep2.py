"""精扫安全候选：色度专用、小 eps 导向、平坦区去噪。耗时单独计。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/csig_bench/classicE")
import evallib as E
import ops as O

_BASE = {}


def bench(cands, crops):
    key = id(crops)
    if key not in _BASE:
        _BASE[key] = [(E.psnr(l, g), E.ssim(l, g)) for _, l, g in crops]
    base = _BASE[key]
    bp = np.array([x[0] for x in base])
    bs = np.array([x[1] for x in base])
    out = []
    for name, fn in cands:
        outs = []
        t = time.perf_counter()
        for _, l, g in crops:
            outs.append(fn(l))
        px = sum(l.shape[0] * l.shape[1] for _, l, _ in crops)
        cost = (time.perf_counter() - t) / px * 12e6 * 1000
        ps = np.array([E.psnr(o, g) for o, (_, _, g) in zip(outs, crops)])
        ss = np.array([E.ssim(o, g) for o, (_, _, g) in zip(outs, crops)])
        dp, ds = ps - bp, ss - bs
        out.append((name, dp.mean(), ds.mean(), dp.min(), ds.min(), cost))
        print(f"{name:<34} dPSNR {dp.mean():+.4f} dSSIM {ds.mean():+.5f} | worst {dp.min():+.4f}/{ds.min():+.5f} | {cost:6.0f}ms")
    return out


if __name__ == "__main__":
    crops = E.crop_set(1024, 3)
    w = sys.argv[1] if len(sys.argv) > 1 else "all"

    if w in ("all", "chroma"):
        print("=== 色度联合导向（Y 引导）===")
        bench([(f"cguided r={r} eps={e}", O.chroma_guided(r, e)) for r in (2, 4, 8) for e in (1, 4, 16, 64)], crops)
        print("=== 色度模糊 ===")
        bench([(f"cblur s={s}", O.chroma_blur(s)) for s in (0.5, 0.8, 1.2, 2.0)], crops)

    if w in ("all", "fine"):
        print("=== 小 eps 导向精扫 ===")
        bench([(f"guided r={r} eps={e}", O.guided(r, e)) for r in (1, 2, 3, 4) for e in (2, 4, 6, 8, 12)], crops)
        print("=== 亮度专用导向 ===")
        bench([(f"lguided r={r} eps={e}", O.luma_guided(r, e)) for r in (2, 4) for e in (2, 4, 8, 16)], crops)

    if w in ("all", "dflat"):
        print("=== 平坦区去噪精扫 ===")
        bench([(f"dflat r={r} eps={e} se={se}", O.denoise_flat(r, e, se))
               for r in (2, 4, 8) for e in (16, 32, 64) for se in (3, 5, 8)], crops)

    if w in ("all", "combo"):
        print("=== 组合：色度导向 + 亮度小 eps 导向 ===")
        c = []
        for ce in (4, 16, 64):
            for le in (2, 4, 8):
                c.append((f"cg(2,{ce})+lg(2,{le})", O.chain(O.chroma_guided(2, ce), O.luma_guided(2, le))))
        bench(c, crops)
