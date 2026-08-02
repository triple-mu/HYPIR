"""逐图逐通道 LQ->GT 传递曲线 + 黑位/白位压死统计。"""
import numpy as np
from PIL import Image
import json, sys

Image.MAX_IMAGE_PIXELS = None
ROOT = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
PAIRS = [("case1_lq.jpg", "case1_gt.png"),
         ("case2_lq.jpg", "case2_gt.png"),
         ("case3_lq.jpg", "case3_gt.jpg")]

out = {}
for lqf, gtf in PAIRS:
    name = lqf.split("_")[0]
    lq = np.asarray(Image.open(f"{ROOT}/{lqf}").convert("RGB"))
    gt = np.asarray(Image.open(f"{ROOT}/{gtf}").convert("RGB"))
    assert lq.shape == gt.shape, (lq.shape, gt.shape)
    H, W, _ = lq.shape
    N = H * W
    rec = {"shape": [H, W]}
    for ci, cn in enumerate("RGB"):
        l = lq[:, :, ci].ravel()
        g = gt[:, :, ci].ravel()
        # 传递曲线：按 LQ 值分桶，统计 GT 的 count/mean/median/std/p10/p90
        order = np.argsort(l, kind="stable")
        ls, gs = l[order], g[order]
        bounds = np.searchsorted(ls, np.arange(257))
        curve = []
        for v in range(256):
            a, b = bounds[v], bounds[v + 1]
            if b - a == 0:
                curve.append([v, 0, None, None, None, None, None])
                continue
            seg = gs[a:b]
            curve.append([v, int(b - a), float(seg.mean()),
                          float(np.median(seg)), float(seg.std()),
                          float(np.percentile(seg, 10)), float(np.percentile(seg, 90))])
        # 反向：GT 值 -> LQ 均值（检查是否单调）
        rec[cn] = {
            "curve": curve,
            "lq_zero_frac": float((l == 0).mean()),
            "lq_255_frac": float((l == 255).mean()),
            "gt_zero_frac": float((g == 0).mean()),
            "gt_255_frac": float((g == 255).mean()),
            "lq_mean": float(l.mean()), "gt_mean": float(g.mean()),
            "lq_std": float(l.std()), "gt_std": float(g.std()),
            "lq_p01": float(np.percentile(l, 1)), "gt_p01": float(np.percentile(g, 1)),
            "lq_p99": float(np.percentile(l, 99)), "gt_p99": float(np.percentile(g, 99)),
            # GT 在 LQ==0 处的分布
            "gt_at_lq0_mean": float(g[l == 0].mean()) if (l == 0).any() else None,
            "gt_at_lq0_med": float(np.median(g[l == 0])) if (l == 0).any() else None,
            "gt_at_lq0_p90": float(np.percentile(g[l == 0], 90)) if (l == 0).any() else None,
            "gt_at_lq0_max": int(g[l == 0].max()) if (l == 0).any() else None,
            "gt_at_lq255_mean": float(g[l == 255].mean()) if (l == 255).any() else None,
            "gt_at_lq255_min": int(g[l == 255].min()) if (l == 255).any() else None,
        }
        # 最优仿射（复核前人）
        A = np.vstack([l.astype(np.float64), np.ones(N)]).T
        # 用 normal equation 省内存
        sl = l.astype(np.float64).sum(); sll = (l.astype(np.float64) ** 2).sum()
        sg = g.astype(np.float64).sum(); slg = (l.astype(np.float64) * g).sum()
        det = N * sll - sl * sl
        a = (N * slg - sl * sg) / det
        b = (sll * sg - sl * slg) / det
        rec[cn]["affine"] = [float(a), float(b)]
    # 全像素 RGB 三通道联合的黑位聚集性
    lq_allzero = (lq == 0).all(axis=2)
    rec["lq_rgb_allzero_frac"] = float(lq_allzero.mean())
    rec["lq_any_zero_frac"] = float((lq == 0).any(axis=2).mean())
    gtv = gt[lq_allzero]
    if gtv.size:
        rec["gt_at_lq_rgb0"] = {"mean": gtv.mean(axis=0).tolist(),
                                "med": np.median(gtv, axis=0).tolist(),
                                "p90": np.percentile(gtv, 90, axis=0).tolist(),
                                "max": gtv.max(axis=0).tolist(),
                                "frac_gt_gte8": float((gtv.max(axis=1) >= 8).mean())}
        # 空间聚集性：黑像素的连通性 —— 用 4-邻域中也是黑的比例衡量
        m = lq_allzero
        nb = np.zeros_like(m, dtype=np.int32)
        nb[1:, :] += m[:-1, :]; nb[:-1, :] += m[1:, :]
        nb[:, 1:] += m[:, :-1]; nb[:, :-1] += m[:, 1:]
        rec["black_nbr_mean"] = float(nb[m].mean())  # 随机散布应≈4*frac
        rec["black_nbr_expect_random"] = float(4 * m.mean())
    out[name] = rec
    print(name, "done", flush=True)

json.dump(out, open("/home/ubuntu/workspace/contest/CSIG-2026/Moebius/analysis_tone/t1.json", "w"))
