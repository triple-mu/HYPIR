"""参数搜索通用框架：对 15 个 crop 跑算子，输出相对 LQ 的指标增量。"""
import numpy as np
import metrics
import functools

CROPS = "/home/ubuntu/workspace/contest/CSIG-2026/deblur_search/crops/"
CASES = [f"{c}_{i}" for c in ("case1", "case2", "case3") for i in range(5)]

_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = (np.load(f"{CROPS}{name}_lq.npy"), np.load(f"{CROPS}{name}_gt.npy"))
    return _cache[name]


@functools.lru_cache(maxsize=1)
def baseline(full=True):
    out = {}
    for n in CASES:
        lq, gt = load(n)
        out[n] = dict(psnr=metrics.psnr(lq, gt), ssim=metrics.ssim(lq, gt),
                      lpips=metrics.lpips_score(lq, gt) if full else 0.0,
                      niqe=metrics.niqe(lq) if full else 0.0)
    return out


def evaluate(fn, cases=None, full=False, verbose=False):
    """返回 (mean_dpsnr, mean_dssim, mean_dlpips, mean_dniqe, per_case)"""
    base = baseline()
    cases = cases or CASES
    rows = {}
    for n in cases:
        lq, gt = load(n)
        out = fn(lq)
        assert out.dtype == np.uint8 and out.shape == lq.shape, (out.dtype, out.shape)
        r = dict(psnr=metrics.psnr(out, gt) - base[n]["psnr"],
                 ssim=metrics.ssim(out, gt) - base[n]["ssim"])
        if full:
            r["lpips"] = metrics.lpips_score(out, gt) - base[n]["lpips"]
            r["niqe"] = metrics.niqe(out) - base[n]["niqe"]
        rows[n] = r
        if verbose:
            print("  ", n, {k: round(v, 4) for k, v in r.items()})
    agg = {k: float(np.mean([rows[n][k] for n in cases])) for k in rows[cases[0]]}
    return agg, rows
