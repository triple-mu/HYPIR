"""多进程参数搜索：对 (算子, 参数) 组合评估 dPSNR/dSSIM（可选 LPIPS/NIQE）。"""
import numpy as np
import os
import sys
import json
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CROPS = HERE + "/crops/"
CASES = [f"{c}_{i}" for c in ("case1", "case2", "case3") for i in range(5)]
BASE_JSON = HERE + "/baseline.json"

_data = {}


def _load(n):
    if n not in _data:
        _data[n] = (np.load(f"{CROPS}{n}_lq.npy"), np.load(f"{CROPS}{n}_gt.npy"))
    return _data[n]


def _work(arg):
    spec, n, full = arg
    import ops
    import metrics
    lq, gt = _load(n)
    out = ops.apply(spec, lq)
    r = [metrics.psnr(out, gt), metrics.ssim(out, gt)]
    if full:
        r += [metrics.lpips_score(out, gt), metrics.niqe(out)]
    return r


def baseline(full=True):
    if os.path.exists(BASE_JSON):
        return json.load(open(BASE_JSON))
    with mp.Pool(8) as p:
        res = p.map(_work, [(("identity", {}), n, full) for n in CASES])
    b = {n: dict(zip(["psnr", "ssim", "lpips", "niqe"], r)) for n, r in zip(CASES, res)}
    json.dump(b, open(BASE_JSON, "w"))
    return b


def run(specs, full=False, nproc=12, cases=None):
    """specs: [(name, params), ...] -> [(spec, agg_dict, per_case), ...]"""
    cases = cases or CASES
    b = baseline()
    jobs = [(s, n, full) for s in specs for n in cases]
    with mp.Pool(nproc) as p:
        res = p.map(_work, jobs, chunksize=1)
    keys = ["psnr", "ssim"] + (["lpips", "niqe"] if full else [])
    out = []
    nc = len(cases)
    for i, s in enumerate(specs):
        rows = {}
        for j, n in enumerate(cases):
            v = res[i * nc + j]
            rows[n] = {k: v[m] - b[n][k] for m, k in enumerate(keys)}
        agg = {k: float(np.mean([rows[n][k] for n in cases])) for k in keys}
        out.append((s, agg, rows))
    return out


def show(results, sort_by="psnr", full=False, top=None):
    results = sorted(results, key=lambda r: -r[1][sort_by] if sort_by != "lpips" else r[1][sort_by])
    if top:
        results = results[:top]
    for s, a, _ in results:
        ps = ",".join(f"{k}={v}" for k, v in s[1].items())
        line = f"{s[0]:<18s} {ps:<44s} dPSNR={a['psnr']:+.4f} dSSIM={a['ssim']:+.5f}"
        if full:
            line += f" dLPIPS={a['lpips']:+.4f} dNIQE={a['niqe']:+.3f}"
        print(line)
    return results
