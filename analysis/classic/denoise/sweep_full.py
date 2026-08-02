"""全图 12MP 扫描，评判标准 = 三张图的 dPSNR/dSSIM 最差值都要 > 0（不接受靠平均掩盖的退化）。"""
import time, pickle
import numpy as np
import cv2
from multiprocessing import Pool
import harness as H

cv2.setNumThreads(2)


def epf(x, s, r):
    return cv2.edgePreservingFilter(x, flags=1, sigma_s=s, sigma_r=r)


def blend(x, y, a):
    return cv2.addWeighted(x, 1 - a, y, a, 0)


def guided(x, r, eps):
    I = x.astype(np.float32) / 255.
    out = np.empty_like(I); k = (2 * r + 1, 2 * r + 1)
    for c in range(3):
        p = I[..., c]
        mu = cv2.blur(p, k); var = cv2.blur(p * p, k) - mu * mu
        a = var / (var + eps); b = mu - a * mu
        out[..., c] = cv2.blur(a, k) * p + cv2.blur(b, k)
    return np.clip(out * 255., 0, 255).astype(np.uint8)


def make_ops():
    ops = [('identity', lambda x: x)]
    for k in [3, 5]:
        ops.append((f'median{k}', lambda x, k=k: cv2.medianBlur(x, k)))
    # 中值与原图混合（削弱强度）
    for a in [0.25, 0.5, 0.75]:
        ops.append((f'median3_a{a}', lambda x, a=a: blend(x, cv2.medianBlur(x, 3), a)))
    # 双边
    for d in [5, 9]:
        for sc in [5, 10, 20, 30]:
            ops.append((f'bilat_d{d}_sc{sc}', lambda x, d=d, sc=sc: cv2.bilateralFilter(x, d, sc, 10)))
    # 双边混合
    for a in [0.3, 0.5]:
        ops.append((f'bilat_d9_sc20_a{a}', lambda x, a=a: blend(x, cv2.bilateralFilter(x, 9, 20, 10), a)))
    # epf 弱化 + 混合
    for s, r in [(10, 0.4), (15, 0.25), (30, 0.15)]:
        for a in [0.25, 0.5, 1.0]:
            ops.append((f'epf_s{s}_r{r}_a{a}', lambda x, s=s, r=r, a=a: blend(x, epf(x, s, r), a)))
    # 高斯
    for sg in [0.3, 0.5, 0.8, 1.0]:
        ops.append((f'gauss{sg}', lambda x, sg=sg: cv2.GaussianBlur(x, (0, 0), sg)))
    # 引导
    for r in [2, 4, 8, 16]:
        for eps in [1e-4, 3e-4, 1e-3]:
            ops.append((f'guided_r{r}_e{eps}', lambda x, r=r, eps=eps: guided(x, r, eps)))
    # NLM 轻档
    for h in [1, 2, 3]:
        ops.append((f'nlm_h{h}', lambda x, h=h: cv2.fastNlMeansDenoisingColored(x, None, h, h, 7, 21)))
    # 只做色度去噪（保亮度，最安全）
    for sg in [1, 2, 4, 8]:
        def _cd(x, sg=sg):
            y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
            y[..., 1] = cv2.GaussianBlur(y[..., 1], (0, 0), sg)
            y[..., 2] = cv2.GaussianBlur(y[..., 2], (0, 0), sg)
            return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        ops.append((f'chromablur{sg}', _cd))
    for k in [3, 5]:
        def _cm(x, k=k):
            y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
            y[..., 1] = cv2.medianBlur(y[..., 1], k); y[..., 2] = cv2.medianBlur(y[..., 2], k)
            return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        ops.append((f'chromamed{k}', _cm))
    return ops


OPS = make_ops()
_PAIRS = None


def _init():
    global _PAIRS
    cv2.setNumThreads(2)
    _PAIRS = H.load_pairs(None)


def _run(idx):
    name, fn = OPS[idx]
    ps, ss, ts = [], [], []
    for lq, gt in _PAIRS:
        t0 = time.time()
        try:
            out = fn(lq)
        except Exception as e:
            return (name, None, None, None, f'ERR {e}')
        ts.append(time.time() - t0)
        ps.append(H.psnr(out, gt)); ss.append(H.ssim(out, gt))
    return (name, ps, ss, float(np.mean(ts)), '')


if __name__ == '__main__':
    print('n ops', len(OPS), flush=True)
    with Pool(6, initializer=_init) as p:
        res = p.map(_run, range(len(OPS)))
    base = [r for r in res if r[0] == 'identity'][0]
    bp, bs = np.array(base[1]), np.array(base[2])
    rows = []
    for name, ps, ss, dt, e in res:
        if e: print('ERR', name, e); continue
        dp, ds = np.array(ps) - bp, np.array(ss) - bs
        rows.append((name, dp.mean(), ds.mean(), dp.min(), ds.min(), dt, list(dp), list(ds)))
    pickle.dump(rows, open('sweep_full.pkl', 'wb'))
    safe = [r for r in rows if r[3] > 0 and r[4] > 0]
    safe.sort(key=lambda r: -r[3])
    print(f'\n===== 三图全部 dPSNR>0 且 dSSIM>0 的算子（按最差 dPSNR 排）=====')
    print(f'{"op":24s} {"meanDP":>8s} {"meanDS":>9s} {"minDP":>8s} {"minDS":>9s} {"s/12MP":>7s}')
    for r in safe:
        print(f'{r[0]:24s} {r[1]:+8.4f} {r[2]:+9.5f} {r[3]:+8.4f} {r[4]:+9.5f} {r[5]:7.2f}')
    print(f'\n===== 全部算子按 mean dPSNR 排 top25（含不安全项）=====')
    rows.sort(key=lambda r: -r[1])
    for r in rows[:25]:
        print(f'{r[0]:24s} {r[1]:+8.4f} {r[2]:+9.5f} {r[3]:+8.4f} {r[4]:+9.5f} {r[5]:7.2f}')
