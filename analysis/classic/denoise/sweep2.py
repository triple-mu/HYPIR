"""精细扫描：围绕 epf 冠军 + 小波 + 组合算子。"""
import time, pickle
import numpy as np
import cv2
from multiprocessing import Pool
import harness as H

cv2.setNumThreads(1)


def _wav(x, s, mode, wav='db1', levels=None):
    from skimage.restoration import denoise_wavelet
    o = denoise_wavelet(x.astype(np.float32) / 255., sigma=s / 255., mode='soft',
                        method=mode, channel_axis=-1, convert2ycbcr=True,
                        wavelet=wav, wavelet_levels=levels, rescale_sigma=True)
    return np.clip(o * 255., 0, 255).astype(np.uint8)


def _guided(x, r, eps):
    I = x.astype(np.float32) / 255.
    out = np.empty_like(I)
    k = (2 * r + 1, 2 * r + 1)
    for c in range(3):
        p = I[..., c]
        mu = cv2.blur(p, k); var = cv2.blur(p * p, k) - mu * mu
        a = var / (var + eps); b = mu - a * mu
        out[..., c] = cv2.blur(a, k) * p + cv2.blur(b, k)
    return np.clip(out * 255., 0, 255).astype(np.uint8)


def make_ops():
    ops = [('identity', lambda x: x)]
    # epf 精细网格（flags=1 RECURS_FILTER, 2 NORMCONV_FILTER）
    for fl in [1, 2]:
        for s in [3, 5, 8, 10, 15, 20, 30]:
            for r in [0.15, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
                ops.append((f'epf_f{fl}_s{s}_r{r}',
                            lambda x, fl=fl, s=s, r=r: cv2.edgePreservingFilter(x, flags=fl, sigma_s=s, sigma_r=r)))
    # 小波
    for wv in ['db1', 'db2', 'sym4', 'coif1']:
        for s in [0.5, 1, 1.5, 2, 3, 5, 8]:
            for md in ['BayesShrink', 'VisuShrink']:
                ops.append((f'wav_{wv}_{md}_s{s}', lambda x, s=s, md=md, wv=wv: _wav(x, s, md, wv)))
    # 引导滤波大半径
    for r in [8, 16, 32]:
        for eps in [3e-4, 1e-3, 3e-3, 1e-2]:
            ops.append((f'guided_r{r}_e{eps}', lambda x, r=r, eps=eps: _guided(x, r, eps)))
    # 组合：epf 后接轻微高斯
    for r in [0.3, 0.4]:
        for sg in [0.4, 0.6, 0.8]:
            ops.append((f'epf_s10_r{r}+g{sg}',
                        lambda x, r=r, sg=sg: cv2.GaussianBlur(
                            cv2.edgePreservingFilter(x, flags=1, sigma_s=10, sigma_r=r), (0, 0), sg)))
    # 组合：中值 + epf
    for r in [0.3, 0.4]:
        ops.append((f'med3+epf_r{r}',
                    lambda x, r=r: cv2.edgePreservingFilter(cv2.medianBlur(x, 3), flags=1, sigma_s=10, sigma_r=r)))
    # 只在色度上去噪（保亮度）
    for sg in [1, 2, 4]:
        def _cd(x, sg=sg):
            y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
            y[..., 1] = cv2.GaussianBlur(y[..., 1], (0, 0), sg)
            y[..., 2] = cv2.GaussianBlur(y[..., 2], (0, 0), sg)
            return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        ops.append((f'chromablur{sg}', _cd))
    # epf 只作用于亮度
    for r in [0.3, 0.4]:
        def _yepf(x, r=r):
            y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
            f = cv2.edgePreservingFilter(cv2.cvtColor(y[..., 0], cv2.COLOR_GRAY2BGR),
                                         flags=1, sigma_s=10, sigma_r=r)
            y[..., 0] = f[..., 0]
            return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        ops.append((f'Yonly_epf_r{r}', _yepf))
    return ops


OPS = make_ops()
_PAIRS = None


def _init():
    global _PAIRS
    cv2.setNumThreads(1)
    _PAIRS = H.load_pairs(1024, 3)


def _run(idx):
    name, fn = OPS[idx]
    t0 = time.time()
    ps, ss = [], []
    for lq, gt in _PAIRS:
        try:
            out = fn(lq)
        except Exception as e:
            return (name, None, None, None, f'ERR {type(e).__name__} {e}')
        ps.append(H.psnr(out, gt)); ss.append(H.ssim(out, gt))
    return (name, float(np.mean(ps)), float(np.mean(ss)), (time.time() - t0) / len(_PAIRS), '')


if __name__ == '__main__':
    print('n ops', len(OPS), flush=True)
    with Pool(22, initializer=_init) as p:
        res = p.map(_run, range(len(OPS)))
    base = [r for r in res if r[0] == 'identity'][0]
    rows = [(n, ps - base[1], ss - base[2], ps, ss, dt) for n, ps, ss, dt, e in res if not e]
    errs = set(e.split(' ')[1] for n, ps, ss, dt, e in res if e)
    if errs: print('ERR types:', errs)
    rows.sort(key=lambda r: -r[1])
    print(f'\nbaseline PSNR={base[1]:.4f} SSIM={base[2]:.5f}\n')
    print(f'{"op":30s} {"dPSNR":>8s} {"dSSIM":>9s} {"s/1MP":>7s}')
    for r in rows[:45]:
        print(f'{r[0]:30s} {r[1]:+8.4f} {r[2]:+9.5f} {r[5]:7.2f}')
    pickle.dump(rows, open('sweep_A2.pkl', 'wb'))
