"""算子族 A（去噪）参数扫描：只算 PSNR/SSIM（快），后续再对入围者补 LPIPS/NIQE。"""
import sys, time, itertools, pickle
import numpy as np
import cv2
from multiprocessing import Pool
import harness as H

cv2.setNumThreads(1)


# ---------- 算子定义：每个返回 (name, fn) ----------
def make_ops():
    ops = [('identity', lambda x: x)]

    # 1. NLM 彩色
    for h in [1, 2, 3, 5, 7, 10]:
        for hc in [1, 3, 10]:
            ops.append((f'nlm_h{h}_hc{hc}',
                        lambda x, h=h, hc=hc: cv2.fastNlMeansDenoisingColored(x, None, h, hc, 7, 21)))

    # 2. 双边
    for d in [5, 9]:
        for sc in [5, 10, 20, 50]:
            for ss in [5, 10, 20]:
                ops.append((f'bilat_d{d}_sc{sc}_ss{ss}',
                            lambda x, d=d, sc=sc, ss=ss: cv2.bilateralFilter(x, d, sc, ss)))

    # 3. 中值
    for k in [3, 5]:
        ops.append((f'median_{k}', lambda x, k=k: cv2.medianBlur(x, k)))

    # 4. TV-Chambolle
    for w in [0.002, 0.005, 0.01, 0.02, 0.05]:
        def _tv(x, w=w):
            from skimage.restoration import denoise_tv_chambolle
            o = denoise_tv_chambolle(x.astype(np.float32) / 255., weight=w, channel_axis=-1)
            return np.clip(o * 255., 0, 255).astype(np.uint8)
        ops.append((f'tv_w{w}', _tv))

    # 5. 小波
    for s in [0.5, 1, 2, 3, 5]:
        for mode in ['BayesShrink', 'VisuShrink']:
            def _wv(x, s=s, mode=mode):
                from skimage.restoration import denoise_wavelet
                o = denoise_wavelet(x.astype(np.float32) / 255., sigma=s / 255., mode='soft',
                                    method=mode, channel_axis=-1, convert2ycbcr=True,
                                    rescale_sigma=True)
                return np.clip(o * 255., 0, 255).astype(np.uint8)
            ops.append((f'wav_{mode}_s{s}', _wv))

    # 6. Perona-Malik 各向异性扩散
    def _pm(x, niter, kappa, gamma=0.15):
        img = x.astype(np.float32)
        out = img.copy()
        for _ in range(niter):
            dn = np.roll(out, -1, 0) - out
            ds = np.roll(out, 1, 0) - out
            de = np.roll(out, -1, 1) - out
            dw = np.roll(out, 1, 1) - out
            g = lambda d: np.exp(-(d / kappa) ** 2)
            out = out + gamma * (g(dn) * dn + g(ds) * ds + g(de) * de + g(dw) * dw)
        return np.clip(out, 0, 255).astype(np.uint8)
    for ni in [2, 5, 10]:
        for kp in [5, 10, 20, 50]:
            ops.append((f'pm_n{ni}_k{kp}', lambda x, ni=ni, kp=kp: _pm(x, ni, kp)))

    # 7. 引导滤波（自引导），r/eps
    def _guided(x, r, eps):
        I = x.astype(np.float32) / 255.
        out = np.empty_like(I)
        k = (2 * r + 1, 2 * r + 1)
        for c in range(3):
            p = I[..., c]
            mu = cv2.blur(p, k)
            var = cv2.blur(p * p, k) - mu * mu
            a = var / (var + eps)
            b = mu - a * mu
            out[..., c] = cv2.blur(a, k) * p + cv2.blur(b, k)
        return np.clip(out * 255., 0, 255).astype(np.uint8)
    for r in [2, 4, 8]:
        for eps in [1e-5, 1e-4, 1e-3]:
            ops.append((f'guided_r{r}_e{eps}', lambda x, r=r, eps=eps: _guided(x, r, eps)))

    # 8. 边缘保持滤波
    for s in [10, 30, 60]:
        for r in [0.1, 0.2, 0.4]:
            ops.append((f'epf_s{s}_r{r}',
                        lambda x, s=s, r=r: cv2.edgePreservingFilter(x, flags=1, sigma_s=s, sigma_r=r)))

    # 9. 对照组：高斯模糊（预期掉 PSNR）
    for sg in [0.3, 0.5, 0.8]:
        ops.append((f'CTRL_gauss{sg}', lambda x, sg=sg: cv2.GaussianBlur(x, (0, 0), sg)))

    return ops


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
            return (name, None, None, None, f'ERR {e}')
        ps.append(H.psnr(out, gt)); ss.append(H.ssim(out, gt))
    dt = (time.time() - t0) / len(_PAIRS)
    return (name, float(np.mean(ps)), float(np.mean(ss)), dt, '')


OPS = make_ops()

if __name__ == '__main__':
    print('n ops', len(OPS), flush=True)
    with Pool(22, initializer=_init) as p:
        res = p.map(_run, range(len(OPS)))
    base = [r for r in res if r[0] == 'identity'][0]
    rows = []
    for name, ps, ss, dt, err in res:
        if err:
            print('ERR', name, err); continue
        rows.append((name, ps - base[1], ss - base[2], ps, ss, dt))
    rows.sort(key=lambda r: -r[1])
    print(f'\nbaseline(identity) PSNR={base[1]:.4f} SSIM={base[2]:.5f}\n')
    print(f'{"op":34s} {"dPSNR":>8s} {"dSSIM":>9s} {"PSNR":>8s} {"SSIM":>8s} {"s/1MP":>7s}')
    for r in rows:
        print(f'{r[0]:34s} {r[1]:+8.4f} {r[2]:+9.5f} {r[3]:8.4f} {r[4]:8.5f} {r[5]:7.2f}')
    pickle.dump(rows, open('sweep_A.pkl', 'wb'))
