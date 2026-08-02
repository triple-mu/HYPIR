"""入围算子的全图 12MP 四指标评测（PSNR/SSIM/LPIPS/NIQE），全部相对未处理 LQ 报增量。"""
import time
import numpy as np
import cv2
import harness as H

cv2.setNumThreads(0)


def epf(x, s, r):
    return cv2.edgePreservingFilter(x, flags=1, sigma_s=s, sigma_r=r)


CANDS = [
    ('identity', lambda x: x),
    ('epf_s10_r0.4_g0.6', lambda x: cv2.GaussianBlur(epf(x, 10, 0.4), (0, 0), 0.6)),
    ('epf_s15_r0.25', lambda x: epf(x, 15, 0.25)),
    ('epf_s30_r0.15', lambda x: epf(x, 30, 0.15)),
    ('epf_s10_r0.4', lambda x: epf(x, 10, 0.4)),
    ('gauss1.0', lambda x: cv2.GaussianBlur(x, (0, 0), 1.0)),
    ('bilat_d9_sc20_ss10', lambda x: cv2.bilateralFilter(x, 9, 20, 10)),
    ('median3', lambda x: cv2.medianBlur(x, 3)),
]


def lpips_tiled(a, b, tile=1024):
    """12MP 图 LPIPS：切 1024 瓦片求均值（相对比较用，口径一致即可）。"""
    h, w = a.shape[:2]
    vs = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            vs.append(H.lpips_dist(a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]))
    return float(np.mean(vs))


def niqe_tiled(a, tile=1024):
    h, w = a.shape[:2]
    vs = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            vs.append(H.niqe(a[y:y + tile, x:x + tile]))
    return float(np.mean(vs))


if __name__ == '__main__':
    pairs = H.load_pairs(None)
    res = {}
    for name, fn in CANDS:
        P, S, L, N, T = [], [], [], [], []
        for lq, gt in pairs:
            t0 = time.time(); out = fn(lq); T.append(time.time() - t0)
            P.append(H.psnr(out, gt)); S.append(H.ssim(out, gt))
            L.append(lpips_tiled(out, gt)); N.append(niqe_tiled(out))
        res[name] = (np.mean(P), np.mean(S), np.mean(L), np.mean(N), np.mean(T), P, S, L, N)
        print(f'{name:20s} PSNR {np.mean(P):.4f} SSIM {np.mean(S):.5f} '
              f'LPIPS {np.mean(L):.5f} NIQE {np.mean(N):.4f} {np.mean(T)*1000:.0f}ms', flush=True)

    b = res['identity']
    print('\n===== 相对 LQ 的增量（全图 12MP, 3 张验证图平均）=====')
    print(f'{"op":20s} {"dPSNR":>8s} {"dSSIM":>9s} {"dLPIPS":>9s} {"dNIQE":>8s} {"ms/12MP":>8s}')
    for name, _ in CANDS:
        r = res[name]
        print(f'{name:20s} {r[0]-b[0]:+8.4f} {r[1]-b[1]:+9.5f} {r[2]-b[2]:+9.5f} '
              f'{r[3]-b[3]:+8.4f} {r[4]*1000:8.0f}')
    print('\n===== 逐图 dPSNR / dSSIM =====')
    for name, _ in CANDS:
        r = res[name]
        print(name, 'dPSNR', [round(x - y, 4) for x, y in zip(r[5], b[5])],
              'dSSIM', [round(x - y, 5) for x, y in zip(r[6], b[6])])
