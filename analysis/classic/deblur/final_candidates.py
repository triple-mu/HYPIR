"""最终候选：签名统一为 f(bgr_uint8) -> bgr_uint8，参数全局固定。"""
import cv2
import numpy as np


def cand_chroma_luma(img):
    """色度中值(5)+高斯(σ2.0) 去噪，亮度高斯 σ0.5。全验证集 PSNR/SSIM 逐图均不下降。"""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    c = cv2.medianBlur(y[:, :, 1:].copy(), 5)
    y[:, :, 1:] = cv2.GaussianBlur(c, (13, 13), 2.0)
    y[:, :, 0] = cv2.GaussianBlur(y[:, :, 0], (5, 5), 0.5)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


def cand_chroma_only(img):
    """只动色度：中值(5)+高斯(σ2.0)。亮度完全不变，NIQE/LPIPS 几乎零代价。"""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    c = cv2.medianBlur(y[:, :, 1:].copy(), 5)
    y[:, :, 1:] = cv2.GaussianBlur(c, (13, 13), 2.0)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


def cand_gblur(img):
    """全通道高斯 σ0.6，最便宜的正收益。"""
    return cv2.GaussianBlur(img, (5, 5), 0.6)


def neg_usm(img):
    """反例：温和 USM(σ2, amount 0.35)。PSNR/SSIM 双降。"""
    x = img.astype(np.float32)
    return np.clip(x + 0.35 * (x - cv2.GaussianBlur(x, (13, 13), 2.0)), 0, 255).astype(np.uint8)


def neg_wiener(img):
    """反例：高斯核维纳反卷积 σ1.0 NSR 0.01。"""
    x = img.astype(np.float32) / 255.0
    H, W = x.shape[:2]
    k1 = cv2.getGaussianKernel(7, 1.0)
    kk = (k1 @ k1.T).astype(np.float32)
    psf = np.zeros((H, W), np.float32)
    psf[:7, :7] = kk
    psf = np.roll(psf, (-3, -3), (0, 1))
    K = np.fft.rfft2(psf)
    G = np.conj(K) / (np.abs(K) ** 2 + 0.01)
    out = np.stack([np.fft.irfft2(np.fft.rfft2(x[:, :, c]) * G, s=(H, W)) for c in range(3)], -1)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def neg_rl(img):
    """反例：Richardson-Lucy 高斯 PSF σ1.0，3 次迭代。"""
    obs = img.astype(np.float32) / 255.0 + 1e-6
    est = obs.copy()
    for _ in range(3):
        conv = cv2.GaussianBlur(est, (7, 7), 1.0)
        est = est * cv2.GaussianBlur(obs / (conv + 1e-6), (7, 7), 1.0)
        np.clip(est, 0, 1, out=est)
    return np.clip(est * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/deblur_search")
    import metrics
    import fullval
    cv2.setNumThreads(0)
    D = [(fullval.load(l), fullval.load(g)) for l, g in fullval.PAIRS]
    B = [(metrics.psnr(l, g), metrics.ssim(l, g), fullval.lpips_tiled(l, g), metrics.niqe(l))
         for l, g in D]
    for fn in [cand_chroma_luma, cand_chroma_only, cand_gblur, neg_usm, neg_wiener, neg_rl]:
        outs, cost = [], 0.0
        for l, _ in D:
            t = time.perf_counter()
            outs.append(fn(l))
            cost += (time.perf_counter() - t) * 1000 / 3
        d = [(metrics.psnr(o, g) - b[0], metrics.ssim(o, g) - b[1],
              fullval.lpips_tiled(o, g) - b[2], metrics.niqe(o) - b[3])
             for (l, g), o, b in zip(D, outs, B)]
        m = np.mean(d, 0)
        print(f"{fn.__name__:<18s} dPSNR={m[0]:+.4f} dSSIM={m[1]:+.5f} dLPIPS={m[2]:+.4f} "
              f"dNIQE={m[3]:+.3f} cost={cost:.0f}ms | perimg P={[round(x[0], 4) for x in d]} "
              f"S={[round(x[1], 5) for x in d]}")
