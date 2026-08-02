"""候选算子注册表。每个算子 f(bgr_uint8, **params) -> bgr_uint8。"""
import numpy as np
import cv2

cv2.setNumThreads(2)  # 留给多进程


def _ksz(s):
    return int(2 * max(1, round(3 * s)) + 1)


def identity(x):
    return x


def gblur(x, s=0.5):
    return cv2.GaussianBlur(x, (_ksz(s), _ksz(s)), s)


def usm(x, s=1.0, amt=0.3):
    xf = x.astype(np.float32)
    bl = cv2.GaussianBlur(xf, (_ksz(s), _ksz(s)), s)
    return np.clip(xf + amt * (xf - bl), 0, 255).astype(np.uint8)


def usm_thr(x, s=1.0, amt=0.3, thr=3.0):
    """带阈值的 USM：只放大幅度超过 thr 的细节，避免放大噪声。"""
    xf = x.astype(np.float32)
    d = xf - cv2.GaussianBlur(xf, (_ksz(s), _ksz(s)), s)
    d = np.sign(d) * np.maximum(np.abs(d) - thr, 0)
    return np.clip(xf + amt * d, 0, 255).astype(np.uint8)


def usm_y(x, s=1.0, amt=0.3):
    """只在亮度通道锐化。"""
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    ycc[:, :, 0] = np.clip(y + amt * (y - cv2.GaussianBlur(y, (_ksz(s), _ksz(s)), s)), 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def wiener(x, s=1.0, nsr=0.01):
    """频域维纳反卷积，高斯核假设。"""
    xf = x.astype(np.float32) / 255.0
    H, W = xf.shape[:2]
    k1 = cv2.getGaussianKernel(_ksz(s), s)
    psf = np.zeros((H, W), np.float32)
    kk = (k1 @ k1.T).astype(np.float32)
    kh = kk.shape[0]
    psf[:kh, :kh] = kk
    psf = np.roll(psf, (-(kh // 2), -(kh // 2)), (0, 1))
    K = np.fft.rfft2(psf)
    G = np.conj(K) / (np.abs(K) ** 2 + nsr)
    out = np.stack([np.fft.irfft2(np.fft.rfft2(xf[:, :, c]) * G, s=(H, W)) for c in range(3)], -1)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def wiener_y(x, s=1.0, nsr=0.01):
    """只在 Y 通道做维纳反卷积。"""
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0] / 255.0
    H, W = y.shape
    k1 = cv2.getGaussianKernel(_ksz(s), s)
    kk = (k1 @ k1.T).astype(np.float32)
    psf = np.zeros((H, W), np.float32)
    kh = kk.shape[0]
    psf[:kh, :kh] = kk
    psf = np.roll(psf, (-(kh // 2), -(kh // 2)), (0, 1))
    K = np.fft.rfft2(psf)
    G = np.conj(K) / (np.abs(K) ** 2 + nsr)
    ycc[:, :, 0] = np.clip(np.fft.irfft2(np.fft.rfft2(y) * G, s=(H, W)) * 255, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def rl(x, s=1.0, it=3):
    """Richardson-Lucy，高斯 PSF，空域实现。"""
    obs = x.astype(np.float32) / 255.0 + 1e-6
    est = obs.copy()
    k = _ksz(s)
    for _ in range(it):
        conv = cv2.GaussianBlur(est, (k, k), s)
        est = est * cv2.GaussianBlur(obs / (conv + 1e-6), (k, k), s)
        np.clip(est, 0, 1, out=est)
    return np.clip(est * 255, 0, 255).astype(np.uint8)


def bilateral(x, d=5, sc=25, ss=5):
    return cv2.bilateralFilter(x, d, sc, ss)


def nlm(x, h=3, hc=3, tw=7, sw=13):
    return cv2.fastNlMeansDenoisingColored(x, None, h, hc, tw, sw)


def median(x, k=3):
    return cv2.medianBlur(x, k)


def guided(x, r=4, eps=100):
    """导向滤波（自导向），保边去噪。"""
    return cv2.ximgproc.guidedFilter(x, x, r, eps)


def denoise_then_usm(x, den="bilateral", dp=(5, 25, 5), s=1.0, amt=0.3):
    base = bilateral(x, *dp) if den == "bilateral" else nlm(x, *dp)
    return usm(base, s, amt)


def usm_on_denoised(x, ds=0.6, s=1.5, amt=0.4):
    """先高斯去噪一点点，再对去噪结果做 USM（细节从去噪图取，噪声不被放大）。"""
    xf = x.astype(np.float32)
    sm = cv2.GaussianBlur(xf, (_ksz(ds), _ksz(ds)), ds)
    det = sm - cv2.GaussianBlur(sm, (_ksz(s), _ksz(s)), s)
    return np.clip(xf + amt * det, 0, 255).astype(np.uint8)


def bilat_usm(x, d=5, sc=20, ss=5, s=1.5, amt=0.4):
    """双边去噪图上取细节，加回原图。"""
    xf = x.astype(np.float32)
    sm = cv2.bilateralFilter(x, d, sc, ss).astype(np.float32)
    det = sm - cv2.GaussianBlur(sm, (_ksz(s), _ksz(s)), s)
    return np.clip(xf + amt * det, 0, 255).astype(np.uint8)


REG = {k: v for k, v in list(globals().items()) if callable(v) and not k.startswith("_")
       and k not in ("np", "cv2")}


def apply(spec, img):
    name, params = spec
    return REG[name](img, **params)


def _sinc_lp(fc, ntap, beta=6.0):
    """窗函数法设计 1D 低通 FIR（截止 fc，单位 cyc/px，0~0.5）。"""
    n = np.arange(ntap) - (ntap - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.kaiser(ntap, beta)
    return (h / h.sum()).astype(np.float32)


def lowpass(x, fc=0.20, ntap=9, beta=6.0):
    """可分离窗函数低通：只砍掉 MTF≈0 的死频段，比高斯保留更多中频。"""
    h = _sinc_lp(fc, ntap, beta)
    return np.clip(cv2.sepFilter2D(x.astype(np.float32), -1, h, h,
                                   borderType=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8)


def lowpass_y(x, fc=0.20, ntap=9, beta=6.0):
    """只低通亮度通道。"""
    h = _sinc_lp(fc, ntap, beta)
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc[:, :, 0] = np.clip(cv2.sepFilter2D(ycc[:, :, 0], -1, h, h,
                                           borderType=cv2.BORDER_REFLECT), 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def lp_mix(x, fc=0.20, ntap=9, w=0.5):
    """低通与原图按 w 混合，等价于 g(f)=1-w+w*H(f) 的温和收缩。"""
    xf = x.astype(np.float32)
    h = _sinc_lp(fc, ntap)
    lp = cv2.sepFilter2D(xf, -1, h, h, borderType=cv2.BORDER_REFLECT)
    return np.clip(xf * (1 - w) + lp * w, 0, 255).astype(np.uint8)


def bilat_lp(x, d=5, sc=30, ss=5, fc=0.22, ntap=9, w=0.6):
    return lp_mix(cv2.bilateralFilter(x, d, sc, ss), fc, ntap, w)


REG.update({k: v for k, v in list(globals().items()) if callable(v) and not k.startswith("_")
            and k not in ("np", "cv2", "apply")})


def chroma_blur(x, s=1.5):
    """只模糊色度通道，亮度不动：NIQE(基于 Y)不受影响，色噪声被抑制。"""
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    k = _ksz(s)
    ycc[:, :, 1:] = cv2.GaussianBlur(ycc[:, :, 1:], (k, k), s)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def chroma_blur_luma_blur(x, sc=2.0, sy=0.6):
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1:] = cv2.GaussianBlur(ycc[:, :, 1:], (_ksz(sc),) * 2, sc)
    ycc[:, :, 0] = cv2.GaussianBlur(ycc[:, :, 0], (_ksz(sy),) * 2, sy)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def chroma_med_blur(x, mk=3, s=1.5):
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    c = cv2.medianBlur(ycc[:, :, 1:].copy(), mk)
    ycc[:, :, 1:] = cv2.GaussianBlur(c, (_ksz(s),) * 2, s)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def chroma_blur_luma_bilat(x, sc=2.0, d=5, scol=30, sspa=5):
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1:] = cv2.GaussianBlur(ycc[:, :, 1:], (_ksz(sc),) * 2, sc)
    ycc[:, :, 0] = cv2.bilateralFilter(ycc[:, :, 0].copy(), d, scol, sspa)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


REG.update({k: v for k, v in list(globals().items()) if callable(v) and not k.startswith("_")
            and k not in ("np", "cv2", "apply")})


def bilat_chroma(x, d=5, sc=30, ss=5, cs=4.0):
    """BGR 双边去噪 + 色度高斯模糊。"""
    y = cv2.cvtColor(cv2.bilateralFilter(x, d, sc, ss), cv2.COLOR_BGR2YCrCb)
    y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (_ksz(cs),) * 2, cs)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


def chroma_blur_luma_lp(x, cs=4.0, fc=0.30, ntap=7):
    y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (_ksz(cs),) * 2, cs)
    h = _sinc_lp(fc, ntap)
    y[:, :, 0] = cv2.sepFilter2D(y[:, :, 0], -1, h, h, borderType=cv2.BORDER_REFLECT)
    return cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


REG.update({k: v for k, v in list(globals().items()) if callable(v) and not k.startswith("_")
            and k not in ("np", "cv2", "apply")})


def chroma_med_blur_luma(x, mk=5, cs=2.0, sy=0.5):
    """色度中值+高斯去噪，亮度极温和高斯：本任务实测唯一 PSNR/SSIM 双正的组合。"""
    y = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    c = cv2.medianBlur(y[:, :, 1:].copy(), mk)
    y[:, :, 1:] = cv2.GaussianBlur(c, (_ksz(cs),) * 2, cs)
    if sy > 0:
        y[:, :, 0] = cv2.GaussianBlur(y[:, :, 0], (_ksz(sy),) * 2, sy)
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


REG.update({k: v for k, v in list(globals().items()) if callable(v) and not k.startswith("_")
            and k not in ("np", "cv2", "apply")})
