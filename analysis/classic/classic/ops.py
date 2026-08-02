"""经典算子候选库。全部 f(bgr_uint8)->bgr_uint8。"""
import cv2
import numpy as np


# ---------- 基础工具 ----------
def guided_self(img_f, r, eps):
    """自导向滤波（I=p），float32 单/多通道逐通道处理。"""
    d = 2 * r + 1
    mean_I = cv2.boxFilter(img_f, -1, (d, d))
    mean_II = cv2.boxFilter(img_f * img_f, -1, (d, d))
    var = mean_II - mean_I * mean_I
    a = var / (var + eps)
    b = mean_I - a * mean_I
    return cv2.boxFilter(a, -1, (d, d)) * img_f + cv2.boxFilter(b, -1, (d, d))


def _u8(x):
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


# ---------- 候选算子 ----------
def usm(amount, radius=2.0, thresh=0.0):
    def fn(img):
        f = img.astype(np.float32)
        blur = cv2.GaussianBlur(f, (0, 0), radius)
        d = f - blur
        if thresh > 0:
            d = np.where(np.abs(d) < thresh, 0, d)
        return _u8(f + amount * d)
    return fn


def gblur(sigma):
    def fn(img):
        return _u8(cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma))
    return fn


def guided(r, eps, amount=1.0):
    """amount=1 -> 纯去噪；amount<0 视为在细节层上做增强。"""
    def fn(img):
        f = img.astype(np.float32)
        base = guided_self(f, r, eps)
        return _u8(base + (1 - amount) * (f - base)) if amount != 1.0 else _u8(base)
    return fn


def guided_sharp(r, eps, boost):
    """导向滤波分解：base + boost*detail。"""
    def fn(img):
        f = img.astype(np.float32)
        base = guided_self(f, r, eps)
        return _u8(base + boost * (f - base))
    return fn


def bilat(d, sc, ss):
    def fn(img):
        return cv2.bilateralFilter(img, d, sc, ss)
    return fn


def nlm(h, hc=None, tw=7, sw=21):
    def fn(img):
        return cv2.fastNlMeansDenoisingColored(img, None, h, h if hc is None else hc, tw, sw)
    return fn


def clahe(clip, grid):
    def fn(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        lab[:, :, 0] = c.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return fn


def gain_bias(g, b):
    def fn(img):
        return _u8(img.astype(np.float32) * g + b)
    return fn


def usm_y(amount, radius=2.0):
    """只在 Y 通道锐化，色度不动。"""
    def fn(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        b = cv2.GaussianBlur(y[:, :, 0], (0, 0), radius)
        y[:, :, 0] = np.clip(y[:, :, 0] + amount * (y[:, :, 0] - b), 0, 255)
        return cv2.cvtColor(_u8(y), cv2.COLOR_YCrCb2BGR)
    return fn


def adaptive_usm(amount, radius, sigma_lo, sigma_hi):
    """按局部标准差调制锐化量：平坦区不锐化，强边缘也不锐化（防振铃）。"""
    def fn(img):
        f = img.astype(np.float32)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = cv2.boxFilter(g, -1, (9, 9))
        v = cv2.boxFilter(g * g, -1, (9, 9)) - m * m
        s = np.sqrt(np.maximum(v, 0))
        w = np.clip((s - sigma_lo) / max(sigma_hi - sigma_lo, 1e-6), 0, 1)[:, :, None]
        blur = cv2.GaussianBlur(f, (0, 0), radius)
        return _u8(f + amount * w * (f - blur))
    return fn


def denoise_flat(r, eps, sigma_edge):
    """只在平坦区做导向去噪，边缘区保留原图。"""
    def fn(img):
        f = img.astype(np.float32)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = cv2.boxFilter(g, -1, (9, 9))
        v = cv2.boxFilter(g * g, -1, (9, 9)) - m * m
        s = np.sqrt(np.maximum(v, 0))
        w = np.clip(1 - s / sigma_edge, 0, 1)[:, :, None]  # 平坦=1
        base = guided_self(f, r, eps)
        return _u8(f + w * (base - f))
    return fn


def msharp(a1, a2, s1=1.0, s2=3.0):
    """多尺度：分别控制中频/高频增益。"""
    def fn(img):
        f = img.astype(np.float32)
        b1 = cv2.GaussianBlur(f, (0, 0), s1)
        b2 = cv2.GaussianBlur(f, (0, 0), s2)
        hi = f - b1        # 高频
        mid = b1 - b2      # 中频
        return _u8(f + a1 * mid + a2 * hi)
    return fn


def wavelet(sigma, mode="soft"):
    from skimage.restoration import denoise_wavelet
    def fn(img):
        x = img.astype(np.float32) / 255.0
        o = denoise_wavelet(x, sigma=sigma / 255.0, mode=mode, channel_axis=2,
                            convert2ycbcr=True, method="BayesShrink", rescale_sigma=True)
        return _u8(o * 255.0)
    return fn


def tv(weight):
    from skimage.restoration import denoise_tv_chambolle
    def fn(img):
        o = denoise_tv_chambolle(img.astype(np.float32) / 255.0, weight=weight, channel_axis=2)
        return _u8(o * 255.0)
    return fn


def median(k):
    def fn(img):
        return cv2.medianBlur(img, k)
    return fn


def chain(*fns):
    def fn(img):
        for f in fns:
            img = f(img)
        return img
    return fn


def guided_joint(guide_f, src_f, r, eps):
    """联合导向滤波：以 guide 为引导滤 src（均为 float32 单通道）。"""
    d = 2 * r + 1
    mI = cv2.boxFilter(guide_f, -1, (d, d))
    mp = cv2.boxFilter(src_f, -1, (d, d))
    mIp = cv2.boxFilter(guide_f * src_f, -1, (d, d))
    mII = cv2.boxFilter(guide_f * guide_f, -1, (d, d))
    a = (mIp - mI * mp) / (mII - mI * mI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, (d, d)) * guide_f + cv2.boxFilter(b, -1, (d, d))


def chroma_guided(r, eps, amount=1.0):
    """只处理 Cr/Cb：用 Y 做引导做联合导向滤波（修 4:2:0 上采样伪影）。"""
    def fn(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        g = y[:, :, 0]
        for c in (1, 2):
            o = guided_joint(g, y[:, :, c], r, eps)
            y[:, :, c] = y[:, :, c] + amount * (o - y[:, :, c])
        return cv2.cvtColor(_u8(y), cv2.COLOR_YCrCb2BGR)
    return fn


def chroma_blur(sigma):
    def fn(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y[:, :, 1:] = cv2.GaussianBlur(y[:, :, 1:], (0, 0), sigma)
        return cv2.cvtColor(_u8(y), cv2.COLOR_YCrCb2BGR)
    return fn


def luma_guided(r, eps, amount=1.0):
    """只对 Y 做自导向去噪，色度不动。"""
    def fn(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        o = guided_self(y[:, :, 0], r, eps)
        y[:, :, 0] = y[:, :, 0] + amount * (o - y[:, :, 0])
        return cv2.cvtColor(_u8(y), cv2.COLOR_YCrCb2BGR)
    return fn


def chroma_reup(r, eps, blend=1.0):
    """把色度按 4:2:0 网格降采样回半分辨率，再用 Y 引导做联合上采样（fast guided filter）。"""
    def fn(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        Y = y[:, :, 0]
        H, W = Y.shape
        hw, hh = (W + 1) // 2, (H + 1) // 2
        Yl = cv2.resize(Y, (hw, hh), interpolation=cv2.INTER_AREA)
        d = 2 * r + 1
        mI = cv2.boxFilter(Yl, -1, (d, d))
        mII = cv2.boxFilter(Yl * Yl, -1, (d, d))
        varI = mII - mI * mI
        for c in (1, 2):
            Pl = cv2.resize(y[:, :, c], (hw, hh), interpolation=cv2.INTER_AREA)
            mp = cv2.boxFilter(Pl, -1, (d, d))
            mIp = cv2.boxFilter(Yl * Pl, -1, (d, d))
            a = (mIp - mI * mp) / (varI + eps)
            b = mp - a * mI
            au = cv2.resize(cv2.boxFilter(a, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
            bu = cv2.resize(cv2.boxFilter(b, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
            o = au * Y + bu
            y[:, :, c] = y[:, :, c] + blend * (o - y[:, :, c])
        return cv2.cvtColor(_u8(y), cv2.COLOR_YCrCb2BGR)
    return fn
