"""最终交付候选：自包含函数 f(bgr_uint8)->bgr_uint8，只依赖 cv2/numpy。"""
import cv2
import numpy as np


def cand_chroma_reup(img):
    """E：只重建色度。按 4:2:0 网格降采样 Cr/Cb，用 Y 做引导做联合上采样。"""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = y[:, :, 0]
    H, W = Y.shape
    hw, hh = (W + 1) // 2, (H + 1) // 2
    d, eps = 5, 16.0
    Yl = cv2.resize(Y, (hw, hh), interpolation=cv2.INTER_AREA)
    mI = cv2.boxFilter(Yl, -1, (d, d))
    varI = cv2.boxFilter(Yl * Yl, -1, (d, d)) - mI * mI
    for c in (1, 2):
        Pl = cv2.resize(y[:, :, c], (hw, hh), interpolation=cv2.INTER_AREA)
        mp = cv2.boxFilter(Pl, -1, (d, d))
        a = (cv2.boxFilter(Yl * Pl, -1, (d, d)) - mI * mp) / (varI + eps)
        b = mp - a * mI
        au = cv2.resize(cv2.boxFilter(a, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
        bu = cv2.resize(cv2.boxFilter(b, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
        y[:, :, c] = au * Y + bu
    return cv2.cvtColor(np.clip(y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _guided_self(f, r, eps):
    d = 2 * r + 1
    m = cv2.boxFilter(f, -1, (d, d))
    var = cv2.boxFilter(f * f, -1, (d, d)) - m * m
    a = var / (var + eps)
    return cv2.boxFilter(a, -1, (d, d)) * f + cv2.boxFilter(m - a * m, -1, (d, d))


def cand_chroma_reup_guided(img):
    """A：色度重建 + 3x3 自导向去噪（eps=4，只削 JPEG 量化噪声，保边）。"""
    o = cand_chroma_reup(img).astype(np.float32)
    return np.clip(_guided_self(o, 1, 4.0) + 0.5, 0, 255).astype(np.uint8)


def cand_chroma_reup_guided_strong(img):
    """C：同 A 但更强（色度 eps=64，亮度 eps=12），均值增益更大、逐图风险略升。"""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = y[:, :, 0]
    H, W = Y.shape
    hw, hh = (W + 1) // 2, (H + 1) // 2
    d, eps = 5, 64.0
    Yl = cv2.resize(Y, (hw, hh), interpolation=cv2.INTER_AREA)
    mI = cv2.boxFilter(Yl, -1, (d, d))
    varI = cv2.boxFilter(Yl * Yl, -1, (d, d)) - mI * mI
    for c in (1, 2):
        Pl = cv2.resize(y[:, :, c], (hw, hh), interpolation=cv2.INTER_AREA)
        mp = cv2.boxFilter(Pl, -1, (d, d))
        a = (cv2.boxFilter(Yl * Pl, -1, (d, d)) - mI * mp) / (varI + eps)
        b = mp - a * mI
        au = cv2.resize(cv2.boxFilter(a, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
        bu = cv2.resize(cv2.boxFilter(b, -1, (d, d)), (W, H), interpolation=cv2.INTER_LINEAR)
        y[:, :, c] = au * Y + bu
    o = cv2.cvtColor(np.clip(y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
    return np.clip(_guided_self(o, 1, 12.0) + 0.5, 0, 255).astype(np.uint8)


def cand_identity(img):
    return img
