"""双重压缩检测：8x8 网格分块度、DCT 直方图梳状周期、对候选一次表的整数残差。"""
import sys, os, io
import numpy as np
from scipy.fft import dctn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegparse import parse, dqt_natural

Image.MAX_IMAGE_PIXELS = None

HUAWEI_LUM = dqt_natural([3,3,2,1,4,4,4,4,4,4,6,4,7,6,3,9,6,6,6,4,17,15,8,4,12,14,9,19,9,9,13,8,
                          13,11,14,12,25,21,21,13,9,18,15,24,27,14,17,12,22,30,22,15,27,13,20,27,
                          30,26,37,36,31,28,22,36])


def load_Y(path, primary=None):
    raw = open(path, 'rb').read()
    if primary:
        raw = raw[:primary]
    im = Image.open(io.BytesIO(raw))
    return np.array(im.convert('YCbCr'))[..., 0]


def blockiness(y):
    """返回 8 个相位上的横向一阶差分绝对均值（相位 7 = 跨越 8x8 边界）"""
    d = np.abs(np.diff(y.astype(np.int16), axis=1)).astype(np.float64)
    W = d.shape[1]
    ph = np.zeros(8)
    for p in range(8):
        ph[p] = d[:, p::8].mean()
    dv = np.abs(np.diff(y.astype(np.int16), axis=0)).astype(np.float64)
    phv = np.zeros(8)
    for p in range(8):
        phv[p] = dv[p::8, :].mean()
    return ph, phv


def blockdct(y, offset=(0, 0)):
    oy, ox = offset
    a = y[oy:, ox:]
    H8, W8 = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
    a = a[:H8, :W8].astype(np.float32) - 128.0
    a = a.reshape(H8//8, 8, W8//8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return dctn(a, axes=(1, 2), norm='ortho')


def comb_score(vals, period):
    """vals: 一维去量化 DCT 系数。返回该周期上的梳状强度（归一化 FFT 幅值）"""
    lim = 60
    h, _ = np.histogram(vals, bins=np.arange(-lim, lim + 1) - 0.5)
    h = h.astype(np.float64)
    h = h - np.convolve(h, np.ones(9)/9, mode='same')  # 去掉平滑趋势
    F = np.abs(np.fft.rfft(h))
    n = len(h)
    k = n / period
    ki = int(round(k))
    if ki <= 1 or ki >= len(F):
        return 0.0
    base = np.median(F[2:len(F)//2])
    return float(F[ki] / (base + 1e-9))
