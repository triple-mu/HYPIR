"""一次压缩量化步长估计：对每个 DCT 频点，用 S(a)=<cos(2*pi*D/a)> 找周期。"""
import sys, os, io
import numpy as np
from scipy.fft import dctn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegparse import parse, dqt_natural

Image.MAX_IMAGE_PIXELS = None

HU_L = dqt_natural([3,3,2,1,4,4,4,4,4,4,6,4,7,6,3,9,6,6,6,4,17,15,8,4,12,14,9,19,9,9,13,8,
                    13,11,14,12,25,21,21,13,9,18,15,24,27,14,17,12,22,30,22,15,27,13,20,27,
                    30,26,37,36,31,28,22,36])
Q95_L = dqt_natural([2,1,1,1,1,1,2,1,1,1,2,2,2,2,2,4,3,2,2,2,2,5,4,4,3,4,6,5,6,6,6,5,6,6,6,7,
                     9,8,6,7,9,7,6,6,8,11,8,9,10,10,10,10,10,6,8,11,12,11,10,12,9,10,10,10])


def blockdct(y):
    H8, W8 = y.shape[0]//8*8, y.shape[1]//8*8
    a = y[:H8, :W8].astype(np.float32) - 128.0
    a = a.reshape(H8//8, 8, W8//8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return dctn(a, axes=(1, 2), norm='ortho')


def S(D, a):
    return float(np.cos(2*np.pi*D/a).mean())


def probe(y, positions, cands=range(2, 41), label=''):
    D = blockdct(y)
    rows = []
    for (u, v) in positions:
        d = D[:, u, v].astype(np.float64)
        d = d[np.abs(d) < 200]
        sc = {a: S(d, a) for a in cands}
        best = max(sc, key=lambda a: sc[a])
        rows.append(((u, v), best, sc[best], sc, float(np.std(d))))
    return rows
