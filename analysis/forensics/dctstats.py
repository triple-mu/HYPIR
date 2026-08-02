"""每张图：解码 Y 通道 -> 8x8 分块 DCT -> 量化系数统计 + q95 重编码码率。"""
import sys, os, io, json, glob
import numpy as np
from scipy.fft import dctn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegparse import parse, dqt_natural

Image.MAX_IMAGE_PIXELS = None


def block_dct_Y(y):
    """y: uint8 HxW -> (nb,8,8) float32 DCT-II 正交，JPEG 约定（-128 偏移）"""
    H, W = y.shape
    H8, W8 = H // 8 * 8, W // 8 * 8
    a = y[:H8, :W8].astype(np.float32) - 128.0
    a = a.reshape(H8 // 8, 8, W8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return dctn(a, axes=(1, 2), norm='ortho').astype(np.float32)


def analyze(path, primary_bytes=None, qtab_nat=None, do_reenc=True):
    raw = open(path, 'rb').read()
    if primary_bytes is not None:
        raw = raw[:primary_bytes]
    im = Image.open(io.BytesIO(raw))
    ycc = im.convert('YCbCr')
    y = np.array(ycc)[..., 0]
    D = block_dct_Y(y)
    nb = D.shape[0]
    out = {'nblocks': nb, 'H': y.shape[0], 'W': y.shape[1]}
    # 每个频率位置的 |DCT| 均值（自然序 8x8）
    absmean = np.abs(D).mean(axis=0)
    out['absmean'] = [round(float(x), 4) for x in absmean.reshape(-1)]
    if qtab_nat is not None:
        q = np.array(qtab_nat, dtype=np.float32).reshape(8, 8)
        C = D / q  # 量化系数（近似整数）
        out['nonzero_frac'] = [round(float(x), 5) for x in (np.abs(C) >= 0.5).mean(axis=0).reshape(-1)]
        # 与最近整数的偏差 -> 检验「像素确实来自该量化表」
        res = np.abs(C - np.round(C))
        out['int_resid_mean'] = [round(float(x), 4) for x in res.mean(axis=0).reshape(-1)]
        out['nz_total'] = round(float((np.abs(C) >= 0.5).mean()), 5)
    if do_reenc:
        b = io.BytesIO()
        im.convert('RGB').save(b, 'JPEG', quality=95, subsampling=2)
        out['reenc_q95_bytes'] = b.tell()
        out['reenc_q95_bpp'] = round(b.tell() * 8 / (y.shape[0] * y.shape[1]), 4)
    out['orig_bytes'] = len(raw)
    out['orig_bpp'] = round(len(raw) * 8 / (y.shape[0] * y.shape[1]), 4)
    return out, D, y


if __name__ == '__main__':
    base = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二'
    files = sorted(glob.glob(base + '/测试集/*.jpg'),
                   key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
    files += sorted(glob.glob(base + '/验证集/*'))
    res = {}
    for f in files:
        r = parse(f)
        if 'error' in r:
            continue
        pb = (r['size'] - r['trailing']) if r['trailing'] else None
        qn = dqt_natural(r['dqt'][0]['zz'])
        o, D, y = analyze(f, pb, qn)
        o['qmean'] = round(sum(qn) / 64, 3)
        res[os.path.basename(f)] = o
        print(os.path.basename(f), o['W'], o['H'], 'origbpp=%.3f' % o['orig_bpp'],
              'q95bpp=%.3f' % o['reenc_q95_bpp'], 'nz=%.4f' % o['nz_total'], flush=True)
        del D, y
    json.dump(res, open('/tmp/dctstats.json', 'w'))
