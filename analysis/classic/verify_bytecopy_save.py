"""检查 save_bytecopy 的 MPO 截断是否像素 bit-identical（验证集 3 张 + 测试集 100 张）。"""
import glob
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

SRCS = sorted(glob.glob('/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集/*_lq.jpg')) + \
       sorted(glob.glob('/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集/*.jpg'))


def save_bytecopy(src_path, dst_path):
    raw = open(src_path, "rb").read()
    i = raw.find(b"\xff\xd9\xff\xd8")
    open(dst_path, "wb").write(raw[:i + 2] if i > 0 else raw)


tmp = tempfile.mkdtemp()
tot_src = tot_dst = 0
bad = []
n_trunc = 0
for p in SRCS:
    d = os.path.join(tmp, os.path.basename(p))
    save_bytecopy(p, d)
    s, t = os.path.getsize(p), os.path.getsize(d)
    tot_src += s
    tot_dst += t
    if t < s:
        n_trunc += 1
    try:
        a = np.array(Image.open(p).convert('RGB'))
        b = np.array(Image.open(d).convert('RGB'))
        ac = cv2.imread(p, cv2.IMREAD_COLOR)
        bc = cv2.imread(d, cv2.IMREAD_COLOR)
        ok = a.shape == b.shape and np.array_equal(a, b) and np.array_equal(ac, bc)
    except Exception as e:
        ok = False
        print('DECODE FAIL', p, e)
    if not ok:
        bad.append((os.path.basename(p), s, t,
                    a.shape if 'a' in dir() else None, b.shape if 'b' in dir() else None))

print(f'files={len(SRCS)} truncated={n_trunc} mismatched={len(bad)}')
print(f'total bytes {tot_src/1e6:.1f}MB -> {tot_dst/1e6:.1f}MB  saved {(tot_src-tot_dst)/1e6:.1f}MB '
      f'({(tot_src-tot_dst)/tot_src*100:.1f}%)')
for b in bad:
    print('MISMATCH', b)
