"""验证候选代码可运行并计时（12MP）。"""
import time

import cv2
import numpy as np

SRC = "赛题二/测试集/case100.jpg"


CAND = {}

CAND["bytecopy"] = r'''
import shutil, cv2, numpy as np
def f(bgr):
    """像素恒等：提交时不重编码，直接复制原始字节（见 save_bytecopy）。"""
    return bgr
def save_bytecopy(src_path, dst_path):
    """零损耗保存：原样拷贝；MPO 多帧文件截断到第一帧（像素 bit-identical，省 55MB）。"""
    raw = open(src_path, "rb").read()
    i = raw.find(b"\xff\xd9\xff\xd8")
    open(dst_path, "wb").write(raw[:i + 2] if i > 0 else raw)
'''

CAND["q96_420_optprog"] = r'''
import cv2, numpy as np
_P = [cv2.IMWRITE_JPEG_QUALITY, 96,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
      cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
def f(bgr):
    """评测方最终看到的像素 = q96/4:2:0 编解码一轮后的结果。"""
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
def save(path, bgr):
    cv2.imencode(".jpg", bgr, _P)[1].tofile(path)
'''

CAND["q95_420_optprog"] = r'''
import cv2, numpy as np
_P = [cv2.IMWRITE_JPEG_QUALITY, 95,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
      cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
def f(bgr):
    """q95/4:2:0 == LQ 源文件的量化表与采样，重编码近乎幂等。"""
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
def save(path, bgr):
    cv2.imencode(".jpg", bgr, _P)[1].tofile(path)
'''

CAND["q98_420_baseline"] = r'''
import cv2, numpy as np
_P = [cv2.IMWRITE_JPEG_QUALITY, 98,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
def save(path, bgr):
    cv2.imencode(".jpg", bgr, _P)[1].tofile(path)
'''

CAND["q98_444_reject"] = r'''
import cv2, numpy as np
_P = [cv2.IMWRITE_JPEG_QUALITY, 98,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
'''

CAND["q85_420_reject"] = r'''
import cv2, numpy as np
_P = [cv2.IMWRITE_JPEG_QUALITY, 85,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
'''


def main():
    bgr = cv2.imread(SRC)
    print("12MP:", bgr.shape)
    for name, src in CAND.items():
        ns = {}
        exec(src, ns)
        f = ns["f"]
        out = f(bgr)
        assert out.dtype == np.uint8 and out.shape == bgr.shape, (name, out.dtype, out.shape)
        t = time.time()
        for _ in range(3):
            f(bgr)
        print(f"{name:20s} OK  shape={out.shape} dtype={out.dtype}  {(time.time()-t)/3*1000:.0f} ms")


if __name__ == "__main__":
    main()
