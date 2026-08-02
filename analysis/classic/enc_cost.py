"""编码代价曲线：一个处理算子的增益，经过不同 JPEG 质量后还剩多少。

用 3 个代表性算子（弱去噪 / 弱锐化 / 去噪+锐化）产生"处理后像素"，
比较 pristine（未编码 float->uint8）与 encode->decode 之后相对 GT 的 PSNR/SSIM。
"""
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_lib import psnr, ssim

VAL = "赛题二/验证集"
CASES = ["case1", "case2", "case3"]
GT = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}
S444 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
S420 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420


def load_bgr(p):
    return np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy()


def center(img, n=2048):
    h, w = img.shape[:2]
    return img[h // 2 - n // 2:h // 2 + n // 2, w // 2 - n // 2:w // 2 + n // 2]


def op_denoise(x):
    return cv2.bilateralFilter(x, 5, 12, 5)


def op_sharpen(x):
    b = cv2.GaussianBlur(x, (0, 0), 1.0)
    return cv2.addWeighted(x, 1.4, b, -0.4, 0)


def op_both(x):
    return op_sharpen(op_denoise(x))


OPS = {"denoise_bilat": op_denoise, "usm0.4": op_sharpen, "denoise+usm": op_both}
QS = [(85, S420), (90, S420), (92, S420), (95, S420), (96, S420),
      (97, S420), (98, S420), (98, S444), (100, S420), (100, S444)]


def main():
    data = {c: (center(load_bgr(os.path.join(VAL, GT[c]))),
                center(load_bgr(os.path.join(VAL, f"{c}_lq.jpg")))) for c in CASES}
    print("(2048x2048 中心 crop, 三图平均)\n")
    lqp = np.mean([psnr(g, l) for g, l in data.values()])
    lqs = np.mean([ssim(g, l) for g, l in data.values()])
    print(f"LQ 基线: psnr={lqp:.4f} ssim={lqs:.5f}\n")
    for oname, op in OPS.items():
        proc = {c: op(l) for c, (g, l) in data.items()}
        pp = np.mean([psnr(data[c][0], proc[c]) for c in CASES])
        ps = np.mean([ssim(data[c][0], proc[c]) for c in CASES])
        print(f"--- {oname}: pristine dPSNR={pp-lqp:+.4f} dSSIM={ps-lqs:+.6f} ---")
        for q, s in QS:
            ds, dp, sz = [], [], []
            for c in CASES:
                ok, buf = cv2.imencode(".jpg", proc[c], [cv2.IMWRITE_JPEG_QUALITY, q,
                                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR, s])
                d = np.asarray(Image.open(io.BytesIO(buf.tobytes())).convert("RGB"))[:, :, ::-1].copy()
                dp.append(psnr(data[c][0], d))
                ds.append(ssim(data[c][0], d))
                sz.append(len(buf))
            sname = "444" if s == S444 else "420"
            print(f"  q{q}_{sname:3s} dPSNR={np.mean(dp)-lqp:+.4f} dSSIM={np.mean(ds)-lqs:+.6f} "
                  f"| 编码吃掉 PSNR {np.mean(dp)-pp:+.4f} SSIM {np.mean(ds)-ps:+.6f} "
                  f"| {np.mean(sz)/1e6:.2f}MB")
        print()


if __name__ == "__main__":
    main()
