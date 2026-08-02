"""决定性对比：候选 q96/420/opt/prog vs 真实已部署基线（main.py save_image, q98 默认采样）。

提交 C（71.5596 分）走的就是 main.py 的 save_image(q=98)，所以"未处理 LQ"在评测方眼里
是 decode(encode(LQ, q98, cv2默认采样))，而不是原始 LQ 像素。
"""
import io
import os

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"

CAND = [cv2.IMWRITE_JPEG_QUALITY, 96,
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
BASE_MAIN = [cv2.IMWRITE_JPEG_QUALITY, 98]  # main.py 实际用的参数


def load_bgr(p):
    return np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy()


def gt_path(c):
    for e in (".png", ".jpg"):
        p = os.path.join(VAL, f"{c}_gt{e}")
        if os.path.exists(p):
            return p


def rt(bgr, p):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, p)[1], cv2.IMREAD_COLOR)


def metrics(o, g):
    return (sk_psnr(g, o, data_range=255),
            sk_ssim(g, o, data_range=255, channel_axis=2))


# cv2 默认采样因子确认
probe = np.random.randint(0, 255, (64, 64, 3), np.uint8)
im = Image.open(io.BytesIO(cv2.imencode(".jpg", probe, BASE_MAIN)[1].tobytes()))
print("cv2 默认(仅设 quality)编码的采样因子 layer =", im.layer)

rows = []
for c in ["case1", "case2", "case3"]:
    lq = load_bgr(os.path.join(VAL, f"{c}_lq.jpg"))
    gt = load_bgr(gt_path(c))
    raw = metrics(lq, gt)
    base = metrics(rt(lq, BASE_MAIN), gt)
    cand = metrics(rt(lq, CAND), gt)
    rows.append((c, raw, base, cand))
    print(f"{c}: raw={raw[0]:.4f}/{raw[1]:.6f}  q98base={base[0]:.4f}/{base[1]:.6f}  "
          f"q96cand={cand[0]:.4f}/{cand[1]:.6f}", flush=True)

print("\n=== 相对 raw LQ 像素（候选方使用的基线）===")
dp = [r[3][0] - r[1][0] for r in rows]
ds = [r[3][1] - r[1][1] for r in rows]
print(f"per-image dPSNR {['%+.4f' % v for v in dp]}  mean {np.mean(dp):+.4f}")
print(f"per-image dSSIM {['%+.6f' % v for v in ds]}  mean {np.mean(ds):+.6f}")

print("\n=== 相对真实部署基线 main.py q98（决定分数的那个）===")
dp = [r[3][0] - r[2][0] for r in rows]
ds = [r[3][1] - r[2][1] for r in rows]
print(f"per-image dPSNR {['%+.4f' % v for v in dp]}  mean {np.mean(dp):+.4f}")
print(f"per-image dSSIM {['%+.6f' % v for v in ds]}  mean {np.mean(ds):+.6f}")
