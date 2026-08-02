"""算子族 D 全图版：12MP 全分辨率扫 JPEG 质量/色度采样/熵编码，含尺寸统计。

PSNR/SSIM 用全图（评判主指标），LPIPS/NIQE 用固定 5 个 1024 crop 采样（patch 级指标，采样足够代表）。
"""
import io
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_lib import psnr, ssim, lpips_score, niqe

VAL = "赛题二/验证集"
CASES = ["case1", "case2", "case3"]
GT = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}
S444 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
S422 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422
S420 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
KEYS = ("psnr", "ssim", "lpips", "niqe")


def load_bgr(path):
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def crops(img):
    """固定 5 块 1024 采样：中心 + 四象限中心。"""
    h, w = img.shape[:2]
    cs = [(h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4),
          (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
    return [img[y - 512:y + 512, x - 512:x + 512] for y, x in cs]


def enc(bgr, q, sampling, optimize=False, progressive=False, chroma=None):
    p = [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling]
    if optimize:
        p += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    if progressive:
        p += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    if chroma:
        p += [cv2.IMWRITE_JPEG_LUMA_QUALITY, q, cv2.IMWRITE_JPEG_CHROMA_QUALITY, chroma]
    ok, buf = cv2.imencode(".jpg", bgr, p)
    assert ok
    return buf.tobytes()


def dec(buf):
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))[:, :, ::-1].copy()


# (name, quality, sampling, optimize, progressive, chroma_quality)
VARIANTS = [
    ("q85_420", 85, S420, 0, 0, None), ("q90_420", 90, S420, 0, 0, None),
    ("q92_420", 92, S420, 0, 0, None), ("q94_420", 94, S420, 0, 0, None),
    ("q95_420", 95, S420, 0, 0, None), ("q95_420_op", 95, S420, 1, 1, None),
    ("q95_422", 95, S422, 0, 0, None), ("q95_444", 95, S444, 0, 0, None),
    ("q96_420", 96, S420, 0, 0, None), ("q97_420", 97, S420, 0, 0, None),
    ("q98_420", 98, S420, 0, 0, None), ("q98_444", 98, S444, 0, 0, None),
    ("q100_420", 100, S420, 0, 0, None), ("q100_444", 100, S444, 0, 0, None),
    ("q98L_90C_420", 98, S420, 0, 0, 90),
]


def measure(gt, img, gtc, size, rt):
    ic = crops(img)
    return dict(psnr=psnr(gt, img), ssim=ssim(gt, img),
                lpips=float(np.mean([lpips_score(a, b) for a, b in zip(gtc, ic)])),
                niqe=float(np.mean([niqe(b) for b in ic])), size=size, rt=rt)


def main():
    res = {}
    for c in CASES:
        lq_path = os.path.join(VAL, f"{c}_lq.jpg")
        gt = load_bgr(os.path.join(VAL, GT[c]))
        lq = load_bgr(lq_path)
        gtc = crops(gt)
        m = measure(gt, lq, gtc, os.path.getsize(lq_path), float("inf"))
        res.setdefault("bytecopy_LQ", {})[c] = m
        print(f"[{c}] {'bytecopy_LQ':12s} psnr={m['psnr']:.4f} ssim={m['ssim']:.5f} "
              f"lpips={m['lpips']:.5f} niqe={m['niqe']:.4f} size={m['size']/1e6:.2f}MB", flush=True)
        for name, q, s, o, pg, ch in VARIANTS:
            buf = enc(lq, q, s, o, pg, ch)
            d = dec(buf)
            m = measure(gt, d, gtc, len(buf), psnr(lq, d))
            res.setdefault(name, {})[c] = m
            print(f"[{c}] {name:12s} psnr={m['psnr']:.4f} ssim={m['ssim']:.5f} "
                  f"lpips={m['lpips']:.5f} niqe={m['niqe']:.4f} rt={m['rt']:.2f} "
                  f"size={m['size']/1e6:.2f}MB", flush=True)
        del gt, lq, gtc

    np.save("csig_bench/enc_full_res.npy", np.array([res], dtype=object), allow_pickle=True)
    ref = {k: np.mean([res["bytecopy_LQ"][c][k] for c in CASES]) for k in KEYS}
    print("\n=== 全图三图平均，Δ 相对 bytecopy_LQ（直接复制原始字节）===")
    print(f"bytecopy_LQ 绝对: psnr={ref['psnr']:.4f} ssim={ref['ssim']:.5f} "
          f"lpips={ref['lpips']:.5f} niqe={ref['niqe']:.4f}")
    for name in res:
        a = {k: np.mean([res[name][c][k] for c in CASES]) for k in KEYS}
        sz = np.mean([res[name][c]["size"] for c in CASES])
        rt = np.mean([res[name][c]["rt"] for c in CASES])
        print(f"{name:12s} dPSNR={a['psnr']-ref['psnr']:+.4f} dSSIM={a['ssim']-ref['ssim']:+.6f} "
              f"dLPIPS={a['lpips']-ref['lpips']:+.6f} dNIQE={a['niqe']-ref['niqe']:+.4f} "
              f"rt={rt:6.2f} size={sz/1e6:.2f}MB")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n耗时 {time.time()-t:.1f}s")
