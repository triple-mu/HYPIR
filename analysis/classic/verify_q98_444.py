"""独立复现 q98_444 候选算子：中心 1024 crop 与全图各测一次，逐图给 delta。"""
import io
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = "/home/ubuntu/workspace/contest/CSIG-2026"
sys.path.insert(0, os.path.join(ROOT, "BasicSR"))
VAL = os.path.join(ROOT, "赛题二/验证集")
CASES = ["case1", "case2", "case3"]
GTF = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}

# ---------------- 候选算子（原样抄自任务） ----------------
_P = [cv2.IMWRITE_JPEG_QUALITY, 98,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)


# ---------------- 指标 ----------------
_lp = None


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255,
                                       gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))


def lpips_score(a, b, tile=1024):
    global _lp
    if _lp is None:
        import lpips as M
        _lp = M.LPIPS(net="alex").cuda().eval()
    h, w = a.shape[:2]
    tot, area = 0.0, 0
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                pa, pb = a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]
                if min(pa.shape[:2]) < 64:
                    continue
                ta = torch.from_numpy(pa[:, :, ::-1].copy()).cuda().permute(2, 0, 1)[None].float() / 127.5 - 1
                tb = torch.from_numpy(pb[:, :, ::-1].copy()).cuda().permute(2, 0, 1)[None].float() / 127.5 - 1
                n = pa.shape[0] * pa.shape[1]
                tot += float(_lp(ta, tb).item()) * n
                area += n
    return tot / area


def niqe(img):
    from basicsr.metrics.niqe import calculate_niqe
    return float(calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y"))


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def cvdec_bytes(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def report(tag, gt, lq, proc):
    m0 = (psnr(gt, lq), ssim(gt, lq), lpips_score(gt, lq), niqe(lq))
    m1 = (psnr(gt, proc), ssim(gt, proc), lpips_score(gt, proc), niqe(proc))
    print(f"  [{tag}] LQ   psnr={m0[0]:9.5f} ssim={m0[1]:.6f} lpips={m0[2]:.6f} niqe={m0[3]:.4f}")
    print(f"  [{tag}] proc psnr={m1[0]:9.5f} ssim={m1[1]:.6f} lpips={m1[2]:.6f} niqe={m1[3]:.4f}")
    print(f"  [{tag}] DELTA dPSNR={m1[0]-m0[0]:+.6f} dSSIM={m1[1]-m0[1]:+.8f} "
          f"dLPIPS={m1[2]-m0[2]:+.6f} dNIQE={m1[3]-m0[3]:+.4f}  "
          f"roundtripPSNR(lq,proc)={psnr(lq, proc):.3f} maxabs={int(np.abs(proc.astype(int)-lq.astype(int)).max())}",
          flush=True)
    return np.array(m1) - np.array(m0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "crop"
    acc = []
    for c in CASES:
        gt = load(os.path.join(VAL, GTF[c]))
        lq_path = os.path.join(VAL, f"{c}_lq.jpg")
        lq = load(lq_path)
        assert gt.shape == lq.shape, (gt.shape, lq.shape)
        lq_cv = cvdec_bytes(lq_path)
        print(f"\n=== {c} shape={gt.shape} PIL-vs-cv2 LQ 解码差 maxabs="
              f"{int(np.abs(lq.astype(int)-lq_cv.astype(int)).max())} "
              f"psnr={psnr(lq, lq_cv):.3f} ===", flush=True)
        if mode == "crop":
            h, w = gt.shape[:2]
            y, x = h // 2 - 512, w // 2 - 512
            gt = gt[y:y + 1024, x:x + 1024].copy()
            lq = lq[y:y + 1024, x:x + 1024].copy()
            lq_cv = lq_cv[y:y + 1024, x:x + 1024].copy()
        # A: 基线与算子输出都基于 PIL 解码的 LQ（算子内部用 cv2 解码自己的 jpeg）
        d = report("PILbase", gt, lq, f(lq))
        if mode == "crop":
            # B: 完全 cv2 口径（基线 = cv2 解码原始文件），排除解码器差异混淆
            d2 = report("CV2base ", gt, lq_cv, f(lq_cv))
        else:
            d2 = d
        acc.append((d, d2))
        del gt, lq, lq_cv
    for i, tag in enumerate(("PILbase", "CV2base")):
        a = np.mean([x[i] for x in acc], axis=0)
        print(f"\n### {mode} 三图平均 [{tag}]: dPSNR={a[0]:+.6f} dSSIM={a[1]:+.8f} "
              f"dLPIPS={a[2]:+.6f} dNIQE={a[3]:+.4f}")
        per = np.array([x[i][0] for x in acc])
        print(f"    逐图 dPSNR: " + " ".join(f"{c}={v:+.6f}" for c, v in zip(CASES, per))
              + f"   正的图数={int((per > 0).sum())}/3")
        pers = np.array([x[i][1] for x in acc])
        print(f"    逐图 dSSIM: " + " ".join(f"{c}={v:+.8f}" for c, v in zip(CASES, pers))
              + f"   正的图数={int((pers > 0).sum())}/3")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n耗时 {time.time()-t:.1f}s")
