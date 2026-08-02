"""独立复现 q85_420 候选算子：中心 1024 crop 与全图各测一次，逐图打印。"""
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT + "/BasicSR")

VAL = os.path.join(ROOT, "赛题二/验证集")
CASES = ["case1", "case2", "case3"]
GT = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}

# ---------------- 候选算子：原样抄自待验证声明 ----------------
_P = [cv2.IMWRITE_JPEG_QUALITY, 85,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]


def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)
# ------------------------------------------------------------

_lpips_net = None


def load_bgr(path):
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        a, b, channel_axis=2, data_range=255,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def lpips_score(a, b, tile=1024, device="cuda"):
    global _lpips_net
    import torch
    if _lpips_net is None:
        import lpips as lpips_mod
        _lpips_net = lpips_mod.LPIPS(net="alex").to(device).eval()
    h, w = a.shape[:2]
    tot, area = 0.0, 0
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                pa, pb = a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]
                if min(pa.shape[:2]) < 64:
                    continue
                ta = torch.from_numpy(pa[:, :, ::-1].copy()).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
                tb = torch.from_numpy(pb[:, :, ::-1].copy()).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
                tot += float(_lpips_net(ta, tb).item()) * pa.shape[0] * pa.shape[1]
                area += pa.shape[0] * pa.shape[1]
    return tot / area


def niqe(img):
    from basicsr.metrics.niqe import calculate_niqe
    return float(calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y"))


def crops5(img):
    """全图 NIQE 太慢，用固定 5 块 1024 采样（中心 + 四象限中心）。"""
    h, w = img.shape[:2]
    cs = [(h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4),
          (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
    return [img[y - 512:y + 512, x - 512:x + 512] for y, x in cs]


def run(mode):
    print(f"\n{'='*78}\n### {mode}\n{'='*78}", flush=True)
    per = {}
    for c in CASES:
        gt = load_bgr(os.path.join(VAL, GT[c]))
        lq = load_bgr(os.path.join(VAL, f"{c}_lq.jpg"))
        assert gt.shape == lq.shape, (c, gt.shape, lq.shape)
        if mode == "crop1024":
            h, w = gt.shape[:2]
            y, x = h // 2 - 512, w // 2 - 512
            gt, lq = gt[y:y + 1024, x:x + 1024], lq[y:y + 1024, x:x + 1024]
        out = f(lq)
        assert out is not None and out.shape == lq.shape, (c, None if out is None else out.shape)

        if mode == "crop1024":
            nq_lq, nq_out = niqe(lq), niqe(out)
            lp_lq, lp_out = lpips_score(gt, lq), lpips_score(gt, out)
        else:
            gl, ll, ol = crops5(gt), crops5(lq), crops5(out)
            nq_lq = float(np.mean([niqe(z) for z in ll]))
            nq_out = float(np.mean([niqe(z) for z in ol]))
            lp_lq = float(np.mean([lpips_score(a, b) for a, b in zip(gl, ll)]))
            lp_out = float(np.mean([lpips_score(a, b) for a, b in zip(gl, ol)]))

        m = dict(psnr=(psnr(gt, lq), psnr(gt, out)), ssim=(ssim(gt, lq), ssim(gt, out)),
                 lpips=(lp_lq, lp_out), niqe=(nq_lq, nq_out))
        per[c] = m
        print(f"[{c}] shape={lq.shape}", flush=True)
        for k, (a, b) in m.items():
            print(f"    {k:6s} LQ={a:10.5f}  OUT={b:10.5f}  d={b-a:+.6f}", flush=True)
        del gt, lq, out

    print(f"\n--- {mode} 三图平均 Δ (OUT - LQ) ---")
    res = {}
    for k in ("psnr", "ssim", "lpips", "niqe"):
        d = [per[c][k][1] - per[c][k][0] for c in CASES]
        res[k] = float(np.mean(d))
        sign = "+" if all(x > 0 for x in d) else ("-" if all(x < 0 for x in d) else "混")
        print(f"d{k.upper():6s} mean={np.mean(d):+.6f}   per-case={[f'{x:+.6f}' for x in d]}  一致性={sign}")
    return res


if __name__ == "__main__":
    t = time.time()
    rc = run("crop1024")
    rf = run("full")
    print(f"\n{'='*78}\n声称: dPSNR=+0.0051 dSSIM=-0.000098 dLPIPS=-0.006183 dNIQE=-0.3417")
    print(f"crop: dPSNR={rc['psnr']:+.4f} dSSIM={rc['ssim']:+.6f} dLPIPS={rc['lpips']:+.6f} dNIQE={rc['niqe']:+.4f}")
    print(f"full: dPSNR={rf['psnr']:+.4f} dSSIM={rf['ssim']:+.6f} dLPIPS={rf['lpips']:+.6f} dNIQE={rf['niqe']:+.4f}")
    print(f"耗时 {time.time()-t:.1f}s")
