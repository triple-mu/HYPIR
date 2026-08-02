"""独立复现 q96_420_optprog 算子在验证集三张图上的指标增量。

不使用候选方提供的任何评测脚本，指标全部本脚本自行计算。
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/BasicSR")
from basicsr.metrics import calculate_niqe  # noqa: E402

import lpips  # noqa: E402

VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"

# ------------------- 候选算子（原样复制自任务描述） -------------------
_P = [cv2.IMWRITE_JPEG_QUALITY, 96,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
      cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]


def f(bgr):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, _P)[1], cv2.IMREAD_COLOR)


# ------------------------------- 指标 -------------------------------
_lpips_net = None


def get_lpips():
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="alex").cuda().eval()
    return _lpips_net


def psnr(a, b):
    return sk_psnr(b, a, data_range=255)


def ssim(a, b):
    return sk_ssim(b, a, data_range=255, channel_axis=2)


def lpips_d(a_bgr, b_bgr):
    net = get_lpips()
    def prep(x):
        t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).cuda()
    with torch.no_grad():
        return float(net(prep(a_bgr), prep(b_bgr)).item())


def niqe(bgr):
    return float(calculate_niqe(bgr, crop_border=0, input_order="HWC", convert_to="y"))


def load_bgr(path):
    """按真实格式解码，忽略扩展名。"""
    im = Image.open(path).convert("RGB")
    return np.asarray(im)[:, :, ::-1].copy()


def center_crop(img, s=1024):
    h, w = img.shape[:2]
    y = ((h - s) // 2) // 8 * 8
    x = ((w - s) // 2) // 8 * 8
    return img[y:y + s, x:x + s].copy()


def gt_path(case):
    for ext in (".png", ".jpg"):
        p = os.path.join(VAL, f"{case}_gt{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(case)


def evaluate(tag, do_niqe=True):
    rows = []
    for case in ["case1", "case2", "case3"]:
        lq = load_bgr(os.path.join(VAL, f"{case}_lq.jpg"))
        gt = load_bgr(gt_path(case))
        assert lq.shape == gt.shape, (lq.shape, gt.shape)
        if tag == "crop":
            lq, gt = center_crop(lq), center_crop(gt)
        t0 = time.time()
        out = f(lq)
        assert out.shape == lq.shape and out.dtype == np.uint8
        r = dict(case=case, shape=lq.shape,
                 psnr_base=psnr(lq, gt), psnr_out=psnr(out, gt),
                 ssim_base=ssim(lq, gt), ssim_out=ssim(out, gt),
                 lpips_base=lpips_d(lq, gt), lpips_out=lpips_d(out, gt),
                 changed_px=float((out != lq).any(axis=2).mean()),
                 maxabs=int(np.abs(out.astype(int) - lq.astype(int)).max()))
        if do_niqe:
            r["niqe_base"] = niqe(lq)
            r["niqe_out"] = niqe(out)
        r["t"] = time.time() - t0
        rows.append(r)
        print(f"[{tag}] {case} done in {r['t']:.1f}s", flush=True)
    return rows


def report(tag, rows):
    print(f"\n===== {tag} =====")
    hdr = f"{'case':6} {'dPSNR':>10} {'dSSIM':>11} {'dLPIPS':>11} {'dNIQE':>10} {'chg%':>7} {'maxΔ':>5}"
    print(hdr)
    acc = {k: [] for k in ["p", "s", "l", "n"]}
    for r in rows:
        dp = r["psnr_out"] - r["psnr_base"]
        ds = r["ssim_out"] - r["ssim_base"]
        dl = r["lpips_out"] - r["lpips_base"]
        dn = (r["niqe_base"] - r["niqe_out"]) if "niqe_base" in r else float("nan")
        acc["p"].append(dp); acc["s"].append(ds); acc["l"].append(dl); acc["n"].append(dn)
        print(f"{r['case']:6} {dp:+10.4f} {ds:+11.6f} {dl:+11.6f} {dn:+10.4f} "
              f"{r['changed_px']*100:6.2f}% {r['maxabs']:5d}")
    print(f"{'MEAN':6} {np.mean(acc['p']):+10.4f} {np.mean(acc['s']):+11.6f} "
          f"{np.mean(acc['l']):+11.6f} {np.mean(acc['n']):+10.4f}")
    print("\n绝对值：")
    for r in rows:
        n = f" niqe {r.get('niqe_base', float('nan')):.4f}->{r.get('niqe_out', float('nan')):.4f}"
        print(f"  {r['case']} {r['shape']}  psnr {r['psnr_base']:.4f}->{r['psnr_out']:.4f}"
              f"  ssim {r['ssim_base']:.6f}->{r['ssim_out']:.6f}"
              f"  lpips {r['lpips_base']:.6f}->{r['lpips_out']:.6f}{n}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("crop", "both"):
        report("CENTER-1024-CROP", evaluate("crop"))
    if which in ("full", "both"):
        report("FULL-IMAGE", evaluate("full"))
