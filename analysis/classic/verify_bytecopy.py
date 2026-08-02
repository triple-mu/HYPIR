"""独立复现 bytecopy_mpotrunc 候选算子的实测指标。"""
import os
import sys

import cv2
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from basicsr.metrics.niqe import calculate_niqe

VAL = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集'
CASES = [('case1_lq.jpg', 'case1_gt.png'),
         ('case2_lq.jpg', 'case2_gt.png'),
         ('case3_lq.jpg', 'case3_gt.jpg')]


# ---- 候选算子（原样抄自提案）----
def f(bgr):
    return bgr


def save_bytecopy(src_path, dst_path):
    raw = open(src_path, "rb").read()
    i = raw.find(b"\xff\xd9\xff\xd8")
    open(dst_path, "wb").write(raw[:i + 2] if i > 0 else raw)
# ---------------------------------


def imread(p):
    return cv2.cvtColor(np.array(Image.open(p).convert('RGB')), cv2.COLOR_RGB2BGR)


def center_crop(img, s=1024):
    h, w = img.shape[:2]
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    return ssim_fn(a, b, channel_axis=2, data_range=255)


_ln = None


def lpips_d(a, b, dev='cuda' if torch.cuda.is_available() else 'cpu'):
    global _ln
    if _ln is None:
        _ln = lpips.LPIPS(net='alex').to(dev).eval()
    def t(x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        return float(_ln(t(a), t(b)).item())


def niqe(img):
    return float(calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y'))


def run(tag, cropper):
    rows = []
    for lq_n, gt_n in CASES:
        lq = imread(os.path.join(VAL, lq_n))
        gt = imread(os.path.join(VAL, gt_n))
        if cropper:
            lq, gt = cropper(lq), cropper(gt)
        out = f(lq.copy())
        assert out.shape == lq.shape
        bit_ident = np.array_equal(out, lq)
        r = dict(case=lq_n[:5], bit_identical=bit_ident,
                 psnr_lq=psnr(lq, gt), psnr_out=psnr(out, gt),
                 ssim_lq=ssim(lq, gt), ssim_out=ssim(out, gt),
                 lpips_lq=lpips_d(lq, gt), lpips_out=lpips_d(out, gt),
                 niqe_lq=niqe(lq), niqe_out=niqe(out))
        rows.append(r)
        print(f"[{tag}] {r['case']} bit_identical={bit_ident} "
              f"PSNR {r['psnr_lq']:.4f}->{r['psnr_out']:.4f} (d={r['psnr_out']-r['psnr_lq']:+.6f})  "
              f"SSIM {r['ssim_lq']:.6f}->{r['ssim_out']:.6f} (d={r['ssim_out']-r['ssim_lq']:+.8f})  "
              f"LPIPS {r['lpips_lq']:.6f}->{r['lpips_out']:.6f} (d={r['lpips_out']-r['lpips_lq']:+.8f})  "
              f"NIQE {r['niqe_lq']:.4f}->{r['niqe_out']:.4f} (d={r['niqe_out']-r['niqe_lq']:+.6f})",
              flush=True)
    m = lambda k1, k2: np.mean([r[k2] - r[k1] for r in rows])
    print(f"[{tag}] MEAN dPSNR={m('psnr_lq','psnr_out'):+.6f} dSSIM={m('ssim_lq','ssim_out'):+.8f} "
          f"dLPIPS={m('lpips_lq','lpips_out'):+.8f} dNIQE={m('niqe_lq','niqe_out'):+.6f}", flush=True)
    return rows


if __name__ == '__main__':
    run('crop1024', center_crop)
    run('full', None)
