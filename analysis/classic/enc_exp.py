"""算子族 D：输出编码参数扫描。对 LQ 原图做不同 JPEG 编码后与 GT 比。"""
import io
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_lib import psnr, ssim, lpips_score, niqe

VAL = "赛题二/验证集"
CASES = ["case1", "case2", "case3"]
GT = {"case1": "case1_gt.png", "case2": "case2_gt.png", "case3": "case3_gt.jpg"}


def load_bgr(path):
    """按真实格式解码（扩展名不可信），返回 uint8 BGR。"""
    im = Image.open(path).convert("RGB")
    return np.asarray(im)[:, :, ::-1].copy()


def cv_enc(bgr, q, sampling=None, optimize=False, progressive=False):
    p = [cv2.IMWRITE_JPEG_QUALITY, q]
    if sampling is not None:
        p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling]
    if optimize:
        p += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    if progressive:
        p += [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
    ok, buf = cv2.imencode(".jpg", bgr, p)
    assert ok
    return buf.tobytes()


def pil_enc(bgr, **kw):
    im = Image.fromarray(bgr[:, :, ::-1])
    b = io.BytesIO()
    im.save(b, format="JPEG", **kw)
    return b.getvalue()


def dec(buf):
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))[:, :, ::-1].copy()


S444 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
S420 = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420


def variants(bgr, src_path):
    """返回 {name: jpeg_bytes}。src_path 用来抄源文件的量化表/采样。"""
    src = Image.open(src_path)
    qt = src.quantization
    ss = JpegImagePlugin.get_sampling(src)
    v = {}
    for q in (85, 92, 95, 98, 100):
        v[f"cv_q{q}_420"] = cv_enc(bgr, q, S420)
        v[f"cv_q{q}_444"] = cv_enc(bgr, q, S444)
    v["cv_q98_420_opt"] = cv_enc(bgr, 98, S420, optimize=True)
    v["cv_q100_444_opt"] = cv_enc(bgr, 100, S444, optimize=True)
    v["pil_srcqt_srcss"] = pil_enc(bgr, qtables=qt, subsampling=ss)
    v["pil_srcqt_444"] = pil_enc(bgr, qtables=qt, subsampling=0)
    v["pil_q100_444"] = pil_enc(bgr, quality=100, subsampling=0)
    v["pil_q95_420"] = pil_enc(bgr, quality=95, subsampling=2)
    return v


def main():
    full = "--full" in sys.argv
    rows = []
    for c in CASES:
        lq_path = os.path.join(VAL, f"{c}_lq.jpg")
        gt = load_bgr(os.path.join(VAL, GT[c]))
        lq = load_bgr(lq_path)
        assert gt.shape == lq.shape, (gt.shape, lq.shape)
        if not full:
            h, w = gt.shape[:2]
            y, x = h // 2 - 512, w // 2 - 512
            gt, lq = gt[y:y + 1024, x:x + 1024], lq[y:y + 1024, x:x + 1024]
        # 基线：直接复制原始字节 == 评测方解码到的就是 lq 像素
        base = dict(psnr=psnr(gt, lq), ssim=ssim(gt, lq),
                    lpips=lpips_score(gt, lq), niqe=niqe(lq),
                    size=os.path.getsize(lq_path), rt=float("inf"))
        rows.append((c, "bytecopy_LQ", base))
        print(f"[{c}] bytecopy psnr={base['psnr']:.4f} ssim={base['ssim']:.5f} "
              f"lpips={base['lpips']:.5f} niqe={base['niqe']:.4f}", flush=True)
        for name, buf in variants(lq, lq_path).items():
            d = dec(buf)
            m = dict(psnr=psnr(gt, d), ssim=ssim(gt, d), lpips=lpips_score(gt, d),
                     niqe=niqe(d), size=len(buf), rt=psnr(lq, d))
            rows.append((c, name, m))
            print(f"[{c}] {name:22s} psnr={m['psnr']:.4f} ssim={m['ssim']:.5f} "
                  f"lpips={m['lpips']:.5f} niqe={m['niqe']:.4f} rt={m['rt']:.2f} "
                  f"size={m['size']/1e6:.2f}MB", flush=True)
    np.save("csig_bench/enc_exp_full.npy" if full else "csig_bench/enc_exp_crop.npy",
            np.array(rows, dtype=object), allow_pickle=True)

    # 汇总
    names = []
    for _, n, _ in rows:
        if n not in names:
            names.append(n)
    b = {n: {} for n in names}
    for cse, n, m in rows:
        b[n][cse] = m
    print("\n=== 三图平均，Δ 相对 bytecopy_LQ ===")
    ref = {k: np.mean([b["bytecopy_LQ"][c][k] for c in CASES]) for k in ("psnr", "ssim", "lpips", "niqe")}
    print(f"bytecopy_LQ 绝对值: psnr={ref['psnr']:.4f} ssim={ref['ssim']:.5f} "
          f"lpips={ref['lpips']:.5f} niqe={ref['niqe']:.4f}")
    for n in names:
        a = {k: np.mean([b[n][c][k] for c in CASES]) for k in ("psnr", "ssim", "lpips", "niqe")}
        sz = np.mean([b[n][c]["size"] for c in CASES])
        rt = np.mean([b[n][c]["rt"] for c in CASES])
        print(f"{n:22s} dPSNR={a['psnr']-ref['psnr']:+.4f} dSSIM={a['ssim']-ref['ssim']:+.6f} "
              f"dLPIPS={a['lpips']-ref['lpips']:+.6f} dNIQE={a['niqe']-ref['niqe']:+.4f} "
              f"rt={rt:.2f} size={sz/1e6:.2f}MB")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n耗时 {time.time()-t:.1f}s")
