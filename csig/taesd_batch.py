"""用指定的 VAE 组合把整个测试集跑一遍，并出一张可视对比图。

    python csig/taesd_batch.py --variant SD/TAESD --input-dir <测试集> --output-dir <出图>
    python csig/taesd_batch.py --contact-sheet a=dirA,b=dirB,c=dirC --out sheet.jpg

对比图用等比裁的中心/边角 crop 拼，而不是缩略整图——缩略会把要看的细节抹掉，
而这几个变体的差异恰恰只在细节（FR 几乎相同，差异在 MANIQA 那一侧）。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_variant(a):
    from taesd_exp import load_runner, build_variants, patch
    m, r = load_runner(a.model_dir)
    import csig_ops as ops
    dt = r.weight_dtype
    tae, variants = build_variants(r, ops, dt)
    r._orig_encode_moments = r.vae.encode_moments
    r._orig_sample_latent = r.vae.sample_latent
    enc, dec = variants[a.variant]
    patch(r, enc, dec, dt, taesd_enc=a.variant.startswith("TAESD"),
          tae=tae, A=0.16668, B=0.01702)

    os.makedirs(a.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.input_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"变体 {a.variant}: {len(files)} 张 -> {a.output_dir}")
    t0 = time.time()
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(a.input_dir, f))
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255.
        with torch.no_grad():
            o = r.enhance(x * 2 - 1)
        o = (((o.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255)
        cv2.imwrite(os.path.join(a.output_dir, f), o.round().astype(np.uint8)[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1f s/张  剩余约 %.1f 分钟"
                  % (i + 1, len(files), el / (i + 1), el / (i + 1) * (len(files) - i - 1) / 60), flush=True)
    print("完成，共 %.1f 分钟" % ((time.time() - t0) / 60))


def contact_sheet(a):
    """每张图取一个 crop，横向拼各变体，纵向堆若干张。"""
    groups = []
    for kv in a.contact_sheet.split(","):
        name, d = kv.split("=", 1)
        groups.append((name, d))
    files = sorted(os.listdir(groups[0][1]))
    picks = a.cases.split(",") if a.cases else files[:a.rows]
    S = a.crop
    rows = []
    for fn in picks:
        fn = fn if fn in files else fn + ".jpg"
        tiles = []
        for name, d in groups:
            img = cv2.imread(os.path.join(d, fn))
            if img is None:
                continue
            H, W = img.shape[:2]
            y, x = (H - S) // 2 + a.dy, (W - S) // 2 + a.dx
            y, x = max(0, min(y, H - S)), max(0, min(x, W - S))
            t = img[y:y + S, x:x + S].copy()
            cv2.putText(t, f"{name} {fn}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(t, f"{name} {fn}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        if tiles:
            rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("对比图: %s  %dx%d  %.1f MB" % (a.out, sheet.shape[1], sheet.shape[0],
                                          os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["SD/SD", "SD/TAESD", "TAESD/SD", "TAESD/TAESD"])
    ap.add_argument("--model-dir", default="model_dir")
    ap.add_argument("--input-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--contact-sheet", help="name=dir,name=dir,...")
    ap.add_argument("--cases", help="逗号分隔的文件名，默认取前 --rows 张")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--crop", type=int, default=440)
    ap.add_argument("--dx", type=int, default=0)
    ap.add_argument("--dy", type=int, default=0)
    ap.add_argument("--out", default="sheet.jpg")
    a = ap.parse_args()
    (contact_sheet if a.contact_sheet else run_variant)(a)
