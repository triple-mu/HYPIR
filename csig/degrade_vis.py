"""把合成 LQ 和真实 LQ 摆在一起看。

之前只量过 FR/NR 两个标量就下结论（v1 那次「均值对齐」正是这么翻车的），
从没目视比过。本脚本对赛题验证集的 GT 施加 v2 退化，与同一张图的真实 LQ 并排，
每张多抽几次以显示分布的宽度——v2 的目标是**覆盖**真实退化而不是等于它，
所以合成样本之间差异大是预期的，关键看真实 LQ 是否落在这些样本的范围内。

    python csig/degrade_vis.py --val <验证集目录> --out <输出目录> [--draws 3] [--crop 1024]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from HYPIR.dataset.csig import (SIGMA_RANGE, ANISO_RANGE, POLE_RANGE, SPATIAL_VAR,
                                AFFINE_A, AFFINE_CHROMA, AFFINE_PIVOT, AFFINE_B_JITTER, JPEG_RANGE, P_AFFINE,
                                P_RESAMPLE, SCALE_RANGE, NOISE_SIGMA, P_CLEAN,
                                CSIGBatchTransform)


@torch.no_grad()
def degrade(hq, tf, seed):
    """完整走一遍 v2 管线，返回 (lq, 本次抽到的参数描述)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    b, _, h, w = hq.shape
    dev = hq.device
    U = lambda lo, hi: (torch.rand(b, generator=g) * (hi - lo) + lo).to(dev)

    info = []
    use_rs = float(torch.rand(1, generator=g)) < P_RESAMPLE
    if use_rs:
        s = float(torch.empty(1).uniform_(*SCALE_RANGE))
        modes = ("bilinear", "bicubic", "area")
        dm = modes[int(torch.randint(3, (1,), generator=g))]
        um = modes[int(torch.randint(2, (1,), generator=g))]
        small = F.interpolate(hq, size=(max(int(h / s), 8), max(int(w / s), 8)), mode=dm,
                              **({} if dm == "area" else {"align_corners": False}))
        out = F.interpolate(small, size=(h, w), mode=um, align_corners=False)
        info.append("重采样 %.1fx %s->%s" % (s, dm, um))
    else:
        sigma = U(*SIGMA_RANGE)
        lo = tf._lowpass(hq, sigma * (1 - SPATIAL_VAR), U(*ANISO_RANGE),
                         U(0.0, float(np.pi)), U(*POLE_RANGE))
        hi = tf._lowpass(hq, sigma * (1 + SPATIAL_VAR), U(*ANISO_RANGE),
                         U(0.0, float(np.pi)), U(*POLE_RANGE))
        m = torch.rand(b, 1, 8, 8, generator=g).to(dev)
        m = F.interpolate(m, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        out = lo * (1 - m) + hi * m
        info.append("频域低通 sigma=%.2f" % float(sigma))
    if float(torch.rand(1, generator=g)) < P_CLEAN:
        out = hq
        info.append("（本次抽中免退化）")
    do = float(torch.rand(1, generator=g)) < P_AFFINE
    if do:
        gg = (torch.rand(b,1,1,1, generator=g)*(AFFINE_A[1]-AFFINE_A[0])+AFFINE_A[0]).to(dev)
        ca = gg + (torch.rand(b,3,1,1, generator=g)*2-1).to(dev)*AFFINE_CHROMA
        cb = (1-ca)*AFFINE_PIVOT + (torch.rand(b,3,1,1, generator=g)*2-1).to(dev)*AFFINE_B_JITTER
        out = (out * ca + cb).clamp(0, 1)
        info.append("色调 a=%.3f" % float(gg))
    else:
        out = out.clamp(0, 1)
    if tf.jpeger is None:
        from HYPIR.dataset.diffjpeg import DiffJPEG
        tf.jpeger = DiffJPEG(differentiable=False).to(dev)
    tf.jpeger.to(out)
    q = float(torch.rand(1, generator=g)) * (JPEG_RANGE[1] - JPEG_RANGE[0]) + JPEG_RANGE[0]
    out = tf.jpeger(out, quality=out.new_full((b,), q))
    info.append("JPEG q%d" % round(q))
    ns = float(torch.rand(1, generator=g)) * (NOISE_SIGMA[1] - NOISE_SIGMA[0]) + NOISE_SIGMA[0]
    if ns > 0.2 / 255:
        out = out + torch.randn(out.shape, generator=g).to(dev) * ns
        info.append("噪声 %.2f/255" % (ns * 255))
    out = torch.clamp((out.clamp(0, 1) * 255.0).round(), 0, 255) / 255.0
    return out, "  ".join(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--iqa", action="store_true", help="同时算 FR/NR（需要 pyiqa）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = torch.device("cuda")
    tf = CSIGBatchTransform()
    score = None
    if a.iqa:
        sys.path.insert(0, HERE)
        import iqa as _iqa
        score = _iqa.score

    import glob
    S = a.crop
    rng = np.random.RandomState(0)
    for lqp in sorted(glob.glob(os.path.join(a.val, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(a.val, base + "_gt.*"))[0]
        gt_f = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_f = cv2.imread(lqp)[:, :, ::-1].copy()
        H, W = gt_f.shape[:2]
        y, x = (H - S) // 2, (W - S) // 2
        gt_c, lq_c = gt_f[y:y + S, x:x + S], lq_f[y:y + S, x:x + S]
        gt = torch.from_numpy(gt_c).permute(2, 0, 1)[None].float().to(dev) / 255.

        tiles, labels = [gt_c, lq_c], ["GT (原图)", "真实 LQ"]
        for k in range(a.draws):
            syn, info = degrade(gt, tf, seed=1000 + k)
            arr = (syn[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            tiles.append(arr)
            labels.append("合成 #%d  %s" % (k + 1, info))

        if score is not None:
            for i, t in enumerate(tiles):
                tt = torch.from_numpy(t).permute(2, 0, 1)[None].float().to(dev) / 255.
                f, n, p = score(tt * 2 - 1, gt * 2 - 1)
                labels[i] += "   FR %.4f  NR %.4f" % (f, n)

        row = []
        for t, lab in zip(tiles, labels):
            t = t[:, :, ::-1].copy()
            cv2.rectangle(t, (0, 0), (S, 34), (0, 0, 0), -1)
            cv2.putText(t, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(t)
        sheet = np.hstack(row)
        p = os.path.join(a.out, "%s_degrade.jpg" % base)
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("%s  %dx%d  %.1f MB" % (p, sheet.shape[1], sheet.shape[0], os.path.getsize(p) / 1e6))
        for lab in labels:
            print("    ", lab)


if __name__ == "__main__":
    main()
