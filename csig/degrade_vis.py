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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from HYPIR.dataset.csig import CSIGBatchTransform  # noqa: E402
from HYPIR.dataset.csig_degradation import degrade_tensor  # noqa: E402


@torch.no_grad()
def degrade(hq, tf, seed):
    """走训练真源并用 sample_seed 完整回放。"""
    out, metas = degrade_tensor(
        hq,
        rng=np.random.default_rng(seed),
        sample_seeds=[seed + i for i in range(hq.shape[0])],
        return_metadata=True,
        degrader=tf.degrader,
    )
    m = metas[0]
    parts = [m["profile"], m["severity"], m["operator"]]
    if m["resize_scale"] is not None:
        parts.append("%.1fx %s->%s" % (m["resize_scale"], m["down_mode"], m["up_mode"]))
    if m["sigma_x"] is not None:
        parts.append("sigma %.2f/%.2f" % (m["sigma_x"], m["sigma_y"]))
    parts.append("tone=%s" % m["tone_mode"])
    parts.append("jpeg=%s" % m["jpeg_qtable_profile"])
    return out, "  ".join(parts)


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
