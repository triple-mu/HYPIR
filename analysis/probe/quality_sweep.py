"""画质旋钮的本地扫描：改一个 config 项，量 TOPIQ-FR + MANIQA，估算感知分 p。

为什么是这两个指标：三组已知 p 的图（模型 2.6549 / LQ 1.0000 / USM3.5 0.8199）里，
只有 TOPIQ-FR（有参考）和 MANIQA（无参考）能复现实测排序；LPIPS、DISTS、NIQE、MUSIQ、
TOPIQ-NR、NIMA、BRISQUE、HyperIQA 全都把过锐化的 USM3.5 排在 LQ 之上，与实测矛盾。
见 FINDINGS 第四节。

拟合出的映射（3 方程 3 未知量，恰定、未经验证，只用来看相对高低）：
    p ≈ 13.674 * TOPIQ-FR + 4.477 * MANIQA - 5.731

在验证集的 3 对 LQ/GT 上跑（测试集没有 GT，无法算有参考指标），切成 512 crop 取统计量。

用法：
    python experiments/quality_sweep.py --sweep model_t --values 100,150,200,250,300
    python experiments/quality_sweep.py --sweep prompt --values-file prompts.txt
    python experiments/quality_sweep.py --baseline          # 只量 LQ 自身作锚点
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = REPO_ROOT / "赛题二" / "验证集"
CROP = 512
P_COEF = (13.674, 4.477, -5.731)  # (TOPIQ-FR, MANIQA, 截距)


def load_pairs():
    """返回 [(名字, lq_path, gt_path)]，验证集是 caseN_lq.jpg / caseN_gt.{png,jpg}。"""
    out = []
    for lq in sorted(VAL_DIR.glob("*_lq.*")):
        stem = lq.name.rsplit("_lq", 1)[0]
        gt = next((p for p in VAL_DIR.glob(f"{stem}_gt.*")), None)
        assert gt is not None, f"缺少 {stem} 的 GT"
        out.append((stem, lq, gt))
    return out


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def to_tensor(bgr, device):
    t = torch.from_numpy(bgr).to(device).flip(-1).permute(2, 0, 1)[None].float()
    return t.div_(127.5).sub_(1.0)


def crops(bgr, stride):
    """按 stride 取不重叠的 CROP x CROP 块（stride>CROP 时相当于抽样）。"""
    h, w = bgr.shape[:2]
    for i in range(0, h - CROP + 1, stride):
        for j in range(0, w - CROP + 1, stride):
            yield bgr[i:i + CROP, j:j + CROP]


def bgr_to_batch(cs, device):
    a = np.stack([c[:, :, ::-1] for c in cs])          # (N,H,W,3) RGB
    return torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float().div_(255.).to(device)


def score_images(enh_bgr, gt_bgr, metrics, device, stride, batch=4):
    """对齐切块后逐批算指标，返回 {指标名: 均值}。"""
    ec = list(crops(enh_bgr, stride))
    gc = list(crops(gt_bgr, stride))
    assert len(ec) == len(gc) and ec, "切块数不匹配"
    acc = {k: [] for k in metrics}
    for s in range(0, len(ec), batch):
        eb = bgr_to_batch(ec[s:s + batch], device)
        gb = bgr_to_batch(gc[s:s + batch], device)
        with torch.no_grad():
            for name, (m, full_ref) in metrics.items():
                v = m(eb, gb) if full_ref else m(eb)
                acc[name] += [float(x) for x in v.flatten()]
        del eb, gb
        torch.cuda.empty_cache()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="画质旋钮扫描")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--sweep", type=str, default="",
                    help="要扫的 config 键，逗号分隔表示联动（model_t,coeff_t 必须一起动）")
    ap.add_argument("--values", type=str, default="", help="逗号分隔的取值")
    ap.add_argument("--stride", type=int, default=1024, help="切块步长，越小样本越多越慢")
    ap.add_argument("--baseline", action="store_true", help="只量 LQ 自身，作 p=1 的锚点")
    args = ap.parse_args()

    import pyiqa
    device = "cuda"
    metrics = {"topiq_fr": (pyiqa.create_metric("topiq_fr", device=device), True),
               "maniqa": (pyiqa.create_metric("maniqa", device=device), False)}
    pairs = load_pairs()
    n_crop = len(list(crops(imread(pairs[0][1]), args.stride))) * len(pairs)
    print(f"验证集 {len(pairs)} 对，stride={args.stride} -> 约 {n_crop} 个 {CROP}x{CROP} crop\n")

    def report(tag, per_img):
        agg = {k: float(np.mean([d[k] for d in per_img])) for k in per_img[0]}
        p = P_COEF[0] * agg["topiq_fr"] + P_COEF[1] * agg["maniqa"] + P_COEF[2]
        print(f"  {tag:22s} TOPIQ-FR {agg['topiq_fr']:.4f}  MANIQA {agg['maniqa']:.4f}"
              f"  -> p≈{p:.4f}")
        return p

    if args.baseline:
        per = []
        for name, lq, gt in pairs:
            per.append(score_images(imread(lq), imread(gt), metrics, device, args.stride))
        report("LQ 原图（锚点）", per)
        return

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import runner as R

    keys = [k.strip() for k in args.sweep.split(",") if k.strip()]
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    assert keys and values, "需要 --sweep 与 --values"
    base_cfg = R._load_config
    results = {}
    for v in values:
        val = int(v) if v.isdigit() else v
        R._load_config = lambda d, _v=val: {**base_cfg(d), **{k: _v for k in keys}}
        t0 = time.time()
        r = R.Runner(args.model_dir)
        per = []
        for name, lq, gt in pairs:
            with torch.no_grad():
                out = r.enhance(to_tensor(imread(lq), r.device))
            enh = ((out[0] + 1) * 127.5).round_().clamp_(0, 255).to(torch.uint8)
            enh = enh.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
            per.append(score_images(enh, imread(gt), metrics, device, args.stride))
            del out
            torch.cuda.empty_cache()
        results[val] = report(f"{'+'.join(keys)}={val}", per)
        print(f"    （{time.time() - t0:.0f}s）")
        del r
        torch.cuda.empty_cache()
    R._load_config = base_cfg

    best = max(results, key=results.get)
    cur = results.get(200 if "model_t" in keys else None)
    print(f"\n最优 {args.sweep}={best}  p≈{results[best]:.4f}"
          + (f"，相对当前值 {results[best] / cur - 1:+.2%}" if cur else ""))


if __name__ == "__main__":
    main()
