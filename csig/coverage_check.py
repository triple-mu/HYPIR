"""检查真实退化是否落进合成退化分布的**支撑集内部**。

v1 的错误是拿均值对齐：合成与真实的 ΔFR +0.012 / ΔNR +0.002，看着完美，
但两者是不同的算子，模型只学会求逆合成那一个。判决实验（domain_split.py）里
同一批赛题图只换退化方式，微调模型从 44.7% 掉到 -0.1%。

所以 v2 的验收标准换成覆盖率：把真实 LQ 的 (FR, NR) 点放进合成样本的散点云里，
看它落在什么分位。目标是**靠近中间**——落在边缘或云外说明模型训练时几乎没见过
这种退化，泛化不过去。

    python csig/coverage_check.py [--n 64]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import iqa  # noqa: E402
from HYPIR.dataset.csig import CSIGBatchTransform  # noqa: E402
from HYPIR.dataset.csig_degradation import degrade_tensor  # noqa: E402

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="每张 GT 抽多少个合成样本")
    a = ap.parse_args()

    dev = torch.device("cuda")
    tf = CSIGBatchTransform()
    S = 512
    rng = np.random.RandomState(0)

    pairs = []
    for lqp in sorted(glob.glob(os.path.join(VAL, "*_lq.jpg"))):
        b = os.path.basename(lqp).replace("_lq.jpg", "")
        g = glob.glob(os.path.join(VAL, b + "_gt.*"))
        if g:
            pairs.append((b, lqp, g[0]))

    real_pts, syn_pts = [], []
    for base, lqp, gtp in pairs:
        gt_full = cv2.imread(gtp)[:, :, ::-1].copy()
        lq_full = cv2.imread(lqp)[:, :, ::-1].copy()
        H, W = gt_full.shape[:2]
        for k in range(a.n):
            y, x = rng.randint(0, H - S), rng.randint(0, W - S)
            gt = torch.from_numpy(gt_full[y:y + S, x:x + S]).permute(2, 0, 1)[None].float().to(dev) / 255.
            if k < 8:   # 真实 LQ 只取前 8 个 crop，够定位就行
                lq = torch.from_numpy(lq_full[y:y + S, x:x + S]).permute(2, 0, 1)[None].float().to(dev) / 255.
                f, n, _ = iqa.score(lq * 2 - 1, gt * 2 - 1)
                real_pts.append((f, n))
            # 合成：走完整的 v2 管线（含重采样支路、噪声、JPEG 抽样）
            syn = _synth(tf, gt)
            f, n, _ = iqa.score(syn * 2 - 1, gt * 2 - 1)
            syn_pts.append((f, n))

    real = np.array(real_pts)
    syn = np.array(syn_pts)
    print(f"真实 LQ 样本 {len(real)} 个，合成样本 {len(syn)} 个\n")
    for i, name in enumerate(["FR", "NR"]):
        r, s = real[:, i], syn[:, i]
        pct = 100.0 * (s < r.mean()).mean()
        print(f"  {name}:  真实均值 {r.mean():.4f} [{r.min():.4f}, {r.max():.4f}]")
        print(f"       合成分布 {s.mean():.4f} [{s.min():.4f}, {s.max():.4f}]")
        print(f"       真实均值落在合成分布的第 {pct:.0f} 百分位"
              f"   {'(靠中间，覆盖良好)' if 15 <= pct <= 85 else '(靠边缘，覆盖不足)'}")
    # 二维覆盖：真实点有多少落在合成点云的凸范围内（用逐维分位近似）
    inside = 0
    for f, n in real:
        pf = (syn[:, 0] < f).mean()
        pn = (syn[:, 1] < n).mean()
        if 0.05 < pf < 0.95 and 0.05 < pn < 0.95:
            inside += 1
    print(f"\n  真实点落在合成分布 5-95 分位区间内的比例: {inside}/{len(real)}")


def _synth(tf, gt):
    """对单张 GT 走训练真源；状态 RNG 让连续调用得到独立样本。"""
    if not hasattr(_synth, "rng"):
        _synth.rng = np.random.default_rng(20260813)
    return degrade_tensor(gt, rng=_synth.rng, degrader=tf.degrader)


if __name__ == "__main__":
    main()
