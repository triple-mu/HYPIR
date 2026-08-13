"""退化配方的分环节归因：哪一步把合成 LQ 的 MANIQA 压垮了。

背景：提交后线上分从 44.04 掉到 20.76。反解出画质 p_rel 从 2.655 掉到 1.252，
真实验证对上微调模型的保留增益是 -0.1%（基线 13.4%）——模型在真实数据上帮倒忙。
根因是合成 LQ 与真实 LQ 的**无参考质量**对不上：

    FR_lq   合成 0.3595   真实 0.3840   （匹配）
    NR_lq   合成 0.2126   真实 0.3015   （合成低 42%）

配方当初只按 FR 标定过，NR 从来没进过标定目标。真实 LQ 只是「糊」，
而合成 LQ 是「又糊又脏」，模型学到的激进策略用在干净输入上就是破坏。

本脚本用**赛题验证集的 GT** 做源、对同一张图施加各变体的退化，再和**同一张图的
真实 LQ** 比——同图配对，不跨数据集。逐环节开关，量各自对 FR/NR 的贡献。

    python csig/degrade_ablate.py [--crops 8]
"""
import argparse
from dataclasses import replace
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import iqa
from HYPIR.dataset.csig import CSIGBatchTransform
from HYPIR.dataset.csig_degradation import DEFAULT_CONFIG, sample_degradation

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")


def degrade(hq, tf, *, lowpass=True, affine=True, jpeg=True, quant=True,
            sigma_scale=1.0, seed=0):
    """按开关施加训练真源；固定 sample_seed 使各变体逐样本可比。"""
    rng = np.random.default_rng(seed)
    config = replace(DEFAULT_CONFIG, near_native_given_weak=0.0)
    metas = [sample_degradation(
        rng, profile="ordinary", config=config, sample_seed=seed + i,
    ) for i in range(hq.shape[0])]
    for meta in metas:
        if not lowpass:
            meta.update(
                operator="none", kernel_family=None, sigma_x=None, sigma_y=None,
                theta=None, pole=None, spatial_variation=0.0, resize_scale=None,
                down_mode=None, down_antialias=None, up_mode=None, subpixel_shift_xy=None,
                edge_aware_strength=0.0, unsharp_amount=0.0,
                fusion_shift_xy=None, fusion_opacity=0.0,
                local_warp_amplitude=0.0,
            )
        elif meta["sigma_x"] is not None:
            meta["sigma_x"] *= sigma_scale
            meta["sigma_y"] *= sigma_scale
        if not affine:
            meta.update(
                tone_mode="identity", tone_gain=[1.0, 1.0, 1.0],
                tone_bias=[0.0, 0.0, 0.0], tone_gamma=1.0,
                highlight_compression=0.0, chroma_blur=0.0,
                chroma_shift_xy=None, saturation=1.0,
            )
    return tf.degrader.apply(hq, metas, apply_jpeg=jpeg, quantize=quant).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=8, help="每对图取几个 512 crop")
    a = ap.parse_args()

    import glob
    pairs = []
    for lqp in sorted(glob.glob(os.path.join(VAL, "*_lq.jpg"))):
        base = os.path.basename(lqp).replace("_lq.jpg", "")
        gtp = glob.glob(os.path.join(VAL, base + "_gt.*"))
        if gtp:
            pairs.append((base, lqp, gtp[0]))
    assert pairs, "找不到验证对"

    dev = torch.device("cuda")
    tf = CSIGBatchTransform()
    S = 512
    rng = np.random.RandomState(0)

    gt_crops, lq_crops = [], []
    for base, lqp, gtp in pairs:
        gt = cv2.imread(gtp)[:, :, ::-1].copy()
        lq = cv2.imread(lqp)[:, :, ::-1].copy()
        assert gt.shape == lq.shape, (base, gt.shape, lq.shape)
        H, W = gt.shape[:2]
        for _ in range(a.crops):
            y, x = rng.randint(0, H - S), rng.randint(0, W - S)
            gt_crops.append(gt[y:y + S, x:x + S])
            lq_crops.append(lq[y:y + S, x:x + S])

    def to_t(arrs):
        return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().to(dev) / 255.0

    GT, LQ = to_t(gt_crops), to_t(lq_crops)
    n = GT.shape[0]

    def score(x, ref):
        fs, ns, ps = [], [], []
        for i in range(x.shape[0]):
            f, nn, p = iqa.score(x[i:i + 1] * 2 - 1, ref[i:i + 1] * 2 - 1)
            fs.append(f)
            ns.append(nn)
            ps.append(p)
        return float(np.mean(fs)), float(np.mean(ns)), float(np.mean(ps))

    print(f"源: 赛题验证集 {len(pairs)} 对 x {a.crops} 个 512 crop = {n} 个样本（同位置配对）\n")
    fr_r, nr_r, p_r = score(LQ, GT)
    print(f"{'变体':<26} {'FR':>8} {'NR':>8} {'p':>8}   {'ΔFR':>7} {'ΔNR':>7}")
    print("-" * 72)
    print(f"{'【真实 LQ】(目标)':<24} {fr_r:8.4f} {nr_r:8.4f} {p_r:8.3f}   {'—':>7} {'—':>7}")

    variants = [
        ("当前配方(全开)",        dict()),
        ("去掉 仿射+clip",        dict(affine=False)),
        ("去掉 JPEG",             dict(jpeg=False)),
        ("去掉 8bit 量化",        dict(quant=False)),
        ("只留低通",              dict(affine=False, jpeg=False, quant=False)),
        ("低通 sigma x0.7",       dict(sigma_scale=0.7)),
        ("低通 sigma x0.5",       dict(sigma_scale=0.5)),
        ("只低通 sigma x0.7",     dict(affine=False, jpeg=False, quant=False, sigma_scale=0.7)),
    ]
    for name, kw in variants:
        out = degrade(GT, tf, seed=0, **kw)
        f, nn, p = score(out, GT)
        print(f"{name:<26} {f:8.4f} {nn:8.4f} {p:8.3f}   {f-fr_r:+7.4f} {nn-nr_r:+7.4f}")

    print("\nΔ 是相对真实 LQ 的偏差，越接近 0 越像真实退化。")
    print("当初只按 FR 标定，所以 ΔFR 小而 ΔNR 大——模型学到的是修复不存在的伪影。")


if __name__ == "__main__":
    main()
