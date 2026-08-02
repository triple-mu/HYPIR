"""零训练可行性探针：预训练的 Moebius（inpainting 模型）能不能直接做画质增强？

Moebius 的训练目标是「把 mask 区域填成与周围一致的背景」，与「保内容提细节」不是一回事。
三种极端配置的预期：
    mask≡0, masked=LQ  -> 训练目标退化为恒等重建，输出 ≈ 输入
    mask≡1, masked=0   -> 纯生成，内容全变，TOPIQ-FR 崩
    mask≡1, masked=LQ  -> OOD，最有希望的一条
本脚本在验证集上把这条轴扫开，用 TOPIQ-FR + MANIQA 定量判断。

判据（FINDINGS §3 的换算）：p_rel = p(out) / p(lq)，LQ 基线恒为 1.0，HYPIR 是 2.65。
    >= 1.5   强正收益，零训练路线可行
    1.1-1.5  弱正收益，移植继续并同时启动微调
    <  1.1   零训练判死
额外的必过门：TOPIQ-FR(out, gt) > TOPIQ-FR(lq, gt)。若 MANIQA 涨而 TOPIQ-FR 跌，
说明只是「变好看但离 GT 更远」——正是 USM3.5 掉 12.88 分的模式，判负。

用法：
    python tools/probe_zero_shot.py                       # 阶段 A
    python tools/probe_zero_shot.py --stage b --steps 1 2 4 8
"""

import argparse
import csv
import itertools
import os
import sys

import cv2
import numpy as np
import torch

from torch.nn.attention import SDPBackend, sdpa_kernel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "model_dir"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import iqa  # noqa: E402
import runner as R  # noqa: E402

VAL_DIR = os.path.join(os.path.dirname(REPO), "赛题二", "验证集")
# 每张 4K 图取 4 个固定坐标的 512 crop，覆盖建筑纹理 / 植被 / 文字 / 平坦区
CROPS = [(600, 600), (1800, 1200), (2600, 900), (1200, 2000)]
ALPHAS = R.make_alphas_cumprod()


def load_pair(case: str):
    lq = cv2.imread(os.path.join(VAL_DIR, "%s_lq.jpg" % case))
    gt = None
    for ext in ("png", "jpg"):
        p = os.path.join(VAL_DIR, "%s_gt.%s" % (case, ext))
        if os.path.exists(p):
            gt = cv2.imread(p)
            break
    assert lq is not None and gt is not None and lq.shape == gt.shape, case
    return lq, gt


def to_tensor(bgr: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(bgr).to(device).flip(-1).permute(2, 0, 1)[None]
    return t.float().div_(127.5).sub_(1.0)


def build_crops(device):
    """返回 [(name, lq_tensor, gt_tensor)]，各 [1,3,512,512] in [-1,1]。"""
    out = []
    for case in ("case1", "case2", "case3"):
        lq, gt = load_pair(case)
        h, w = lq.shape[:2]
        for k, (y, x) in enumerate(CROPS):
            y, x = min(y, h - 512), min(x, w - 512)
            out.append(("%s_c%d" % (case, k),
                        to_tensor(lq[y:y + 512, x:x + 512], device),
                        to_tensor(gt[y:y + 512, x:x + 512], device)))
    return out


@torch.no_grad()
def infer_cfg(r, x, mask_v, masked_scale, t_start, steps):
    """按给定配置跑一次单 tile 推理。不改 runner.py，去噪循环在这里就地展开。"""
    r._setup_schedule(ALPHAS, {"t_start": t_start, "steps": steps})
    mask = torch.full_like(r._mask, mask_v)

    ref = x.to(r.device, torch.float32)
    xin = ref.to(r.weight_dtype).contiguous(memory_format=r._vae_mf)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        z_lq = R.sample_latent(r.vae.encode_moments(xin),
                               r._enc_noise) * R.VAE_SCALING_FACTOR
        z_lq = z_lq.to(r.weight_dtype).contiguous(memory_format=r._unet_mf)
        z_masked = z_lq * masked_scale
        z = r._a_start ** 0.5 * z_lq + (1.0 - r._a_start) ** 0.5 * r._init_noise
        for i, (a_t, a_prev) in enumerate(r._schedule):
            eps = r.unet(torch.cat([z, mask, z_masked], dim=1), r._temb_table[i])
            x0 = (z - (1.0 - a_t) ** 0.5 * eps) / a_t ** 0.5
            z = a_prev ** 0.5 * x0 + (1.0 - a_prev) ** 0.5 * eps
        z = (z / R.VAE_SCALING_FACTOR).to(r.weight_dtype).contiguous(
            memory_format=r._vae_mf)
        y = r.vae.decode(z).float()
    return R.wavelet_reconstruction(y.contiguous(), ref).clamp(-1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a", choices=["a", "b"])
    ap.add_argument("--mask", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--masked-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--t-start", type=int, nargs="+", default=[100, 150, 250, 400, 600, 900])
    ap.add_argument("--steps", type=int, nargs="+", default=[4])
    ap.add_argument("--out", default=os.path.join(REPO, "tools", "probe_results.csv"))
    args = ap.parse_args()

    r = R.Runner(os.path.join(REPO, "model_dir"))
    crops = build_crops(r.device)
    print("%d 个 crop" % len(crops))

    # LQ 基线
    base = {}
    for name, lq, gt in crops:
        base[name] = iqa.score(lq, gt)
    b_fr = np.mean([v[0] for v in base.values()])
    b_nr = np.mean([v[1] for v in base.values()])
    b_p = np.mean([v[2] for v in base.values()])
    print("LQ 基线: TOPIQ-FR %.4f  MANIQA %.4f  proxy %.4f\n" % (b_fr, b_nr, b_p))

    grid = list(itertools.product(args.mask, args.masked_scale, args.t_start, args.steps))
    rows = []
    print("%-6s %-7s %-8s %-6s | %-9s %-8s %-8s %-7s" %
          ("mask", "mscale", "t_start", "steps", "TOPIQ-FR", "MANIQA", "proxy", "p_rel"))
    print("-" * 74)
    for mask_v, mscale, t_start, steps in grid:
        frs, nrs, ps = [], [], []
        for name, lq, gt in crops:
            out = infer_cfg(r, lq, mask_v, mscale, t_start, steps)
            f, n, p = iqa.score(out, gt)
            frs.append(f); nrs.append(n); ps.append(p)
        f, n, p = np.mean(frs), np.mean(nrs), np.mean(ps)
        p_rel = p / b_p
        rows.append(dict(mask=mask_v, masked_scale=mscale, t_start=t_start, steps=steps,
                         topiq_fr=f, maniqa=n, proxy=p, p_rel=p_rel,
                         fr_gain=f - b_fr))
        flag = ""
        if f > b_fr:
            flag = "  <-- FR 过门"
        print("%-6.2f %-7.2f %-8d %-6d | %-9.4f %-8.4f %-8.4f %-7.4f%s" %
              (mask_v, mscale, t_start, steps, f, n, p, p_rel, flag))

    rows.sort(key=lambda d: -d["p_rel"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n写入 %s" % args.out)
    print("\nTop-5 按 p_rel:")
    for d in rows[:5]:
        print("  mask=%.2f mscale=%.2f t=%d steps=%d -> p_rel %.4f  (FR %.4f, 基线 %.4f, %s)" %
              (d["mask"], d["masked_scale"], d["t_start"], d["steps"], d["p_rel"],
               d["topiq_fr"], b_fr, "FR 过门" if d["fr_gain"] > 0 else "FR 未过门"))


if __name__ == "__main__":
    main()
