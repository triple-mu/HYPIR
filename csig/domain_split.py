"""把「内容域」和「退化」两个变量拆开，定位微调失效的原因。

事实：
    P0（训练域内容 + 我的合成退化）     基线 19.0%  ->  微调 45.6%   大幅提升
    赛题验证集（赛题内容 + 真实退化）   基线 13.4%  ->  微调 -0.1%   彻底失效
    线上                                44.04      ->  20.76

两者差了两个变量。退化那一项已被 degrade_ablate.py 排除——同图配对下
合成退化与真实退化的 ΔFR +0.012 / ΔNR +0.002，匹配得很好。

本脚本补上缺的那一格：**赛题内容 + 合成退化**。

    A 高 -> 退化不是问题，是真实退化里有合成没复现的成分
    A 低 -> 内容域是问题，模型只在训练图源上有效

    python csig/domain_split.py
"""
import os
import sys
import glob

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import iqa
from degrade_ablate import degrade
from HYPIR.dataset.csig import CSIGBatchTransform
from HYPIR.enhancer.sd2 import SD2Enhancer

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")
PROMPT = ("A high-resolution photograph with fine natural detail: modern Chinese city "
          "high-rise buildings with glass curtain walls and tiled facades, construction "
          "cranes and shop signage, distant hills, and close-up green foliage with flowers.")
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
WEIGHTS = [("基线(未微调)", f"{CSIG}/weights/HYPIR_sd2.pth"),
           ("P2@7000(已提交)", f"{CSIG}/out/best/p2_7000_ema.pth")]


def main():
    dev = torch.device("cuda")
    pairs = []
    for lqp in sorted(glob.glob(os.path.join(VAL, "*_lq.jpg"))):
        b = os.path.basename(lqp).replace("_lq.jpg", "")
        g = glob.glob(os.path.join(VAL, b + "_gt.*"))
        if g:
            pairs.append((b, lqp, g[0]))

    # 整图，与之前的真实退化测量口径一致
    GT, LQ_real, LQ_syn = [], [], []
    tf = CSIGBatchTransform()
    for i, (b, lqp, gtp) in enumerate(pairs):
        gt = torch.from_numpy(cv2.imread(gtp)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev) / 255.
        lq = torch.from_numpy(cv2.imread(lqp)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev) / 255.
        GT.append(gt); LQ_real.append(lq)
        LQ_syn.append(degrade(gt, tf, seed=i))

    def report(tag, outs, refs, lqs):
        f = np.mean([iqa.score(o * 2 - 1, r * 2 - 1)[0] for o, r in zip(outs, refs)])
        n = np.mean([iqa.score(o * 2 - 1, r * 2 - 1)[1] for o, r in zip(outs, refs)])
        po = np.mean([iqa.score(o * 2 - 1, r * 2 - 1)[2] for o, r in zip(outs, refs)])
        pl = np.mean([iqa.score(l * 2 - 1, r * 2 - 1)[2] for l, r in zip(lqs, refs)])
        pg = np.mean([iqa.score(r * 2 - 1, r * 2 - 1)[2] for r in refs])
        ret = (po - pl) / max(pg - pl, 1e-9)
        print("  %-22s FR %.4f  NR %.4f  p_out %6.3f  (p_lq %.3f)  保留增益 %6.1f%%"
              % (tag, f, n, po, pl, 100 * ret))
        return ret

    for wtag, wpath in WEIGHTS:
        en = SD2Enhancer(base_model_path=f"{CSIG}/weights", weight_path=wpath,
                         lora_modules=LORA_MODULES, lora_rank=256,
                         model_t=200, coeff_t=200, device="cuda")
        en.init_models()
        print(f"\n=== {wtag} ===")
        for name, lqs in [("赛题内容 + 真实退化", LQ_real), ("赛题内容 + 合成退化", LQ_syn)]:
            outs = []
            for lq in lqs:
                with torch.no_grad():
                    outs.append(en.enhance(lq, prompt=PROMPT, upscale=1, patch_size=512,
                                           stride=256, return_type="pt").clamp(0, 1))
            report(name, outs, GT, lqs)
        del en
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
