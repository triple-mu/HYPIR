"""真实退化闸门：在赛题验证集的真实 LQ/GT 对上给权重打分。

**这道闸门今天要是存在，就不会有 20.76 那次提交。**

P0 评估集是拿我们自己的合成退化造的，它只能回答「模型有多会求逆我这个算子」。
P3 在 P0 上从 19.0% 涨到 45.6%，线上却从 44.04 掉到 20.76——因为真实退化是
另一个算子。判决实验见 csig/domain_split.py。

样本只有 3 对，太少，做不了训练判据；但**足够当否决闸门**——它一眼就分出了
基线 13.5% 和微调 -0.1%。

    python csig/real_gate.py <weight_path> [--baseline]

任何权重打不过基线的 13.5%，就不许进提交包。
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
import iqa
from HYPIR.enhancer.sd2 import SD2Enhancer

from prompt import PROMPT as TRAIN_PROMPT

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
VAL = os.path.join(CSIG, "data/csig_bench/赛题二/验证集")
# 默认用训练时的 prompt（csig/prompt.py，当前是空串）。写死赛题长句会造成 train/infer 失配 ——
# 本轮全部权重都是在空串条件下训的。--prompt legacy 可切回那句长 prompt 做对照。
LEGACY_PROMPT = ("A high-resolution photograph with fine natural detail: modern Chinese city "
                 "high-rise buildings with glass curtain walls and tiled facades, construction "
                 "cranes and shop signage, distant hills, and close-up green foliage with flowers.")
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
BASELINE_RETAINED = 0.135   # 未微调 HYPIR 在这 3 对上的实测值


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight_path")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompt", default="train", choices=["train", "legacy"],
                    help="train=csig/prompt.py 的训练用 prompt（默认）；legacy=旧的赛题长句")
    # 默认 sd 是为了与历史数字可比 —— 注意这是个口径陷阱：HYPIR/enhancer/base.py 写死
    # AutoencoderKL，所以此前记录的全部闸门数字（基线 13.5%、drt@3000 18.9%）都是
    # SD-VAE 推理出来的，而提交包 runner.py 用的是 TAESD。换口径后基线必须重跑。
    ap.add_argument("--vae", default="sd", choices=["sd", "taesd"])
    ap.add_argument("--tae-weight", default=None,
                    help="训练产出的编码器权重，与同目录 LoRA 成对取用（raw 配 raw，EMA 配 EMA）")
    a = ap.parse_args()
    prompt = TRAIN_PROMPT if a.prompt == "train" else LEGACY_PROMPT
    print("prompt=%s %r" % (a.prompt, prompt[:60]))

    pairs = []
    for lqp in sorted(glob.glob(os.path.join(VAL, "*_lq.jpg"))):
        b = os.path.basename(lqp).replace("_lq.jpg", "")
        g = glob.glob(os.path.join(VAL, b + "_gt.*"))
        if g:
            pairs.append((b, lqp, g[0]))
    assert pairs, f"找不到验证对: {VAL}"

    en = SD2Enhancer(base_model_path=os.path.join(CSIG, "weights"),
                     weight_path=a.weight_path, lora_modules=LORA_MODULES,
                     lora_rank=256, model_t=200, coeff_t=200, device="cuda")
    en.init_models()
    if a.vae == "taesd":
        from HYPIR.utils.taesd import build_taesd
        en.vae = build_taesd(en.weight_dtype, en.device, enc_weight=a.tae_weight)
    print("vae=%s tae_weight=%s" % (a.vae, a.tae_weight))

    def load(p):
        return torch.from_numpy(cv2.imread(p)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255.

    rows = []
    for base, lqp, gtp in pairs:
        lq, gt = load(lqp), load(gtp)
        with torch.no_grad():
            out = en.enhance(lq, prompt=prompt, upscale=1, patch_size=512,
                             stride=256, return_type="pt").clamp(0, 1)
        fo, no, po = iqa.score(out * 2 - 1, gt * 2 - 1)
        _, _, pl = iqa.score(lq * 2 - 1, gt * 2 - 1)
        _, _, pg = iqa.score(gt * 2 - 1, gt * 2 - 1)
        rows.append((base, fo, no, po, pl, pg))
        print("  %-8s FR %.4f  NR %.4f  p_out %7.3f  (p_lq %.3f  p_gt %.3f)"
              % (base, fo, no, po, pl, pg))

    r = np.array([x[1:] for x in rows], dtype=float)
    ret = (r[:, 2].mean() - r[:, 3].mean()) / max(r[:, 4].mean() - r[:, 3].mean(), 1e-9)
    print("\n  %-8s FR %.4f  NR %.4f  p_out %7.3f            保留增益 %.1f%%"
          % ("合计", r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), 100 * ret))
    print("  基线（未微调）在同样 3 对上是 %.1f%%" % (100 * BASELINE_RETAINED))
    ok = ret > BASELINE_RETAINED
    print("\n  闸门: %s" % ("通过 —— 可以考虑打包" if ok
                            else "**不通过** —— 打不过基线，不许进提交包"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
