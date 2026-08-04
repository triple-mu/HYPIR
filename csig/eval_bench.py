"""在 DIV2K_V2_val 上给一份 LoRA 权重打分（论文 Table 1 的同一套协议）。

评估集是 StableSR 放出的 DIV2K_V2_val：3000 对 128x128 LQ / 512x512 GT，
退化用的就是 Real-ESRGAN 那一套 —— 与 HYPIR 论文 Table 1 的 DIV2K 一栏、
以及本轮训练用的退化管线是同一个分布。推理参数与官方 test.py 的命令行逐项一致
（upscale 4 / patch_size 512 / stride 256 / seed 231）。

三个 arm：
    A  官方 HYPIR_sd2.pth + SD-VAE   --vae sd     官方上限
    B  官方 HYPIR_sd2.pth + TAESD    --vae taesd  零样本起点
    C  本轮训出的 LoRA   + TAESD     --vae taesd  目标

论文报的 Ours-SD2 / DIV2K 是
PSNR 22.16 / SSIM 0.5877 / LPIPS 0.2318 / NIQE 3.832 / MUSIQ 72.58 / MANIQA 0.5838 / CLIP-IQA 0.7467。
跨实现的绝对值会有偏差（pyiqa 与论文用的实现不一定同源），判据以三个 arm 之间的相对关系为准。

    python csig/eval_bench.py --weight $CSIG/weights/HYPIR_sd2.pth --vae sd --tag a_official
    python csig/eval_bench.py --weight $CSIG/out/taesd_main/snapshots/step-005000/ema_state_dict.pth \
        --vae taesd --tag c_5000_ema --limit 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt import PROMPT
from HYPIR.enhancer.sd2 import SD2Enhancer

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
BENCH = os.path.join(CSIG, "data/bench/StableSR_testsets/DIV2K_V2_val")
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
# 论文 Table 1 那一套（DeQA 没进 pyiqa，略过），外加 topiq_fr。
# topiq_fr 不在论文表里，加它是因为 docs/FINDINGS-赛题逆向.md 反解出赛题的实际公式是
# 感知分 ≈ 13.674·TOPIQ-FR + 4.477·MANIQA − 5.731 —— CLIP-IQA 权重为零、不在公式里。
# 两套口径同时报，免得优化了一个不进分的指标。
FR_METRICS = ["psnr", "ssim", "lpips", "topiq_fr"]
NR_METRICS = ["niqe", "musiq", "maniqa", "clipiqa"]


def load(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1)[None].float().cuda() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--vae", choices=["sd", "taesd"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bench-dir", default=BENCH)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 对，扫曲线时用 300")
    # 评估集 GT 是 512x512，enhance() 里 `min(h0,w0) <= patch_size` 取等号也成立，
    # 所以 patch_size>512 会把 512 上采样到 patch_size 再跑、最后缩回来 ——
    # 等于「把 UNet 的内部工作分辨率抬高」，是个零改造成本的探针。
    # 而 patch_size=512 时只切出 1 个 tile，此时 stride 完全不起作用。
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    a = ap.parse_args()

    names = sorted(os.listdir(os.path.join(a.bench_dir, "gt")))
    if a.limit:
        # 均匀抽而不是取前 N：DIV2K_V2_val 是按源图编号排的，取前 N 只会覆盖头几张图
        names = names[:: max(1, len(names) // a.limit)][:a.limit]

    en = SD2Enhancer(base_model_path=os.path.join(CSIG, "weights"), weight_path=a.weight,
                     lora_modules=LORA_MODULES, lora_rank=256, model_t=200, coeff_t=200,
                     device="cuda")
    en.init_models()
    if a.vae == "taesd":
        from HYPIR.utils.taesd import build_taesd
        en.vae = build_taesd(en.weight_dtype, en.device)

    import pyiqa
    metrics = {k: pyiqa.create_metric(k, device="cuda") for k in FR_METRICS + NR_METRICS}

    rows = {k: [] for k in FR_METRICS + NR_METRICS}
    per_item = []
    torch.manual_seed(231)
    for i, name in enumerate(names):
        gt = load(os.path.join(a.bench_dir, "gt", name))
        lq = load(os.path.join(a.bench_dir, "lq", name))
        with torch.no_grad():
            out = en.enhance(lq, prompt=PROMPT, upscale=4, patch_size=a.patch_size,
                             stride=a.stride, return_type="pt").cuda().clamp(0, 1)
        item = {"name": name}
        for k in FR_METRICS:
            v = float(metrics[k](out, gt))
            rows[k].append(v)
            item[k] = v
        for k in NR_METRICS:
            v = float(metrics[k](out))
            rows[k].append(v)
            item[k] = v
        per_item.append(item)
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(names)), flush=True)

    summary = {k: float(np.mean(v)) for k, v in rows.items()}
    print("\n%-10s n=%d  vae=%s  weight=%s" % (a.tag, len(names), a.vae, a.weight))
    print("  ".join("%s=%.4f" % (k, summary[k]) for k in FR_METRICS + NR_METRICS))

    out_path = os.path.join(CSIG, "out", "bench_%s.json" % a.tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"tag": a.tag, "vae": a.vae, "weight": a.weight, "n": len(names),
               "prompt": PROMPT, "patch_size": a.patch_size, "stride": a.stride,
               "summary": summary, "per_item": per_item},
              open(out_path, "w"))
    print("写到 %s" % out_path)


if __name__ == "__main__":
    main()
