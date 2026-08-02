"""在 P0 评估集上给一份 LoRA 权重打分。

打分只用 TOPIQ-FR + MANIQA —— FINDINGS §4 实测只有这两个能复现官方排序，
LPIPS/DISTS/MUSIQ/NIQE/PSNR/SSIM 全都把过锐化的 USM3.5 排在 LQ 原图之上。

主指标是**保留增益** retained = (p_out - p_lq)/(p_gt - p_lq)，有界且可跨数据集比。
不要用 p_out/p_lq 比值：合成评估集上 p_lq 常接近 0（甚至为负），比值会爆炸或变号。

用法:
    python eval_p0.py <weight_path> [--tag 名字] [--limit N]
"""
import argparse, json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iqa
from HYPIR.enhancer.sd2 import SD2Enhancer

CSIG = "/root/.cache/huggingface/csig"
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
PROMPT = ("A high-resolution photograph with fine natural detail: modern Chinese city "
          "high-rise buildings with glass curtain walls and tiled facades, construction "
          "cranes and shop signage, distant hills, and close-up green foliage with flowers.")


def load(p):
    return torch.from_numpy(cv2.imread(p)).cuda().flip(-1).permute(2, 0, 1)[None].float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight_path")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-dir", default=os.path.join(CSIG, "data/eval_p0"))
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.eval_dir, "meta.json")))["items"]
    if a.limit:
        meta = meta[:a.limit]

    en = SD2Enhancer(base_model_path=os.path.join(CSIG, "weights"), weight_path=a.weight_path,
                     lora_modules=LORA_MODULES, lora_rank=256, model_t=200, coeff_t=200,
                     device="cuda")
    en.init_models()

    rows, per_item = {}, []
    for i, it in enumerate(meta):
        gt = load(os.path.join(a.eval_dir, "gt", it["name"]))
        lq = load(os.path.join(a.eval_dir, "lq", it["name"]))
        with torch.no_grad():
            out = en.enhance(lq, prompt=PROMPT, upscale=1, patch_size=512, stride=256,
                             return_type="pt").clamp(0, 1)
        # iqa.score 吃 [-1,1]
        f_o, n_o, p_o = iqa.score(out * 2 - 1, gt * 2 - 1)
        f_l, n_l, p_l = iqa.score(lq * 2 - 1, gt * 2 - 1)
        f_g, n_g, p_g = iqa.score(gt * 2 - 1, gt * 2 - 1)
        rows.setdefault(it["cat"], []).append((f_o, n_o, p_o, f_l, n_l, p_l, p_g))
        per_item.append({"name": it["name"], "cat": it["cat"], "fr_out": float(f_o),
                         "nr_out": float(n_o), "p_out": float(p_o), "p_lq": float(p_l),
                         "p_gt": float(p_g)})
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(meta)), flush=True)

    def line(tag, v):
        ret = (v[:, 2].mean() - v[:, 5].mean()) / max(v[:, 6].mean() - v[:, 5].mean(), 1e-9)
        print("%-12s %5d | %7.4f %7.4f %7.4f | %7.3f %7.3f %7.3f | %8.1f%%" %
              (tag, len(v), v[:, 3].mean(), v[:, 0].mean(), v[:, 1].mean(),
               v[:, 5].mean(), v[:, 2].mean(), v[:, 6].mean(), 100 * ret))
        return ret
    print("\n%-12s %5s | %7s %7s %7s | %7s %7s %7s | %9s" %
          ("类别", "n", "FR_lq", "FR_out", "NR_out", "p_lq", "p_out", "p_gt", "保留增益"))
    print("-" * 82)
    allr = []
    for cat in sorted(rows):
        v = np.array(rows[cat]); allr.append(v)
        line(cat, v)
    v = np.concatenate(allr)
    print("-" * 82)
    ret = line("总计", v)
    print("\n保留增益 = (p_out-p_lq)/(p_gt-p_lq)。0%=没改善, 100%=完美复原。")
    print("参考: 赛题验证集上 HYPIR 吃到 22.5%%（p=2.65, 天花板 8.34）")
    if a.tag:
        # FR 与 NR 分量必须一起存：p 是二者的线性组合，只看 p 会漏掉「FR 涨、NR 跌、
        # 净值持平」这种情况——两个指标反向走时，模型在往哪边跑只有分量能回答。
        json.dump({"tag": a.tag, "weight": a.weight_path,
                   "retained": float(ret), "p_out": float(v[:, 2].mean()),
                   "fr_out": float(v[:, 0].mean()), "nr_out": float(v[:, 1].mean()),
                   "fr_lq": float(v[:, 3].mean()), "nr_lq": float(v[:, 4].mean()),
                   "p_lq": float(v[:, 5].mean()), "p_gt": float(v[:, 6].mean()),
                   "n": int(len(v)),
                   "per_item": per_item,
                   "by_cat": {cat: {"n": int(len(np.array(rows[cat]))),
                                    "fr_out": float(np.array(rows[cat])[:, 0].mean()),
                                    "nr_out": float(np.array(rows[cat])[:, 1].mean()),
                                    "p_out": float(np.array(rows[cat])[:, 2].mean())}
                              for cat in sorted(rows)}},
                  open(os.path.join(CSIG, "out", "eval_%s.json" % a.tag), "w"))


if __name__ == "__main__":
    main()
