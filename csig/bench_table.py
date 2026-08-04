"""把 $CSIG/out/bench_*.json 汇成一张表。

    python csig/bench_table.py [--filter taesd_main] [--sort musiq]

判据（见 docs 与训练计划）：arm C（本轮 LoRA + TAESD）要显著超过 arm B
（官方 LoRA + TAESD 零样本），并追平 arm A（官方 LoRA + SD-VAE）。
论文 Table 1 的绝对值不能直接比 —— pyiqa 与论文的指标实现口径不同，
实测这套评估集上 GT 自身的 MUSIQ 才 63.0，而论文给 HYPIR 报 72.58。
"""
import argparse
import glob
import json
import os
import re

CSIG = os.environ.get("CSIG", "/root/.cache/huggingface/csig")
COLS = ["psnr", "ssim", "lpips", "niqe", "musiq", "maniqa", "clipiqa"]


def step_of(tag):
    m = re.search(r"_(\d+)_(raw|ema)$", tag)
    return (int(m.group(1)), m.group(2)) if m else (-1, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--sort", default="step", choices=["step"] + COLS)
    a = ap.parse_args()

    rows = []
    for p in glob.glob(os.path.join(CSIG, "out", "bench_*.json")):
        d = json.load(open(p))
        if a.filter and a.filter not in d["tag"]:
            continue
        rows.append(d)
    if not rows:
        print("没有匹配的 bench_*.json")
        return

    if a.sort == "step":
        rows.sort(key=lambda d: (step_of(d["tag"]), d["tag"]))
    else:
        rows.sort(key=lambda d: d["summary"][a.sort], reverse=a.sort not in ("lpips", "niqe"))

    w = max(len(d["tag"]) for d in rows) + 2
    print("%-*s %5s %5s | %s" % (w, "tag", "n", "vae",
                                 "  ".join("%8s" % (c + ("↓" if c in ("lpips", "niqe") else "")) for c in COLS)))
    print("-" * (w + 14 + 10 * len(COLS)))
    for d in rows:
        print("%-*s %5d %5s | %s" % (w, d["tag"], d["n"], d["vae"],
                                     "  ".join("%8.4f" % d["summary"][c] for c in COLS)))


if __name__ == "__main__":
    main()
