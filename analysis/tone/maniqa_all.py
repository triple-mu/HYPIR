"""对 100 张测试图 + 验证集 6 张算 MANIQA。

MANIQA 对分辨率敏感，必须与评测口径一致。这里同时算两种口径：
  full  —— 整图直接喂（与线上评测最接近，若 OOM 则退化到分块）
  crop5 —— 5 个 512 crop（中心+四角）的均值，作为内容无关的稳健参考
"""
import os, json, sys
import numpy as np
import torch
from PIL import Image
import pyiqa

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
OUT = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
dev = "cuda"
m = pyiqa.create_metric("maniqa", device=dev)


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)


@torch.no_grad()
def score_full(a):
    """整图。4K 直接喂显存不够时按 1024 网格分块平均。"""
    try:
        return float(m(to_t(a)))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        H, W = a.shape[:2]
        vals = []
        for y in range(0, H - 1023, 1024):
            for x in range(0, W - 1023, 1024):
                vals.append(float(m(to_t(a[y:y + 1024, x:x + 1024]))))
        return float(np.mean(vals))


@torch.no_grad()
def score_crops(a, s=512):
    H, W = a.shape[:2]
    cy, cx = (H - s) // 2, (W - s) // 2
    pos = [(cy, cx), (0, 0), (0, W - s), (H - s, 0), (H - s, W - s)]
    return [float(m(to_t(a[y:y + s, x:x + s]))) for y, x in pos]


def run(path, name):
    a = np.asarray(Image.open(path).convert("RGB"))
    cs = score_crops(a)
    torch.cuda.empty_cache()
    f = score_full(a)
    torch.cuda.empty_cache()
    r = {"name": name, "maniqa_full": f, "maniqa_crop5": float(np.mean(cs)),
         "crops": cs, "crop_std": float(np.std(cs))}
    print(json.dumps(r), flush=True)
    return r


res = []
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        c = [f"{VAL}/case{i}_{k}.png", f"{VAL}/case{i}_{k}.jpg"]
        res.append(run([x for x in c if os.path.exists(x)][0], f"val_case{i}_{k}"))
for i in range(1, 101):
    res.append(run(f"{TEST}/case{i}.jpg", f"test_case{i}"))
json.dump(res, open(f"{OUT}/maniqa_all.json", "w"))
print("WROTE")
