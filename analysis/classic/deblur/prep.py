"""从 3 对验证图裁出多个 1024x1024 patch，存为 npy（uint8 BGR），供参数搜索复用。"""
import numpy as np
from PIL import Image
import os

VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集/"
OUT = "/home/ubuntu/workspace/contest/CSIG-2026/deblur_search/crops/"
S = 1024


def load(p):
    return np.array(Image.open(p).convert("RGB"))[:, :, ::-1].copy()  # BGR


files = {
    "case1": ("case1_lq.jpg", "case1_gt.png"),
    "case2": ("case2_lq.jpg", "case2_gt.png"),
    "case3": ("case3_lq.jpg", "case3_gt.jpg"),
}

for k, (flq, fgt) in files.items():
    lq = load(VAL + flq)
    gt = load(VAL + fgt)
    assert lq.shape == gt.shape, (k, lq.shape, gt.shape)
    H, W = lq.shape[:2]
    # 中心 + 4 个偏移位置，覆盖不同内容
    pos = [
        (H // 2 - S // 2, W // 2 - S // 2),
        (H // 4 - S // 2, W // 4 - S // 2),
        (H * 3 // 4 - S // 2, W // 4 - S // 2),
        (H // 4 - S // 2, W * 3 // 4 - S // 2),
        (H * 3 // 4 - S // 2, W * 3 // 4 - S // 2),
    ]
    for i, (y, x) in enumerate(pos):
        y = max(0, min(y, H - S))
        x = max(0, min(x, W - S))
        np.save(f"{OUT}{k}_{i}_lq.npy", lq[y:y + S, x:x + S])
        np.save(f"{OUT}{k}_{i}_gt.npy", gt[y:y + S, x:x + S])
    print(k, lq.shape, "saved 5 crops")
