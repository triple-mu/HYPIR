"""生成缩略图 + 9 宫格接触表，供人眼/VLM 分类内容。"""
import os
import numpy as np
from PIL import Image, ImageDraw

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
OUT = "/tmp/csig_sheets"
THUMB = "/tmp/csig_thumbs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(THUMB, exist_ok=True)

TILE = 620


def tile_of(path, label):
    im = Image.open(path).convert("RGB")
    im.thumbnail((TILE, TILE), Image.LANCZOS)
    im.save(f"{THUMB}/{label}.jpg", quality=92)
    canvas = Image.new("RGB", (TILE, TILE), (24, 24, 24))
    canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 150, 34], fill=(0, 0, 0))
    d.text((6, 8), label, fill=(255, 230, 0))
    return canvas


def sheet(paths, labels, out):
    n = len(paths)
    cols = 3
    rows = (n + cols - 1) // cols
    S = Image.new("RGB", (cols * TILE, rows * TILE), (12, 12, 12))
    for k, (p, l) in enumerate(zip(paths, labels)):
        S.paste(tile_of(p, l), ((k % cols) * TILE, (k // cols) * TILE))
    S.save(out, quality=88)
    print(out)


paths = [f"{TEST}/case{i}.jpg" for i in range(1, 101)]
labels = [f"c{i}" for i in range(1, 101)]
for b in range(0, 100, 9):
    sheet(paths[b:b + 9], labels[b:b + 9], f"{OUT}/sheet_{b//9:02d}.jpg")

vp, vl = [], []
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        c = [f"{VAL}/case{i}_{k}.png", f"{VAL}/case{i}_{k}.jpg"]
        vp.append([x for x in c if os.path.exists(x)][0])
        vl.append(f"val{i}_{k}")
sheet(vp, vl, f"{OUT}/sheet_val.jpg")
