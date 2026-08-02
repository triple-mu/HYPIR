"""1:1 像素裁切对比：相机原生 vs 重编码，同题材（夜景远景城市）。"""
import numpy as np
from PIL import Image, ImageDraw

T = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
V = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
S = 460


def crop(path, label, frac=(0.5, 0.45)):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x = int(W * frac[0]) - S // 2
    y = int(H * frac[1]) - S // 2
    c = im.crop((x, y, x + S, y + S))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, 210, 26], fill=(0, 0, 0))
    d.text((5, 6), label, fill=(255, 235, 0))
    return c


def grid(items, out, cols=3):
    rows = (len(items) + cols - 1) // cols
    G = Image.new("RGB", (cols * S, rows * S), (0, 0, 0))
    for k, (p, l, f) in enumerate(items):
        G.paste(crop(p, l, f), ((k % cols) * S, (k // cols) * S))
    G.save(out, quality=95)
    print(out)


# 夜景远景城市：原生 vs 重编码
grid([(f"{T}/case64.jpg", "c64 NATIVE night-city", (0.5, 0.45)),
      (f"{T}/case65.jpg", "c65 NATIVE night-city", (0.5, 0.45)),
      (f"{T}/case66.jpg", "c66 NATIVE night-city", (0.5, 0.45)),
      (f"{T}/case68.jpg", "c68 REENC night-city", (0.5, 0.45)),
      (f"{T}/case92.jpg", "c92 REENC night-city", (0.5, 0.45)),
      (f"{T}/case37.jpg", "c37 REENC night-city", (0.5, 0.45))],
     "/tmp/csig_sheets/crop_night.jpg")

# 白天：原生 c38 vs 重编码近景街景/建筑
grid([(f"{T}/case38.jpg", "c38 NATIVE day-street", (0.45, 0.4)),
      (f"{T}/case19.jpg", "c19 REENC day-street", (0.5, 0.5)),
      (f"{T}/case50.jpg", "c50 REENC day-street", (0.5, 0.5)),
      (f"{T}/case5.jpg", "c5 NATIVE night-facade", (0.5, 0.5)),
      (f"{T}/case97.jpg", "c97 REENC night-facade", (0.5, 0.5)),
      (f"{T}/case90.jpg", "c90 REENC day-facade", (0.5, 0.5))],
     "/tmp/csig_sheets/crop_day.jpg")

# 验证集 LQ vs GT 参照
grid([(f"{V}/case1_lq.jpg", "val1 LQ", (0.5, 0.5)),
      (f"{V}/case1_gt.png", "val1 GT", (0.5, 0.5)),
      (f"{T}/case1.jpg", "c1 NATIVE", (0.4, 0.35)),
      (f"{V}/case3_lq.jpg", "val3 LQ", (0.5, 0.5)),
      (f"{V}/case3_gt.jpg", "val3 GT", (0.5, 0.5)),
      (f"{T}/case99.jpg", "c99 REENC", (0.5, 0.5))],
     "/tmp/csig_sheets/crop_val.jpg")
