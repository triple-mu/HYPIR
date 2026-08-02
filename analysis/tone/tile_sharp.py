"""纹理条件化的锐度指标，避开「雾/平坦区」这个内容混淆项。

把图切成 128x128 tile，每个 tile 做 8x8 DCT，算
  HFR = (DCT 环带 4-7 的能量) / (DCT 环带 1-3 的能量)
这是尺度不变、对比度不变的量：只看「高频相对中频的占比」。
取所有 tile 的 p95（最锐的 5% 区域），代表这张图「最好的地方能有多锐」。
带限图即使在最锐处 HFR 也上不去。
"""
import json, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

T = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
V = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}

# 8x8 DCT 的环带索引（按 u+v 分组）
uu, vv = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
ring = uu + vv
LOW = (ring >= 1) & (ring <= 3)
HIGH = (ring >= 6) & (ring <= 14)


def hfr_map(gray):
    """对 8x8 分块做 DCT，返回每块的 (低频能量, 高频能量)。"""
    h, w = gray.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    g = gray[:h8, :w8].astype(np.float32)
    # 分块 DCT：reshape 成 (nb, 8, 8) 后逐块 dct 太慢，用可分离 DCT 矩阵
    C = np.zeros((8, 8), np.float32)
    for k in range(8):
        for n in range(8):
            C[k, n] = np.sqrt((1 if k else 0.5) / 4) * np.cos(np.pi * (2 * n + 1) * k / 16)
    B = g.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)   # (by,bx,8,8)
    D = np.einsum("ij,bcjk,lk->bcil", C, B, C)
    lo = (D[..., LOW] ** 2).sum(-1)
    hi = (D[..., HIGH] ** 2).sum(-1)
    return lo, hi


def metric(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    lo, hi = hfr_map(gray)
    # 128x128 tile = 16x16 个 8x8 块
    by, bx = lo.shape
    ty, tx = by // 16, bx // 16
    lo = lo[:ty * 16, :tx * 16].reshape(ty, 16, tx, 16).sum((1, 3))
    hi = hi[:ty * 16, :tx * 16].reshape(ty, 16, tx, 16).sum((1, 3))
    r = hi / np.maximum(lo, 1e-9)
    # 只保留有内容的 tile（低频能量前 50%），再取 HFR 的 p95
    m = lo > np.median(lo)
    rv = r[m]
    return float(np.percentile(rv, 95)), float(np.percentile(rv, 50)), int(m.sum())


print("HFR = DCT 高频环带(6-14) 能量 / 低频环带(1-3) 能量，128px tile，取有内容 tile 的 p95")
print()
print("=== 验证集（LQ vs GT 的标尺）===")
import os
ref = {}
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        p = [x for x in (f"{V}/case{i}_{k}.png", f"{V}/case{i}_{k}.jpg") if os.path.exists(x)][0]
        p95, p50, n = metric(p)
        ref[f"{i}{k}"] = p95
        print(f"  val{i}_{k}  HFR_p95 {p95:.5f}  HFR_p50 {p50:.5f}")
for i in (1, 2, 3):
    print(f"  case{i}: GT/LQ = {ref[f'{i}gt']/ref[f'{i}lq']:.2f}x")
lqs = [ref[f"{i}lq"] for i in (1, 2, 3)]
gts = [ref[f"{i}gt"] for i in (1, 2, 3)]
print(f"  LQ 区间 {min(lqs):.5f}~{max(lqs):.5f}   GT 区间 {min(gts):.5f}~{max(gts):.5f}")

print()
print("=== 测试集 100 张 ===")
res = {}
for i in range(1, 101):
    res[i] = metric(f"{T}/case{i}.jpg")
    if i % 25 == 0:
        print(f"  ...{i}", flush=True)
nv = np.array([res[i][0] for i in sorted(NATIVE)])
rv = np.array([res[i][0] for i in range(1, 101) if i not in NATIVE])
print(f"\n9 张原生 HFR_p95: {['%.5f'%x for x in nv]}")
print(f"  均值 {nv.mean():.5f} 范围 {nv.min():.5f}~{nv.max():.5f}")
print(f"91 张重编码 HFR_p95: min {rv.min():.5f} p10 {np.percentile(rv,10):.5f} "
      f"p50 {np.percentile(rv,50):.5f} p90 {np.percentile(rv,90):.5f} max {rv.max():.5f}")
print(f"两组是否分离: 原生最小 {nv.min():.5f} vs 重编最大 {rv.max():.5f} -> "
      f"{'完全分离' if nv.min() > rv.max() else '有重叠'}")
print(f"91 张中 HFR_p95 >= 原生最小值 的: "
      f"{[i for i in range(1,101) if i not in NATIVE and res[i][0] >= nv.min()]}")
print(f"91 张中 HFR_p95 >= 验证集 GT 最小值({min(gts):.5f}) 的: "
      f"{[i for i in range(1,101) if i not in NATIVE and res[i][0] >= min(gts)]}")
print(f"91 张中 HFR_p95 > 验证集 LQ 最大值({max(lqs):.5f}) 的: "
      f"{[i for i in range(1,101) if i not in NATIVE and res[i][0] > max(lqs)]}")

print("\n全部 100 张按 HFR_p95 降序前 20：")
for x, i in sorted(((res[i][0], i) for i in range(1, 101)), reverse=True)[:20]:
    print(f"  case{i:<4} {x:.5f}  {'原生*' if i in NATIVE else '重编 '}  "
          f"{L[i][0]},{L[i][1]},{L[i][2]}")
json.dump({str(i): res[i] for i in res} | {f"val{i}{k}": ref[f"{i}{k}"] for i in (1,2,3) for k in ("lq","gt")},
          open("/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone/hfr.json", "w"))
