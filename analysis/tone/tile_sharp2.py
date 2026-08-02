"""修正版：先选出「最有纹理的 10% tile」，再在这些 tile 上聚合求比值。

上一版用逐 tile 的 hi/lo 再取 p95，在天空等平坦区 lo≈0 会把比值撑爆
（case21/22/11/71 全是大片白天空），指标失效。这版改成先选后聚合。
"""
import json, sys, os
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

T = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
V = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}

uu, vv = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
ring = uu + vv
LOW = (ring >= 1) & (ring <= 3)
HIGH = (ring >= 6) & (ring <= 14)
C = np.zeros((8, 8), np.float32)
for k in range(8):
    for n in range(8):
        C[k, n] = np.sqrt((1 if k else 0.5) / 4) * np.cos(np.pi * (2 * n + 1) * k / 16)


def metric(path, top=0.10):
    a = np.asarray(Image.open(path).convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h8, w8 = g.shape[0] // 8 * 8, g.shape[1] // 8 * 8
    B = g[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    D = np.einsum("ij,bcjk,lk->bcil", C, B, C)
    lo = (D[..., LOW] ** 2).sum(-1)
    hi = (D[..., HIGH] ** 2).sum(-1)
    by, bx = lo.shape
    ty, tx = by // 16, bx // 16
    LO = lo[:ty * 16, :tx * 16].reshape(ty, 16, tx, 16).sum((1, 3)).ravel()
    HI = hi[:ty * 16, :tx * 16].reshape(ty, 16, tx, 16).sum((1, 3)).ravel()
    k = max(1, int(len(LO) * top))
    sel = np.argsort(LO)[-k:]                     # 纹理最强的 10% tile
    return float(HI[sel].sum() / max(LO[sel].sum(), 1e-9))


print("HFR = sum(DCT高频环带6-14) / sum(DCT低频环带1-3)，只在纹理最强的 10% 个 128px tile 上聚合")
print()
ref = {}
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        p = [x for x in (f"{V}/case{i}_{k}.png", f"{V}/case{i}_{k}.jpg") if os.path.exists(x)][0]
        ref[f"{i}{k}"] = metric(p)
print("=== 验证集 ===")
for i in (1, 2, 3):
    print(f"  case{i}  LQ {ref[f'{i}lq']:.5f}   GT {ref[f'{i}gt']:.5f}   GT/LQ {ref[f'{i}gt']/ref[f'{i}lq']:.1f}x")
lqs = [ref[f"{i}lq"] for i in (1, 2, 3)]
gts = [ref[f"{i}gt"] for i in (1, 2, 3)]
print(f"  LQ 区间 {min(lqs):.5f}~{max(lqs):.5f}   GT 区间 {min(gts):.5f}~{max(gts):.5f}  （无重叠）")

res = {i: metric(f"{T}/case{i}.jpg") for i in range(1, 101)}
nv = np.array([res[i] for i in sorted(NATIVE)])
rv = np.array([res[i] for i in range(1, 101) if i not in NATIVE])
print()
print("=== 测试集 ===")
print(f"9 张原生: " + "  ".join(f"c{i}={res[i]:.5f}" for i in sorted(NATIVE)))
print(f"  均值 {nv.mean():.5f}  范围 {nv.min():.5f}~{nv.max():.5f}")
print(f"91 张重编码: min {rv.min():.5f} p10 {np.percentile(rv,10):.5f} p50 {np.percentile(rv,50):.5f} "
      f"p90 {np.percentile(rv,90):.5f} p99 {np.percentile(rv,99):.5f} max {rv.max():.5f}")
print(f"分离性: 原生最小 {nv.min():.5f} vs 重编最大 {rv.max():.5f} -> "
      f"{'完全分离' if nv.min() > rv.max() else '有重叠'}")
over = [i for i in range(1, 101) if i not in NATIVE and res[i] > max(lqs)]
print(f"\n91 张重编码里 HFR > 验证集 LQ 最大值({max(lqs):.5f}) 的: {over}")
overg = [i for i in range(1, 101) if i not in NATIVE and res[i] >= min(gts)]
print(f"91 张重编码里 HFR >= 验证集 GT 最小值({min(gts):.5f}) 的: {overg}")
print(f"9 张原生里 HFR >= 验证集 GT 最小值 的: "
      f"{[i for i in sorted(NATIVE) if res[i] >= min(gts)]}")
print(f"9 张原生里 HFR <= 验证集 LQ 最大值 的: "
      f"{[i for i in sorted(NATIVE) if res[i] <= max(lqs)]}")

print("\n全部 100 张按 HFR 降序前 20：")
for i in sorted(range(1, 101), key=lambda j: -res[j])[:20]:
    print(f"  case{i:<4} {res[i]:.5f}  {'原生*' if i in NATIVE else '重编 '}  "
          f"{L[i][0]},{L[i][1]},{L[i][2]}")
print("末尾 8 张：")
for i in sorted(range(1, 101), key=lambda j: res[j])[:8]:
    print(f"  case{i:<4} {res[i]:.5f}  {'原生*' if i in NATIVE else '重编 '}  "
          f"{L[i][0]},{L[i][1]},{L[i][2]}")

# 内容受控：夜景远景城市
nc = [1, 2, 3, 4, 64, 65, 66]
rc = [14, 37, 49, 68, 92]
print(f"\n内容受控（夜景远景城市）原生 {[f'{res[i]:.4f}' for i in nc]} 均值 {np.mean([res[i] for i in nc]):.5f}")
print(f"                        重编 {[f'{res[i]:.4f}' for i in rc]} 均值 {np.mean([res[i] for i in rc]):.5f}")
print(f"  比值 {np.mean([res[i] for i in nc])/np.mean([res[i] for i in rc]):.1f}x, "
      f"完全分离: {min(res[i] for i in nc) > max(res[i] for i in rc)}")
json.dump({"test": res, "val": ref}, open("/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone/hfr2.json", "w"))
