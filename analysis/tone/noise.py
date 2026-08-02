"""平坦区噪声（Immerkaer sigma）—— 与内容几乎无关的判据。

FINDINGS §3：验证集 LQ sigma 0.19~0.37（几乎无噪），GT 0.46~1.03（有噪），
case3 的 LQ 平坦区 std 严格为 0。若某张测试图带相机噪声，它就没走过那条抹噪的管线。
只在「最平坦的 5% tile」上估计，避开纹理污染。
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
K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def sigma_flat(path, tile=64, frac=0.05):
    g = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = np.abs(cv2.filter2D(g, cv2.CV_32F, K))
    h, w = g.shape
    ty, tx = h // tile, w // tile
    r = r[:ty * tile, :tx * tile].reshape(ty, tile, tx, tile)
    gg = g[:ty * tile, :tx * tile].reshape(ty, tile, tx, tile)
    m = r.mean((1, 3)).ravel()            # 每 tile 的响应均值
    sd = gg.std((1, 3)).ravel()           # 每 tile 的像素 std（用来选平坦区）
    k = max(1, int(len(sd) * frac))
    sel = np.argsort(sd)[:k]              # 最平坦的 5% tile
    # Immerkaer: sigma = mean|response| * sqrt(pi/2) / 6
    return float(np.mean(m[sel]) * np.sqrt(np.pi / 2) / 6), float(np.median(sd[sel]))


print("sigma_flat = 最平坦 5% 的 64px tile 上的 Immerkaer 噪声估计（8bit 灰度单位）")
print()
print("=== 验证集 ===")
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        p = [x for x in (f"{V}/case{i}_{k}.png", f"{V}/case{i}_{k}.jpg") if os.path.exists(x)][0]
        s, sd = sigma_flat(p)
        print(f"  val{i}_{k}  sigma {s:.4f}   平坦tile中位std {sd:.4f}")

res = {}
for i in range(1, 101):
    res[i] = sigma_flat(f"{T}/case{i}.jpg")
nv = np.array([res[i][0] for i in sorted(NATIVE)])
rv = np.array([res[i][0] for i in range(1, 101) if i not in NATIVE])
print()
print("=== 测试集 ===")
print("9 张相机原生：")
for i in sorted(NATIVE):
    print(f"  case{i:<4} sigma {res[i][0]:.4f}  平坦tile中位std {res[i][1]:.4f}  {L[i][0]},{L[i][1]}")
print(f"  -> 均值 {nv.mean():.4f} 范围 {nv.min():.4f}~{nv.max():.4f}")
print(f"91 张重编码 sigma: min {rv.min():.4f} p10 {np.percentile(rv,10):.4f} "
      f"p50 {np.percentile(rv,50):.4f} p90 {np.percentile(rv,90):.4f} max {rv.max():.4f}")
print(f"分离性：原生最小 {nv.min():.4f} vs 重编最大 {rv.max():.4f} -> "
      f"{'完全分离' if nv.min() > rv.max() else '有重叠'}")
ov = [i for i in range(1, 101) if i not in NATIVE and res[i][0] >= nv.min()]
print(f"91 张里 sigma >= 原生最小值 的: {ov}")
print()
print("全 100 张 sigma 降序前 15：")
for i in sorted(range(1, 101), key=lambda j: -res[j][0])[:15]:
    print(f"  case{i:<4} {res[i][0]:.4f}  {'原生*' if i in NATIVE else '重编 '}  {L[i][0]},{L[i][1]}")
json.dump({str(i): res[i] for i in res}, open("/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone/noise.json", "w"))
