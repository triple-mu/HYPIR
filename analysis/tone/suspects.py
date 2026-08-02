"""在 91 张重编码图里找是否还藏着未退化的图。

判据：同一量化表(q0_mean=5.766)下，比特率 bpp 直接正比于图里真实存在的高频量。
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}


def bpp(r):
    return r["filesize"] * 8 / (r["W"] * r["H"])


print("bpp = 文件比特数 / 像素数。同量化表下 bpp 越高 = 真实高频越多。")
print(f"\n9 张相机原生（量化表 q0_mean=14.30，比验证集LQ粗 2.5 倍）：")
for i in sorted(NATIVE):
    r = S[f"test_case{i}"]
    print(f"  case{i:<4} {r['filesize']/1e6:>5.2f}MB  bpp {bpp(r):>5.2f}  "
          f"lapv {r['lap_var']:>7.1f}  psdHi {r['psd_e_gt025']*100:>6.3f}%  {L[i][0]},{L[i][1]}")
a = [bpp(S[f"test_case{i}"]) for i in NATIVE]
print(f"  -> bpp 均值 {np.mean(a):.2f}, 范围 {min(a):.2f}~{max(a):.2f}")

rest = [i for i in range(1, 101) if i not in NATIVE]
b = np.array([bpp(S[f"test_case{i}"]) for i in rest])
print(f"\n91 张重编码（量化表 q0_mean=5.766，更细）bpp: "
      f"min {b.min():.2f} p50 {np.median(b):.2f} p90 {np.percentile(b,90):.2f} max {b.max():.2f}")
print(f"  两组 bpp 是否重叠: {'重叠' if b.max() > min(a) else '完全分离'} "
      f"(原生最小 {min(a):.2f}, 重编最大 {b.max():.2f})")

print(f"\n验证集（同量化表 5.766）:")
for i in (1, 2, 3):
    l, g = S[f"val_case{i}_lq"], S[f"val_case{i}_gt"]
    fmt = "JPEG" if g["format"] == "JPEG" else "PNG(无损)"
    print(f"  LQ{i} {l['filesize']/1e6:>5.2f}MB bpp {bpp(l):>5.2f} lapv {l['lap_var']:>7.1f} | "
          f"GT{i} {g['filesize']/1e6:>6.2f}MB {fmt} lapv {g['lap_var']:>7.1f}")
print("  注：GT 存的是 PNG/高质量 JPEG，bpp 不可直接比；只有 LQ 与测试集 91 张同管线。")
lqb = [bpp(S[f"val_case{i}_lq"]) for i in (1, 2, 3)]
print(f"  验证集 LQ bpp: {['%.2f'%x for x in lqb]} 均值 {np.mean(lqb):.2f}")

print("\n91 张重编码里 bpp 最高的 15 张（若有未退化图应在此）：")
for x, i in sorted(((bpp(S[f"test_case{i}"]), i) for i in rest), reverse=True)[:15]:
    r = S[f"test_case{i}"]
    print(f"  case{i:<4} bpp {x:>5.2f}  {r['filesize']/1e6:>5.2f}MB  lapv {r['lap_var']:>7.1f}  "
          f"psdHi {r['psd_e_gt025']*100:>6.3f}%  gradm {r['grad_mean']:>5.1f}  "
          f"{L[i][0]},{L[i][1]},{L[i][2]}")
print("91 张里 bpp 最低的 6 张：")
for x, i in sorted(((bpp(S[f"test_case{i}"]), i) for i in rest))[:6]:
    r = S[f"test_case{i}"]
    print(f"  case{i:<4} bpp {x:>5.2f}  lapv {r['lap_var']:>7.1f}  {L[i][0]},{L[i][1]},{L[i][2]}")
