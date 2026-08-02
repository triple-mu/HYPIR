"""内容受控对比：同为「夜景远景城市」的 7 张相机原生 vs 5 张重编码。

这消除了内容差异这个最大混淆项——同一摄影师、同一相机、同一批次、同一题材。
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
try:
    M = {r["name"]: r["maniqa_full"] for r in json.load(open(f"{D}/maniqa_all.json"))}
except FileNotFoundError:
    M = {}

NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}
nc = [i for i in range(1, 101) if L[i][0] == "city_far" and L[i][1] == "night" and i in NATIVE]
rc = [i for i in range(1, 101) if L[i][0] == "city_far" and L[i][1] == "night" and i not in NATIVE]
print(f"原生夜景城市 {nc}")
print(f"重编码夜景城市 {rc}")
print()
K = ["lum_mean", "lum_std", "grad_mean", "lap_var", "lap_var_norm",
     "psd_e_gt025", "psd_e_0375_05", "filesize"]
hdr = f"{'case':<8}" + "".join(f"{k[:11]:>13}" for k in K) + f"{'MANIQA':>9}"
print(hdr)
for tag, grp in (("原生", nc), ("重编", rc)):
    for i in grp:
        t = S[f"test_case{i}"]
        print(f"{tag}c{i:<5}" + "".join(f"{t[k]:>13.5g}" for k in K) +
              (f"{M.get(f'test_case{i}', float('nan')):>9.4f}" if M else ""))
    print("-" * len(hdr))
print()
for k in K + (["MANIQA"] if M else []):
    if k == "MANIQA":
        a = np.array([M[f"test_case{i}"] for i in nc])
        b = np.array([M[f"test_case{i}"] for i in rc])
    else:
        a = np.array([S[f"test_case{i}"][k] for i in nc])
        b = np.array([S[f"test_case{i}"][k] for i in rc])
    print(f"{k:<16} 原生均值 {a.mean():>12.5g}   重编均值 {b.mean():>12.5g}   "
          f"比值 {a.mean()/max(b.mean(),1e-30):>7.2f}x   "
          f"重叠? {'否(完全分离)' if a.min() > b.max() or a.max() < b.min() else '是'}")

print()
print("=== 验证集 LQ/GT 作为标尺（同为城市题材）===")
for k in K[:7]:
    lq = np.mean([S[f"val_case{i}_lq"][k] for i in (1, 2, 3)])
    gt = np.mean([S[f"val_case{i}_gt"][k] for i in (1, 2, 3)])
    print(f"{k:<16} LQ {lq:>12.5g}   GT {gt:>12.5g}   GT/LQ {gt/max(lq,1e-30):.2f}x")

print()
print("=== 全 100 张：原生 9 vs 重编 91（不控内容，供参考）===")
for k in K + (["MANIQA"] if M else []):
    if k == "MANIQA":
        a = np.array([M[f"test_case{i}"] for i in sorted(NATIVE)])
        b = np.array([M[f"test_case{i}"] for i in range(1, 101) if i not in NATIVE])
    else:
        a = np.array([S[f"test_case{i}"][k] for i in sorted(NATIVE)])
        b = np.array([S[f"test_case{i}"][k] for i in range(1, 101) if i not in NATIVE])
    from scipy import stats as st
    u = st.mannwhitneyu(a, b, alternative="greater")
    print(f"{k:<16} 原生中位 {np.median(a):>12.5g}  重编中位 {np.median(b):>12.5g}  "
          f"Mann-Whitney p(原生>重编)={u.pvalue:.2e}")
