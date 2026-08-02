"""汇总：内容分类 x 统计量 x JPEG 段 x MANIQA，输出分布与可疑图列表。"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L, VAL

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
try:
    M = {r["name"]: r for r in json.load(open(f"{D}/maniqa_all.json"))}
except FileNotFoundError:
    M = {}

test = [S[f"test_case{i}"] for i in range(1, 101)]
lq = [S[f"val_case{i}_lq"] for i in (1, 2, 3)]
gt = [S[f"val_case{i}_gt"] for i in (1, 2, 3)]

KEYS = ["filesize", "lum_mean", "lum_std", "sat_mean", "frac_zero_any", "frac_255_any",
        "grad_mean", "lap_var", "lap_var_norm",
        "psd_e_gt025", "psd_ratio_hi_mid", "psd_f_cut1e3",
        "q0_dc", "q0_mean", "q0_max", "q1_mean"]


def pct_rank(vals, x):
    return 100.0 * float(np.mean(np.asarray(vals) < x))


print("=" * 100)
print("A. 各统计量的测试集分布，以及验证集 3LQ / 3GT 落点（括号内是在 100 张测试集中的百分位）")
print("=" * 100)
hdr = f"{'metric':<18}{'min':>10}{'p10':>10}{'p50':>10}{'p90':>10}{'max':>10} | {'LQ1/2/3 (pct)':<36} {'GT1/2/3 (pct)'}"
print(hdr)
for k in KEYS:
    v = np.array([t.get(k, np.nan) for t in test], float)
    ok = v[~np.isnan(v)]
    q = np.percentile(ok, [0, 10, 50, 90, 100])
    sl = " ".join(f"{x[k]:.4g}({pct_rank(ok, x[k]):.0f})" for x in lq if k in x)
    sg = " ".join(f"{x[k]:.4g}({pct_rank(ok, x[k]):.0f})" for x in gt if k in x)
    print(f"{k:<18}" + "".join(f"{x:>10.4g}" for x in q) + f" | {sl:<36} {sg}")

print()
print("=" * 100)
print("B. JPEG 段签名分组：量化表 + APP 标记")
print("=" * 100)
val_q = (lq[0]["q0_mean"], lq[0]["q1_mean"])
print(f"验证集 LQ 量化表签名 = q0_mean {val_q[0]:.4f}, q1_mean {val_q[1]:.4f}")
same_q, diff_q = [], []
for i, t in enumerate(test, 1):
    if abs(t.get("q0_mean", -1) - val_q[0]) < 1e-6 and abs(t.get("q1_mean", -1) - val_q[1]) < 1e-6:
        same_q.append(i)
    else:
        diff_q.append(i)
print(f"与验证集 LQ 量化表完全相同: {len(same_q)} 张 -> {same_q}")
print(f"量化表不同(粗): {len(diff_q)} 张 -> {diff_q}")

has_exif, has_mpf, only_jfif = [], [], []
for i, t in enumerate(test, 1):
    tags = [a[1] for a in t.get("apps", [])]
    if any("Exif" in x for x in tags):
        has_exif.append(i)
    if any("MPF" in x for x in tags):
        has_mpf.append(i)
    if set(x.strip("\x00") for x in tags) <= {"JFIF"}:
        only_jfif.append(i)
print(f"带 EXIF: {len(has_exif)} -> {has_exif}")
print(f"带 MPF : {len(has_mpf)} -> {has_mpf}")
print(f"只有 JFIF(被重编码): {len(only_jfif)} 张")

print()
print("APP 段组合的去重清单：")
from collections import Counter
sig = Counter()
for i, t in enumerate(test, 1):
    sig[tuple(sorted(set(a[1].strip('\x00') for a in t.get("apps", []))))] += 1
for k, v in sig.most_common():
    print(f"  {v:>3} 张  {k}")

print()
print("量化表 (q0_dc,q0_mean,q0_max) 的去重清单：")
sig2 = Counter()
for i, t in enumerate(test, 1):
    sig2[(t.get("q0_dc"), round(t.get("q0_mean", 0), 4), t.get("q0_max"),
          round(t.get("q1_mean", 0), 4))] += 1
for k, v in sig2.most_common():
    print(f"  {v:>3} 张  q0_dc={k[0]} q0_mean={k[1]} q0_max={k[2]} q1_mean={k[3]}")

if M:
    print()
    print("=" * 100)
    print("C. MANIQA")
    print("=" * 100)
    for tag in ("maniqa_full", "maniqa_crop5"):
        mv = np.array([M[f"test_case{i}"][tag] for i in range(1, 101)])
        lqv = [M[f"val_case{i}_lq"][tag] for i in (1, 2, 3)]
        gtv = [M[f"val_case{i}_gt"][tag] for i in (1, 2, 3)]
        print(f"\n--- {tag} ---")
        print(f"测试集 100 张: min {mv.min():.4f} p10 {np.percentile(mv,10):.4f} "
              f"p25 {np.percentile(mv,25):.4f} p50 {np.percentile(mv,50):.4f} "
              f"p75 {np.percentile(mv,75):.4f} p90 {np.percentile(mv,90):.4f} max {mv.max():.4f} "
              f"mean {mv.mean():.4f}")
        print(f"验证 LQ: {['%.4f'%x for x in lqv]}  均值 {np.mean(lqv):.4f}")
        print(f"验证 GT: {['%.4f'%x for x in gtv]}  均值 {np.mean(gtv):.4f}")
        print(f"测试集中 > LQ 均值({np.mean(lqv):.4f}) 的: {int((mv>np.mean(lqv)).sum())} 张")
        print(f"测试集中 > GT 均值({np.mean(gtv):.4f}) 的: {int((mv>np.mean(gtv)).sum())} 张")
        print(f"测试集中 > 0.32 的: {int((mv>0.32).sum())} 张")
        hist, edges = np.histogram(mv, bins=np.arange(0.10, 0.62, 0.02))
        for h, e in zip(hist, edges):
            print(f"    [{e:.2f},{e+0.02:.2f})  {'#'*h} {h}")

    mv = np.array([M[f"test_case{i}"]["maniqa_full"] for i in range(1, 101)])
    hi = [i for i in range(1, 101) if M[f"test_case{i}"]["maniqa_full"] > 0.32]
    print()
    print(f"MANIQA_full > 0.32 的 case（{len(hi)} 张），按分数降序：")
    for i in sorted(hi, key=lambda j: -M[f"test_case{j}"]["maniqa_full"]):
        t = S[f"test_case{i}"]
        lab = L[i]
        tags = sorted(set(a[1].strip("\x00") for a in t.get("apps", [])))
        flag = []
        if i in has_exif: flag.append("EXIF")
        if i in has_mpf: flag.append("MPF")
        if i in diff_q: flag.append("粗量化表")
        print(f"  case{i:<4} MANIQA {M[f'test_case{i}']['maniqa_full']:.4f}  "
              f"{lab[0]:<11}{lab[1]:<6}{lab[2]:<9} "
              f"{t['filesize']/1e6:>5.2f}MB q0m={t.get('q0_mean',0):>5.2f} "
              f"psdHi={t['psd_e_gt025']*100:>5.2f}% lapv={t['lap_var']:>7.1f} "
              f"[{','.join(flag) if flag else '-'}]")

    print()
    print("=" * 100)
    print("D. 交叉验证：三批可疑集合的交并")
    print("=" * 100)
    A = set(hi)
    B = set(has_exif) | set(has_mpf)
    C = set(diff_q)
    big = set(i for i in range(1, 101) if S[f"test_case{i}"]["filesize"] > 4e6)
    print(f"A = MANIQA>0.32          : {len(A)} 张 {sorted(A)}")
    print(f"B = 带 EXIF 或 MPF        : {len(B)} 张 {sorted(B)}")
    print(f"C = 量化表≠验证集LQ(粗)   : {len(C)} 张 {sorted(C)}")
    print(f"E = 文件 > 4MB            : {len(big)} 张 {sorted(big)}")
    print(f"A∩B = {sorted(A & B)}")
    print(f"A∩C = {sorted(A & C)}")
    print(f"B∩C = {sorted(B & C)}")
    print(f"A∩E = {sorted(A & big)}")
    print(f"B 的 MANIQA: {['%.3f'%M[f'test_case{i}']['maniqa_full'] for i in sorted(B)]}")
    print(f"B 的 MANIQA 均值 {np.mean([M[f'test_case{i}']['maniqa_full'] for i in B]):.4f} "
          f"vs 其余 {np.mean([M[f'test_case{i}']['maniqa_full'] for i in range(1,101) if i not in B]):.4f}")
    print(f"C 的 MANIQA 均值 {np.mean([M[f'test_case{i}']['maniqa_full'] for i in C]):.4f} "
          f"vs 其余 {np.mean([M[f'test_case{i}']['maniqa_full'] for i in range(1,101) if i not in C]):.4f}")

    print()
    print("按内容类别的 MANIQA 均值（这决定了 3 张验证图的代表性）：")
    from collections import defaultdict
    g = defaultdict(list)
    for i in range(1, 101):
        g[L[i][0]].append(M[f"test_case{i}"]["maniqa_full"])
    for k, v in sorted(g.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:<12} n={len(v):>3}  mean {np.mean(v):.4f}  min {min(v):.4f} max {max(v):.4f}")
    g2 = defaultdict(list)
    for i in range(1, 101):
        g2[L[i][2]].append(M[f"test_case{i}"]["maniqa_full"])
    for k, v in sorted(g2.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  dist={k:<10} n={len(v):>3}  mean {np.mean(v):.4f}")
    g3 = defaultdict(list)
    for i in range(1, 101):
        g3[L[i][1]].append(M[f"test_case{i}"]["maniqa_full"])
    for k, v in sorted(g3.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  light={k:<9} n={len(v):>3}  mean {np.mean(v):.4f}")

print()
print("=" * 100)
print("E. 按内容类别的关键统计量（判断验证集覆盖了哪块）")
print("=" * 100)
from collections import defaultdict
gg = defaultdict(list)
for i in range(1, 101):
    gg[L[i][0]].append(S[f"test_case{i}"])
print(f"{'scene':<12}{'n':>4}{'lum_mean':>10}{'sat_mean':>10}{'grad_mean':>11}{'lap_var':>10}"
      f"{'psd>0.25%':>11}{'zero%':>8}{'255%':>8}")
for k, v in sorted(gg.items(), key=lambda kv: -len(kv[1])):
    print(f"{k:<12}{len(v):>4}" + f"{np.mean([x['lum_mean'] for x in v]):>10.1f}"
          f"{np.mean([x['sat_mean'] for x in v]):>10.1f}"
          f"{np.mean([x['grad_mean'] for x in v]):>11.2f}"
          f"{np.mean([x['lap_var'] for x in v]):>10.1f}"
          f"{np.mean([x['psd_e_gt025'] for x in v])*100:>11.2f}"
          f"{np.mean([x['frac_zero_any'] for x in v])*100:>8.2f}"
          f"{np.mean([x['frac_255_any'] for x in v])*100:>8.2f}")
print(f"{'VAL_LQ':<12}{3:>4}" + f"{np.mean([x['lum_mean'] for x in lq]):>10.1f}"
      f"{np.mean([x['sat_mean'] for x in lq]):>10.1f}"
      f"{np.mean([x['grad_mean'] for x in lq]):>11.2f}"
      f"{np.mean([x['lap_var'] for x in lq]):>10.1f}"
      f"{np.mean([x['psd_e_gt025'] for x in lq])*100:>11.2f}"
      f"{np.mean([x['frac_zero_any'] for x in lq])*100:>8.2f}"
      f"{np.mean([x['frac_255_any'] for x in lq])*100:>8.2f}")
print(f"{'VAL_GT':<12}{3:>4}" + f"{np.mean([x['lum_mean'] for x in gt]):>10.1f}"
      f"{np.mean([x['sat_mean'] for x in gt]):>10.1f}"
      f"{np.mean([x['grad_mean'] for x in gt]):>11.2f}"
      f"{np.mean([x['lap_var'] for x in gt]):>10.1f}"
      f"{np.mean([x['psd_e_gt025'] for x in gt])*100:>11.2f}"
      f"{np.mean([x['frac_zero_any'] for x in gt])*100:>8.2f}"
      f"{np.mean([x['frac_255_any'] for x in gt])*100:>8.2f}")

print()
print("=" * 100)
print("F. 逐张：psd_e_gt025（>0.25 cyc/px 的能量占比）—— 带限程度的直接代理")
print("=" * 100)
v = sorted(((S[f'test_case{i}']['psd_e_gt025'], i) for i in range(1, 101)), reverse=True)
print("最高 20 张（高频能量最多 = 最不像带限 LQ）：")
for x, i in v[:20]:
    print(f"  case{i:<4} {x*100:6.3f}%  {L[i][0]:<11}{L[i][1]:<6}{L[i][2]:<9} "
          f"{'EXIF' if i in has_exif else '':<5}{'粗表' if i in diff_q else ''}")
print("最低 10 张：")
for x, i in v[-10:]:
    print(f"  case{i:<4} {x*100:6.3f}%  {L[i][0]:<11}{L[i][1]:<6}{L[i][2]}")
print("验证集：")
for i in (1, 2, 3):
    print(f"  LQ{i} {S[f'val_case{i}_lq']['psd_e_gt025']*100:6.3f}%   "
          f"GT{i} {S[f'val_case{i}_gt']['psd_e_gt025']*100:6.3f}%   "
          f"比值 {S[f'val_case{i}_gt']['psd_e_gt025']/S[f'val_case{i}_lq']['psd_e_gt025']:.2f}x")
