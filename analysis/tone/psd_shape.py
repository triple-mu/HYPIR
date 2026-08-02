"""径向 PSD 形状：区分「带限（2x 下上采样）」与「自然滚降」。

带限图在 f=0.25 cyc/px 处应有一个陡崖；自然图是平滑幂律。
用 BLI = log10( P(0.20~0.24) / P(0.28~0.32) ) 度量崖的陡度。
再看 P(0.44~0.50)/P(0.20~0.24) 判断是否存在噪声底（噪声底会把崖填平）。
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
freq = (np.arange(128) + 0.5) / 128 * 0.5


def band(p, lo, hi):
    m = (freq >= lo) & (freq < hi)
    return float(np.mean(np.asarray(p)[m]))


def feats(r):
    p = r["psd_prof"]
    a = band(p, 0.20, 0.24)
    b = band(p, 0.28, 0.32)
    c = band(p, 0.44, 0.50)
    d = band(p, 0.10, 0.14)
    return dict(bli=np.log10(a / max(b, 1e-30)),
                tail=np.log10(c / max(a, 1e-30)),
                slope_lo=np.log10(d / max(a, 1e-30)))


NATIVE = [1, 2, 3, 4, 5, 38, 64, 65, 66]
print("BLI = log10(P[0.20-0.24]/P[0.28-0.32])  越大 = 0.25 处崖越陡 = 越像带限")
print("tail= log10(P[0.44-0.50]/P[0.20-0.24])  越大 = 高频尾巴越平 = 有噪声底/伪影")
print("slope_lo = log10(P[0.10-0.14]/P[0.20-0.24])  自然滚降参考斜率")
print()
print(f"{'name':<16}{'BLI':>8}{'tail':>8}{'slope_lo':>10}   scene")
print("-- 验证集 --")
for i in (1, 2, 3):
    for k in ("lq", "gt"):
        f = feats(S[f"val_case{i}_{k}"])
        print(f"val{i}_{k:<12}{f['bli']:>8.2f}{f['tail']:>8.2f}{f['slope_lo']:>10.2f}")
print("-- 测试集：9 张相机原生 --")
for i in NATIVE:
    f = feats(S[f"test_case{i}"])
    print(f"case{i:<12}{f['bli']:>8.2f}{f['tail']:>8.2f}{f['slope_lo']:>10.2f}   {L[i][0]},{L[i][1]}")
print("-- 测试集：91 张重编码，BLI 分布 --")
rest = [feats(S[f"test_case{i}"]) for i in range(1, 101) if i not in NATIVE]
for k in ("bli", "tail", "slope_lo"):
    v = np.array([x[k] for x in rest])
    print(f"  {k:<9} min {v.min():6.2f} p10 {np.percentile(v,10):6.2f} p50 {np.percentile(v,50):6.2f} "
          f"p90 {np.percentile(v,90):6.2f} max {v.max():6.2f}")
nv = np.array([feats(S[f'test_case{i}'])['bli'] for i in NATIVE])
rv = np.array([x["bli"] for x in rest])
print(f"\nBLI: 原生 9 张 均值 {nv.mean():.2f} (范围 {nv.min():.2f}~{nv.max():.2f})")
print(f"BLI: 重编码 91 张 均值 {rv.mean():.2f} (范围 {rv.min():.2f}~{rv.max():.2f})")
print(f"BLI: 验证 LQ 均值 {np.mean([feats(S[f'val_case{i}_lq'])['bli'] for i in (1,2,3)]):.2f}, "
      f"GT 均值 {np.mean([feats(S[f'val_case{i}_gt'])['bli'] for i in (1,2,3)]):.2f}")
print(f"重编码 91 张中 BLI < 原生最大值({nv.max():.2f}) 的比例: "
      f"{(rv < nv.max()).mean()*100:.0f}%")

print()
print("=== 逐张 BLI 排序（低 = 不带限）===")
allv = sorted(((feats(S[f"test_case{i}"])["bli"], i) for i in range(1, 101)))
print("最低 15 张:")
for x, i in allv[:15]:
    print(f"  case{i:<4}{x:7.2f}  {'原生' if i in NATIVE else '重编码'}  {L[i][0]},{L[i][1]},{L[i][2]}")
print("最高 8 张:")
for x, i in allv[-8:]:
    print(f"  case{i:<4}{x:7.2f}  {'原生' if i in NATIVE else '重编码'}  {L[i][0]},{L[i][1]},{L[i][2]}")

print()
print("=== 归一化径向 PSD 曲线（以 f=0.125 处为 0 dB），单位 dB ===")
fs = [0.125, 0.15, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.45, 0.49]
print(f"{'name':<14}" + "".join(f"{f:>7.3f}" for f in fs))
def curve(r):
    p = r["psd_prof"]
    ref = band(p, 0.115, 0.135)
    return [10 * np.log10(max(band(p, f - 0.008, f + 0.008), 1e-30) / ref) for f in fs]
for nm in ["val_case1_lq", "val_case1_gt", "val_case2_lq", "val_case2_gt",
           "val_case3_lq", "val_case3_gt"]:
    print(f"{nm:<14}" + "".join(f"{x:>7.1f}" for x in curve(S[nm])))
for i in NATIVE:
    print(f"{'case'+str(i)+'*':<14}" + "".join(f"{x:>7.1f}" for x in curve(S[f"test_case{i}"])))
for i in [8, 13, 32, 54, 76, 99, 56, 100]:
    print(f"{'case'+str(i):<14}" + "".join(f"{x:>7.1f}" for x in curve(S[f"test_case{i}"])))
