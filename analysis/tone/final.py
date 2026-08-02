"""收尾：MANIQA 与真实锐度的相关性、验证集代表性的量化、逐张总表。"""
import json, sys
import numpy as np
from scipy import stats as st
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L, VAL

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
M = {r["name"]: r for r in json.load(open(f"{D}/maniqa_all.json"))}
H = json.load(open(f"{D}/hfr2.json"))
hfr = {int(k): v for k, v in H["test"].items()}
hval = H["val"]
NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}

mn = np.array([M[f"test_case{i}"]["maniqa_full"] for i in range(1, 101)])
hf = np.array([hfr[i] for i in range(1, 101)])
lv = np.array([S[f"test_case{i}"]["lap_var"] for i in range(1, 101)])
gm = np.array([S[f"test_case{i}"]["grad_mean"] for i in range(1, 101)])
sm = np.array([S[f"test_case{i}"]["sat_mean"] for i in range(1, 101)])
lm = np.array([S[f"test_case{i}"]["lum_mean"] for i in range(1, 101)])

print("=" * 92)
print("MANIQA 到底在测什么？（100 张测试图上的 Spearman 相关）")
print("=" * 92)
for nm, v in [("HFR 纹理条件锐度", hf), ("lap_var", lv), ("grad_mean", gm),
              ("sat_mean 饱和度", sm), ("lum_mean 亮度", lm)]:
    r = st.spearmanr(mn, v)
    print(f"  MANIQA vs {nm:<20} rho = {r.statistic:+.3f}  p = {r.pvalue:.2g}")
veg = np.array([1.0 if L[i][0] == "vegetation" else 0.0 for i in range(1, 101)])
bld = np.array([1.0 if L[i][0] == "building" else 0.0 for i in range(1, 101)])
print(f"  MANIQA vs 是否植被            rho = {st.spearmanr(mn, veg).statistic:+.3f}"
      f"  p = {st.spearmanr(mn, veg).pvalue:.2g}")
print(f"  MANIQA vs 是否建筑            rho = {st.spearmanr(mn, bld).statistic:+.3f}"
      f"  p = {st.spearmanr(mn, bld).pvalue:.2g}")
print()
print("  反例（MANIQA 高但实际极糊）:")
for i in [25, 24, 11, 29, 82, 81]:
    print(f"    case{i:<4} MANIQA {M[f'test_case{i}']['maniqa_full']:.4f}  "
          f"lap_var {S[f'test_case{i}']['lap_var']:>6.1f} (全集 p50={np.median(lv):.1f})  "
          f"HFR {hfr[i]:.5f} (全集 p50={np.median(hf):.5f})  {L[i][0]},{L[i][1]}")
print("  反例（确定未退化但 MANIQA 低）:")
for i in [5, 1, 2, 65]:
    print(f"    case{i:<4} MANIQA {M[f'test_case{i}']['maniqa_full']:.4f}  "
          f"lap_var {S[f'test_case{i}']['lap_var']:>6.1f}  HFR {hfr[i]:.5f}  相机原生")

print()
print("=" * 92)
print("验证集 3 张 LQ 在测试集分布中的位置（百分位）—— 代表性风险量化")
print("=" * 92)
KEYS = [("lum_mean", "亮度均值"), ("lum_std", "亮度std"), ("sat_mean", "饱和度"),
        ("grad_mean", "梯度能量"), ("lap_var", "拉普拉斯方差"),
        ("frac_zero_any", "暗部0值占比"), ("frac_255_any", "亮部截断占比")]
print(f"{'指标':<14}{'测试集p50':>12}{'LQ1 pct':>10}{'LQ2 pct':>10}{'LQ3 pct':>10}   覆盖判断")
for k, nm in KEYS:
    v = np.array([S[f"test_case{i}"][k] for i in range(1, 101)])
    ps = [100 * float((v < S[f"val_case{i}_lq"][k]).mean()) for i in (1, 2, 3)]
    span = max(ps) - min(ps)
    cov = "覆盖窄" if span < 40 else "覆盖中"
    if max(ps) < 40: cov += "，全部偏低端"
    if min(ps) > 60: cov += "，全部偏高端"
    print(f"{nm:<14}{np.median(v):>12.4g}" + "".join(f"{p:>10.0f}" for p in ps) + f"   {cov}")
print()
print("HFR（纹理条件锐度）：")
hlq = [hval[f"{i}lq"] for i in (1, 2, 3)]
for j, i in enumerate((1, 2, 3)):
    print(f"  val{i}_lq HFR {hlq[j]:.5f}  在测试集 91 张重编码中的百分位 "
          f"{100*float(np.mean([hfr[t] < hlq[j] for t in range(1,101) if t not in NATIVE])):.0f}")
print()
print("暗部压死（FINDINGS §4 的对比收缩结论来自 case3）：")
z = np.array([S[f"test_case{i}"]["frac_zero_any"] for i in range(1, 101)])
print(f"  val3_lq frac_zero_any = {S['val_case3_lq']['frac_zero_any']*100:.2f}%")
print(f"  测试集 100 张: p50 {np.median(z)*100:.3f}%  p90 {np.percentile(z,90)*100:.2f}%  "
      f"max {z.max()*100:.2f}% (case{int(np.argmax(z))+1})")
print(f"  测试集里 frac_zero_any > 2% 的只有 {int((z>0.02).sum())} 张: "
      f"{[i for i in range(1,101) if z[i-1]>0.02]}")
print(f"  -> val3_lq 的 {S['val_case3_lq']['frac_zero_any']*100:.2f}% 超过测试集 100 张的最大值 "
      f"{z.max()*100:.2f}%，是分布外样本")

print()
print("=" * 92)
print("测试集内容分布 vs 验证集覆盖")
print("=" * 92)
from collections import Counter
cs = Counter(v[0] for v in L.values())
vs = Counter(VAL[i][0] for i in (1, 2, 3))
print(f"{'场景':<12}{'测试集张数':>10}{'占比':>8}{'验证集张数':>12}   MANIQA均值")
for k, n in cs.most_common():
    mm = np.mean([M[f"test_case{i}"]["maniqa_full"] for i in range(1, 101) if L[i][0] == k])
    print(f"{k:<12}{n:>10}{n:>7}%{vs.get(k,0):>12}   {mm:.4f}"
          + ("   <-- 验证集零覆盖" if vs.get(k, 0) == 0 else ""))
print()
cd = Counter(v[2] for v in L.values())
vd = Counter(VAL[i][2] for i in (1, 2, 3))
for k, n in cd.most_common():
    print(f"距离 {k:<10}{n:>6} 张  验证集 {vd.get(k,0)} 张"
          + ("   <-- 验证集零覆盖" if vd.get(k, 0) == 0 else ""))
cl = Counter(v[1] for v in L.values())
vl = Counter(VAL[i][1] for i in (1, 2, 3))
print()
for k, n in cl.most_common():
    print(f"光照 {k:<10}{n:>6} 张  验证集 {vl.get(k,0)} 张")

print()
print("=" * 92)
print("最终逐张判定表（未退化嫌疑）")
print("=" * 92)
gtmin = min(hval[f"{i}gt"] for i in (1, 2, 3))
lqmax = max(hval[f"{i}lq"] for i in (1, 2, 3))
print(f"参照：验证集 LQ HFR 区间 [{min(hlq):.5f}, {lqmax:.5f}]，"
      f"GT HFR 区间 [{gtmin:.5f}, {max(hval[f'{i}gt'] for i in (1,2,3)):.5f}]")
print(f"{'case':<8}{'HFR':>9}{'MANIQA':>9}{'lap_var':>9}{'文件MB':>8}{'量化表':>8}{'EXIF/MPF':>10}  判定")
for i in sorted(range(1, 101), key=lambda j: -hfr[j])[:14]:
    r = S[f"test_case{i}"]
    v = "未退化(相机原生)" if i in NATIVE else ("可疑" if hfr[i] > lqmax else "已退化")
    print(f"case{i:<4}{hfr[i]:>9.5f}{M[f'test_case{i}']['maniqa_full']:>9.4f}"
          f"{r['lap_var']:>9.1f}{r['filesize']/1e6:>8.2f}{r.get('q0_mean',0):>8.2f}"
          f"{'有' if i in NATIVE else '无':>10}  {v}  {L[i][0]},{L[i][1]}")
