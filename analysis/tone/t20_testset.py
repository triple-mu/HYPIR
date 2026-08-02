"""全部 100 张测试图的黑位/白位/色偏统计，按 原生 vs 重编码 两组对比。"""
import numpy as np, cv2, glob, os, json
TEST="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
NATIVE={1,2,3,4,5,38,64,65,66}
rows={}
for p in sorted(glob.glob(TEST+"/*.jpg"), key=lambda x:int(''.join(filter(str.isdigit,os.path.basename(x))))):
    n=int(''.join(filter(str.isdigit,os.path.basename(p))))
    a=cv2.cvtColor(cv2.imread(p,cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    f=a.reshape(-1,3)
    rows[n]=dict(
        native=n in NATIVE, shape=a.shape[:2],
        z=[float(100*(f[:,c]==0).mean()) for c in range(3)],
        s=[float(100*(f[:,c]>=255).mean()) for c in range(3)],
        mn=[int(f[:,c].min()) for c in range(3)], mx=[int(f[:,c].max()) for c in range(3)],
        mean=[float(f[:,c].mean()) for c in range(3)], std=[float(f[:,c].std()) for c in range(3)],
        p01=[float(np.percentile(f[:,c],0.1)) for c in range(3)],
    )
json.dump(rows, open("testset_stats.json","w"))
nat=[v for v in rows.values() if v['native']]; rec=[v for v in rows.values() if not v['native']]
def agg(g,key,f=np.mean):
    return np.array([v[key] for v in g])
print(f"{'组':10s} {'n':>3s} {'零值R/G/B %':>26s} {'饱和R/G/B %':>24s} {'最小值':>14s} {'gray-world R/G,B/G':>22s}")
for nm,g in (("原生(9)",nat),("重编码(91)",rec)):
    z=agg(g,'z'); s=agg(g,'s'); mn=agg(g,'mn'); mu=agg(g,'mean')
    print(f"{nm:10s} {len(g):3d} {str(np.round(z.mean(0),3)):>26s} {str(np.round(s.mean(0),4)):>24s} "
          f"{str(np.round(mn.mean(0),2)):>14s} {(mu[:,0]/mu[:,1]).mean():.4f},{(mu[:,2]/mu[:,1]).mean():.4f}")
print()
print("零值率(三通道最大) 分布:")
for nm,g in (("原生",nat),("重编码",rec)):
    v=np.array([max(x['z']) for x in g])
    print(f"  {nm:6s} n={len(v)} min={v.min():.3f} p25={np.percentile(v,25):.3f} 中位={np.median(v):.3f} p75={np.percentile(v,75):.3f} max={v.max():.3f} 均值={v.mean():.3f}")
    print(f"          零值率>1% 的张数: {(v>1).sum()}/{len(v)}   >0.1%: {(v>0.1).sum()}/{len(v)}")
print("\n最小值(三通道最大的那个最小值) —— 有黑位抬升的图 min>0:")
for nm,g in (("原生",nat),("重编码",rec)):
    v=np.array([max(x['mn']) for x in g]); print(f"  {nm:6s} min>0 的张数 {(v>0).sum()}/{len(v)}, 取值 {sorted(set(v.tolist()))[:15]}")
print("\n逐张（原生 9 张）：")
for n in sorted(NATIVE):
    v=rows[n]; print(f"  case{n:<3d} 零值={np.round(v['z'],3)} 饱和={np.round(v['s'],4)} min={v['mn']} max={v['mx']} std={np.round(v['std'],1)}")
print("\n重编码组零值率最高的 10 张：")
for n,_ in sorted(((n,max(v['z'])) for n,v in rows.items() if not v['native']), key=lambda x:-x[1])[:10]:
    v=rows[n]; print(f"  case{n:<3d} 零值={np.round(v['z'],3)} min={v['mn']} max={v['mx']} std={np.round(v['std'],1)}")
