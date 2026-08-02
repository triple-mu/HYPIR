import json, numpy as np
d = json.load(open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/spec.json"))
E = np.array(d["edges"]); S = {k: np.array(v) for k, v in d["spec"].items()}
ctr = (E[:-1] + E[1:]) / 2
NAT = {1,2,3,4,5,38,64,65,66}

def ratio(s, lo1, hi1, lo2, hi2):
    a = s[(ctr>=lo1)&(ctr<hi1)].mean(); b = s[(ctr>=lo2)&(ctr<hi2)].mean()
    return a/b

best = []
grid = [0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40]
for i in range(len(grid)-1):
  for j in range(i+1, len(grid)):
    for k in range(len(grid)-1):
      for l in range(k+1, len(grid)):
        if grid[k] < grid[j]: continue
        nat = [ratio(S[f"test{n}"], grid[k],grid[l], grid[i],grid[j]) for n in sorted(NAT)]
        oth = [ratio(S[f"test{n}"], grid[k],grid[l], grid[i],grid[j]) for n in range(1,101) if n not in NAT]
        margin = min(nat)/max(oth)
        best.append((margin, grid[k],grid[l],grid[i],grid[j], min(nat), max(oth)))
best.sort(reverse=True)
print("按「9张最小值 / 91张最大值」排序的最佳频带组合:")
for m,a,b,c,e,mn,mx in best[:8]:
    print(f"  margin={m:6.3f}x   高带[{a:.2f},{b:.2f})/低带[{c:.2f},{e:.2f})   nat_min={mn:.5f}  rec_max={mx:.5f}")

m,a,b,c,e,mn,mx = best[0]
print(f"\n=== 采用 HFR = E[{a},{b}) / E[{c},{e}) ===")
for n in (1,2,3):
    print(f"  val{n}: LQ={ratio(S[f'val{n}_lq'],a,b,c,e):.5f}  GT={ratio(S[f'val{n}_gt'],a,b,c,e):.5f}")
nat = sorted((ratio(S[f"test{n}"],a,b,c,e), n) for n in NAT)
oth = sorted((ratio(S[f"test{n}"],a,b,c,e), n) for n in range(1,101) if n not in NAT)
print("  9 原生: " + " ".join(f"c{n}:{v:.4f}" for v,n in nat))
print(f"  91 重编: min={oth[0][0]:.5f} p50={oth[45][0]:.5f} p90={oth[81][0]:.5f} max={oth[-1][0]:.5f}(c{oth[-1][1]})")
print("  91 中最高 6: " + " ".join(f"c{n}:{v:.4f}" for v,n in oth[-6:][::-1]))
