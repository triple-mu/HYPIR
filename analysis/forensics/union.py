import json, numpy as np
d = json.load(open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/spec.json"))
E = np.array(d["edges"]); S = {k: np.array(v) for k,v in d["spec"].items()}
ctr = (E[:-1]+E[1:])/2
B = json.load(open("/home/ubuntu/workspace/contest/CSIG-2026/verdict/blocky.json"))
NAT = {1,2,3,4,5,38,64,65,66}
# 复现 report6 的频带（128-DCT ring 6-14 / 1-3 -> cyc/px）
hfr = lambda s: s[(ctr>=0.023)&(ctr<0.055)].mean()/s[(ctr>=0.003)&(ctr<0.012)].mean()
blk = lambda i: max(B[f"test{i}"])
H = {i: hfr(S[f"test{i}"]) for i in range(1,101)}
print("val: " + "  ".join(f"c{c} LQ={hfr(S[f'val{c}_lq']):.4f}/GT={hfr(S[f'val{c}_gt']):.4f}" for c in (1,2,3)))
nat = sorted((H[i], i) for i in NAT); oth = sorted((H[i], i) for i in range(1,101) if i not in NAT)
print("9 原生 HFR: " + " ".join(f"c{i}:{v:.4f}" for v,i in nat))
print(f"91 重编 HFR: p50={oth[45][0]:.4f} p90={oth[81][0]:.4f} max={oth[-1][0]:.4f}(c{oth[-1][1]})  次高 c{oth[-2][1]}:{oth[-2][0]:.4f}")
for th in (0.10,0.13,0.15,0.20):
    print(f"  HFR>={th}: 命中{sum(1 for v,_ in nat if v>=th)}/9 误报{sum(1 for v,_ in oth if v>=th)}")
print()
for hth in (0.10,0.13,0.15):
    for bth in (2.5,3.0,4.0):
        hit = {i for i in NAT if H[i]>=hth or blk(i)>=bth}
        fp  = {i for i in range(1,101) if i not in NAT and (H[i]>=hth or blk(i)>=bth)}
        print(f"  并集 HFR>={hth} or blocky>={bth}: 命中 {len(hit)}/9 {sorted(hit)}  误报 {len(fp)} {sorted(fp)}")
