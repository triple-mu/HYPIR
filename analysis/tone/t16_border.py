import numpy as np
from common import *
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    print(f"\n### {k} {L.shape} ###")
    for nm,A in (("LQ",L),("GT",G)):
        rm=A.reshape(A.shape[0],-1).mean(1); cm=A.transpose(1,0,2).reshape(A.shape[1],-1).mean(1)
        # 找上下/左右恒定暗边
        def runlen(v, thr):
            i=0
            while i<len(v) and v[i]<thr: i+=1
            j=len(v)-1
            while j>=0 and v[j]<thr: j-=1
            return i, len(v)-1-j
        t,b=runlen(rm,12); l_,r=runlen(cm,12)
        print(f" {nm}: 行均值 首8={rm[:8].round(2)} 末8={rm[-8:].round(2)}")
        print(f"     暗边(行均值<12): top={t} bottom={b} ; 列 left={l_} right={r}")
    # 边条区域内 LQ/GT 值
    for band,sl in (("top40", np.s_[:40,:]), ("bot40", np.s_[-40:,:])):
        lv=L[sl]; gv=G[sl]
        print(f"  {band}: LQ mean={lv.mean(axis=(0,1)).round(2)} max={lv.max(axis=(0,1))} frac0={[round(100*(lv[...,c]==0).mean(),2) for c in range(3)]}")
        print(f"  {band}: GT mean={gv.mean(axis=(0,1)).round(2)} max={gv.max(axis=(0,1))} frac0={[round(100*(gv[...,c]==0).mean(),2) for c in range(3)]}")
    # 排除边条后重算压死率
    if k=="case3":
        for pad in (0,40,80,120):
            core_l=L[pad:L.shape[0]-pad]; core_g=G[pad:G.shape[0]-pad]
            print(f"  去掉上下 {pad} 行后: LQ frac0={[round(100*(core_l[...,c]==0).mean(),3) for c in range(3)]} "
                  f"GT frac0={[round(100*(core_g[...,c]==0).mean(),3) for c in range(3)]} GT min={core_g.min(axis=(0,1))}")
