"""黑位/白位压死统计 + 压死像素在 GT 里的真实值分布 + 空间聚集性。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=200)
CH="RGB"
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    print(f"\n########## {k} ##########")
    print(f"{'ch':3s} {'LQ==0%':>8s} {'GT==0%':>8s} {'LQ<=2%':>8s} {'GT<=2%':>8s} {'LQ==255%':>9s} {'GT==255%':>9s} {'LQ>=253%':>9s} {'GT>=253%':>9s}")
    for c in range(3):
        l,g=L[...,c],G[...,c]; n=l.size
        print(f"{CH[c]:3s} {100*(l==0).mean():8.3f} {100*(g==0).mean():8.3f} {100*(l<=2).mean():8.3f} {100*(g<=2).mean():8.3f} "
              f"{100*(l==255).mean():9.3f} {100*(g==255).mean():9.3f} {100*(l>=253).mean():9.3f} {100*(g>=253).mean():9.3f}")
    # 三通道同时为 0 = 全黑像素
    allz_l=((L==0).all(2)); allz_g=((G==0).all(2))
    print(f"三通道全 0: LQ {100*allz_l.mean():.3f}%  GT {100*allz_g.mean():.3f}%")
    # 压死像素在 GT 里的值分布
    for c in range(3):
        m = L[...,c]==0
        if m.sum()<100: continue
        gv=G[...,c][m]
        print(f"  LQ {CH[c]}==0 的 {m.sum()} 个像素 -> GT 同位置 {CH[c]}: mean={gv.mean():.2f} "
              f"pct[1,10,50,90,99]={np.percentile(gv,[1,10,50,90,99]).astype(int)} frac(GT<=2)={100*(gv<=2).mean():.2f}%")
        # 空间聚集性：压死掩膜的连通性 vs 同密度随机
        mm=m.astype(np.uint8)
        nlab,lab,stats,_=cv2.connectedComponentsWithStats(mm,8)
        areas=stats[1:,4]
        print(f"       连通域 {nlab-1} 个, 最大 {areas.max() if len(areas) else 0} px, "
              f"top5 面积 {np.sort(areas)[-5:][::-1] if len(areas)>=5 else areas}, "
              f"占压死总数 {100*np.sort(areas)[-5:].sum()/m.sum():.1f}%")
        # 局部密度：8x8 block 内压死比例的分布
        h,w=m.shape; B=64
        blk=m[:h//B*B,:w//B*B].reshape(h//B,B,w//B,B).mean((1,3))
        print(f"       64x64 块压死率: mean={blk.mean()*100:.2f}% frac(块全压死)={100*(blk>0.99).mean():.3f}% frac(块无压死)={100*(blk<1e-9).mean():.2f}%")
