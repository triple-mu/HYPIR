import numpy as np, cv2
from common import *
def down(a,f): return cv2.resize(a,(a.shape[1]//f,a.shape[0]//f),interpolation=cv2.INTER_AREA)

for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    Ld,Gd=down(L,4).astype(np.float32),down(G,4).astype(np.float32)
    ly=Ld.mean(2); gy=Gd.mean(2)
    best=None
    for dy in range(-4,5):
        for dx in range(-4,5):
            a=ly[8+dy:ly.shape[0]-8+dy, 8+dx:ly.shape[1]-8+dx]
            b=gy[8:gy.shape[0]-8, 8:gy.shape[1]-8]
            r=float(np.sqrt(((a-b)**2).mean()))
            if best is None or r<best[0]: best=(r,dx,dy)
    r0=float(np.sqrt((( ly[8:-8,8:-8]-gy[8:-8,8:-8])**2).mean()))
    print(f"{k}: 4x域 RMSE(shift0)={r0:.3f}  best shift dx={best[1]} dy={best[2]} RMSE={best[0]:.3f}  (即原图 {best[1]*4},{best[2]*4} px)")
    # 逐 tile 的 RMSE 分布，看差异是全局还是局部
    T=32; hh,ww=ly.shape
    tl=[]
    for i in range(0,hh-T,T):
        for j in range(0,ww-T,T):
            tl.append(np.sqrt(((ly[i:i+T,j:j+T]-gy[i:i+T,j:j+T])**2).mean()))
    tl=np.array(tl)
    print(f"    tile RMSE: p10={np.percentile(tl,10):.2f} p50={np.percentile(tl,50):.2f} p90={np.percentile(tl,90):.2f} max={tl.max():.2f}")
    # 亮度直方图对比
    for c,nm in enumerate("RGB"):
        lo=np.percentile(L[...,c],[0.1,1,50,99,99.9]); go=np.percentile(G[...,c],[0.1,1,50,99,99.9])
        print(f"    {nm} LQ pct(0.1/1/50/99/99.9)={lo} GT={go}")
