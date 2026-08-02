"""亚像素配准搜索：LQ 平移 (dx,dy) 后与 GT 的 RMSE。"""
import numpy as np, cv2
from common import *
def shift(a, dx, dy):
    M=np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(a, M, (a.shape[1],a.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
for k,(lp,gp) in PAIRS.items():
    L=cv2.cvtColor(imread(lp),cv2.COLOR_RGB2GRAY).astype(np.float32)
    G=cv2.cvtColor(imread(gp),cv2.COLOR_RGB2GRAY).astype(np.float32)
    h,w=L.shape; cy,cx=h//2,w//2; S=1536
    l=L[cy-S//2:cy+S//2, cx-S//2:cx+S//2]; g=G[cy-S//2:cy+S//2, cx-S//2:cx+S//2]
    best=None; grid={}
    for dy in np.arange(-1.0,1.01,0.25):
        for dx in np.arange(-1.0,1.01,0.25):
            r=float(np.sqrt(((shift(l,dx,dy)[40:-40,40:-40]-g[40:-40,40:-40])**2).mean()))
            grid[(round(dx,2),round(dy,2))]=r
            if best is None or r<best[0]: best=(r,dx,dy)
    r00=grid[(0.0,0.0)]
    print(f"{k}: RMSE(0,0)={r00:.4f}  最优 (dx={best[1]:+.2f}, dy={best[2]:+.2f}) RMSE={best[0]:.4f}  改善 {r00-best[0]:+.4f}")
    # 高频带相关性：LQ 与 GT 各自减去 s=4 低通后的残差之间的相关
    def hp(a,s=4): return a-cv2.GaussianBlur(a,(0,0),s,borderType=cv2.BORDER_REFLECT)
    hl,hg=hp(l),hp(g)
    print(f"    高频带(σ=4 以上): std(LQ_hf)={hl.std():.3f} std(GT_hf)={hg.std():.3f} corr={np.corrcoef(hl.ravel(),hg.ravel())[0,1]:.4f}")
    # LQ 高频里有多少是 GT 里没有的
    a=np.corrcoef(hl.ravel(),hg.ravel())[0,1]
    print(f"    LQ 高频中与 GT 无关的能量占比 = {100*(1-a**2):.2f}%  -> std(n)≈{hl.std()*np.sqrt(1-a**2):.2f}")
