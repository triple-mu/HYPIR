"""float64、抗混叠低通下的尺度分析：identity / affine / 单调1D曲线 三者的残差。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=250)
def gdown(a, s):
    b = cv2.GaussianBlur(a.astype(np.float32), (0,0), s, borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return b[s//2::s, s//2::s] if s>1 else b

def fit1d(x,y,nb=64):
    """分位数分箱的条件均值曲线，再线性插值。"""
    edges=np.quantile(x, np.linspace(0,1,nb+1)); edges=np.unique(edges)
    idx=np.clip(np.searchsorted(edges,x,'right')-1,0,len(edges)-2)
    cnt=np.bincount(idx,minlength=len(edges)-1)
    sx =np.bincount(idx,weights=x,minlength=len(edges)-1)
    sy =np.bincount(idx,weights=y,minlength=len(edges)-1)
    ok=cnt>=20
    cx,cy=sx[ok]/cnt[ok], sy[ok]/cnt[ok]
    return cx,cy
def apply1d(x,cx,cy): return np.interp(x,cx,cy)
def rms(e): return float(np.sqrt(np.mean(e**2)))

CH="RGB"
print(f"{'img':6s} {'s':>3s} {'ch':2s} {'identity':>9s} {'affine':>9s} {'1Dcurve':>9s} {'a':>8s} {'b':>8s} {'曲线额外解释%':>12s}")
STORE={}
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    for s in (1,4,8,16,32,64):
        Lg,Gg = gdown(L,s), gdown(G,s)
        for c in range(3):
            x,y=Lg[...,c].ravel(),Gg[...,c].ravel()
            if len(x)>4_000_000:
                sel=np.random.default_rng(0).choice(len(x),4_000_000,replace=False); x,y=x[sel],y[sel]
            a,b=np.polyfit(x,y,1)
            cx,cy=fit1d(x,y)
            r0,r1,r2 = rms(y-x), rms(y-(a*x+b)), rms(y-apply1d(x,cx,cy))
            ex = 100*(1-(r2/r0)**2)
            print(f"{k:6s} {s:3d} {CH[c]:2s} {r0:9.4f} {r1:9.4f} {r2:9.4f} {a:8.4f} {b:+8.3f} {ex:11.2f}%")
            if s in (1,32): STORE[(k,s,c)]=(cx,cy,r0,r1,r2,a,b)
    print()
np.save("scale_curves.npy", {f"{k}|{s}|{c}":v for (k,s,c),v in STORE.items()}, allow_pickle=True)
