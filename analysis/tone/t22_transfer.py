"""(a) case3 逐通道正向 LUT 在暗部的形状；(b) 色调校正的留一泛化测试。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=250)
def fwd_lut(G,L,c,minc=200):
    x=G[...,c].ravel(); y=L[...,c].ravel().astype(np.float64)
    cnt=np.bincount(x,minlength=256).astype(float); s=np.bincount(x,weights=y,minlength=256)
    return np.where(cnt>=minc, s/np.maximum(cnt,1), np.nan), cnt
print("=== case3 正向 E[LQ|GT=v] 暗部逐通道（原分辨率，含边条） ===")
L,G=imread(PAIRS["case3"][0]),imread(PAIRS["case3"][1])
print("GT v " + "".join(f"{v:>7d}" for v in range(6,40,2)))
for c,nm in enumerate("RGB"):
    m,cnt=fwd_lut(G,L,c)
    print(f" {nm}   " + "".join((f"{m[v]:>7.2f}" if cnt[v]>=200 else "      ·") for v in range(6,40,2)))
print("GT占比%" + "".join(f"{100*np.bincount(G[...,0].ravel(),minlength=256)[v]/G[...,0].size:>7.3f}" for v in range(6,40,2)))
print("\n去掉上下 80 行边条后重算：")
L2,G2=L[80:-80],G[80:-80]
print("GT v " + "".join(f"{v:>7d}" for v in range(6,40,2)))
for c,nm in enumerate("RGB"):
    m,cnt=fwd_lut(G2,L2,c)
    print(f" {nm}   " + "".join((f"{m[v]:>7.2f}" if cnt[v]>=200 else "      ·") for v in range(6,40,2)))

print("\n\n=== 留一泛化：色调校正能不能跨图迁移？ ===")
def fit_affine(L,G):
    return [np.polyfit(L[...,c].ravel().astype(np.float64),G[...,c].ravel().astype(np.float64),1) for c in range(3)]
def fit_lut(L,G):
    luts=[]
    for c in range(3):
        l=L[...,c].ravel(); g=G[...,c].ravel().astype(np.float64)
        cnt=np.bincount(l,minlength=256).astype(float); s=np.bincount(l,weights=g,minlength=256)
        m=np.where(cnt>=50,s/np.maximum(cnt,1),np.nan); idx=np.arange(256); ok=~np.isnan(m)
        m=np.maximum.accumulate(np.interp(idx,idx[ok],m[ok])); luts.append(np.clip(m,0,255))
    return luts
def psnr(a,b): return 10*np.log10(255**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
D={k:(imread(v[0]),imread(v[1])) for k,v in PAIRS.items()}
AB={k:fit_affine(*D[k]) for k in D}; LU={k:fit_lut(*D[k]) for k in D}
print(f"{'施加于':8s} {'校正来源':22s} {'PSNR':>8s} {'ΔPSNR':>8s}")
for k in D:
    L,G=D[k]; base=psnr(L,G)
    print(f"{k:8s} {'无校正':22s} {base:8.4f} {0.0:8.4f}")
    for src in D:
        ab=AB[src]; o=np.clip(np.stack([ab[c][0]*L[...,c]+ab[c][1] for c in range(3)],-1),0,255)
        tag = "自身(oracle)" if src==k else f"来自 {src}"
        print(f"{'':8s} {'仿射 '+tag:22s} {psnr(o,G):8.4f} {psnr(o,G)-base:+8.4f}")
    for src in D:
        lu=LU[src]; o=np.stack([lu[c][L[...,c]] for c in range(3)],-1)
        tag = "自身(oracle)" if src==k else f"来自 {src}"
        print(f"{'':8s} {'LUT '+tag:22s} {psnr(o,G):8.4f} {psnr(o,G)-base:+8.4f}")
    # 三图平均仿射（留一：只用另外两张）
    others=[s for s in D if s!=k]
    a=np.mean([[AB[s][c][0] for c in range(3)] for s in others],0); b=np.mean([[AB[s][c][1] for c in range(3)] for s in others],0)
    o=np.clip(np.stack([a[c]*L[...,c]+b[c] for c in range(3)],-1),0,255)
    print(f"{'':8s} {'仿射 留一均值':22s} {psnr(o,G):8.4f} {psnr(o,G)-base:+8.4f}   (a={a.round(4)} b={b.round(3)})")
