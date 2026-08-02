import numpy as np, cv2
from common import *

def cond_curve(lq, gt):
    """E[GT|LQ=v] and median[GT|LQ=v] and count, for v=0..255."""
    l = lq.ravel().astype(np.int64); g = gt.ravel().astype(np.float64)
    cnt = np.bincount(l, minlength=256).astype(np.float64)
    s   = np.bincount(l, weights=g, minlength=256)
    mean = np.where(cnt>0, s/np.maximum(cnt,1), np.nan)
    # median via per-bin 2D histogram
    H = np.zeros((256,256), np.int64)
    np.add.at(H, (l, gt.ravel().astype(np.int64)), 1)
    cs = H.cumsum(1); tot = cs[:,-1:]
    med = np.full(256, np.nan)
    ok = tot[:,0]>0
    med[ok] = np.argmax(cs[ok] >= tot[ok]*0.5, axis=1)
    return mean, med, cnt

def affine(lq, gt):
    x = lq.ravel().astype(np.float64); y = gt.ravel().astype(np.float64)
    a,b = np.polyfit(x,y,1); return a,b

def down(a, f):
    return cv2.resize(a, (a.shape[1]//f, a.shape[0]//f), interpolation=cv2.INTER_AREA)

np.set_printoptions(suppress=True, linewidth=250)
CH = "RGB"
res = {}
for k,(lp,gp) in PAIRS.items():
    L, G = imread(lp), imread(gp)
    Ld, Gd = down(L,8).astype(np.float64), down(G,8).astype(np.float64)
    for c in range(3):
        mean, med, cnt = cond_curve(L[...,c], G[...,c])
        a,b = affine(L[...,c], G[...,c])
        # blur-matched affine (removes regression dilution from bandlimit loss)
        ad,bd = np.polyfit(Ld[...,c].ravel(), Gd[...,c].ravel(), 1)
        r = np.corrcoef(Ld[...,c].ravel(), Gd[...,c].ravel())[0,1]
        res[(k,c)] = (mean, med, cnt, a, b, ad, bd, r)
        print(f"{k} {CH[c]}: fullres affine a={a:.4f} b={b:+.3f} | 8x-down affine a={ad:.4f} b={bd:+.3f} corr={r:.6f}")
np.save("curves.npy", {str(kk):(v[0],v[1],v[2]) for kk,v in res.items()}, allow_pickle=True)
print()
print("=== E[GT|LQ=v] minus v, sampled ===")
vs = [0,1,2,4,8,16,32,48,64,96,128,160,192,224,240,248,252,254,255]
print("v      ", "".join(f"{v:>7d}" for v in vs))
for k in PAIRS:
    for c in range(3):
        mean = res[(k,c)][0]; cnt = res[(k,c)][2]
        print(f"{k} {CH[c]}", "".join((f"{mean[v]-v:>7.2f}" if cnt[v]>50 else "      -") for v in vs))
print()
print("=== count of LQ pixels at each v (fraction %) ===")
for k in PAIRS:
    for c in range(3):
        cnt = res[(k,c)][2]; n=cnt.sum()
        print(f"{k} {CH[c]}", "".join(f"{100*cnt[v]/n:>7.3f}" for v in vs))
