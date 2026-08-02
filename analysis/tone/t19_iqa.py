"""任务6：用 TOPIQ-FR + MANIQA 量化色调/色彩校正的收益上限。"""
import sys, numpy as np, cv2, torch, json
sys.path.insert(0,"/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
from iqa import score
sys.path.insert(0,"/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from common import *

M=np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
def rgb2ycc(a): o=a.astype(np.float64)@M.T; o[...,1:]+=128; return o
def ycc2rgb(a): a=a.astype(np.float64).copy(); a[...,1:]-=128; return a@np.linalg.inv(M).T

def fit_affine(L,G):
    out=[]
    for c in range(3):
        a,b=np.polyfit(L[...,c].ravel().astype(np.float64), G[...,c].ravel().astype(np.float64),1); out.append((a,b))
    return out
def apply_affine(L,ab):
    o=np.stack([ab[c][0]*L[...,c].astype(np.float64)+ab[c][1] for c in range(3)],-1)
    return np.clip(o,0,255)
def fit_lut(L,G):
    """逐通道 E[GT|LQ=v] 的单调化 LUT（256 项）。"""
    luts=[]
    for c in range(3):
        l=L[...,c].ravel(); g=G[...,c].ravel().astype(np.float64)
        cnt=np.bincount(l,minlength=256).astype(float); s=np.bincount(l,weights=g,minlength=256)
        m=np.where(cnt>=50, s/np.maximum(cnt,1), np.nan)
        idx=np.arange(256); ok=~np.isnan(m)
        m=np.interp(idx, idx[ok], m[ok])
        m=np.maximum.accumulate(m)                      # 强制单调
        luts.append(np.clip(m,0,255))
    return luts
def apply_lut(L,luts): return np.stack([luts[c][L[...,c]] for c in range(3)],-1)

def crops(shape, n=8, S=512, skip_top=0):
    H,W=shape[:2]; rng=np.random.default_rng(1234)
    ys=np.linspace(skip_top, H-S-skip_top, 4).astype(int); xs=np.linspace(0, W-S, 3).astype(int)
    return [(y,x) for y in ys for x in xs][:n]

def to_t(a):
    return (torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))[None]).float().cuda()/127.5-1.0)

def eval_variant(name, imgs_lq, imgs_gt):
    fr=[];nr=[];pp=[]
    for l,g in zip(imgs_lq,imgs_gt):
        f,n,p=score(to_t(l), to_t(g))
        fr.append(f);nr.append(n);pp.append(p)
        torch.cuda.empty_cache()
    return name, float(np.mean(fr)), float(np.mean(nr)), float(np.mean(pp))

RES={}
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    skip = 100 if k=="case3" else 0
    cs=crops(L.shape, skip_top=skip)
    ab=fit_affine(L,G); luts=fit_lut(L,G)
    Aff=apply_affine(L,ab).astype(np.uint8)
    Lut=apply_lut(L,luts).astype(np.uint8)
    # oracle 色度替换
    Ly,Gy=rgb2ycc(L),rgb2ycc(G); t=Ly.copy(); t[...,1:]=Gy[...,1:]
    ChromaGT=np.clip(ycc2rgb(t),0,255).astype(np.uint8)
    # 固定校正（不看 GT）：三图平均的仿射
    FIX=np.clip(0.965*L.astype(np.float64)+3.0,0,255).astype(np.uint8)
    FIX2=np.clip(0.98*L.astype(np.float64)+1.5,0,255).astype(np.uint8)
    variants={"LQ(baseline)":L,"oracle affine":Aff,"oracle per-ch LUT":Lut,
              "oracle chroma=GT":ChromaGT,"fixed a=.965 b=+3":FIX,"fixed a=.98 b=+1.5":FIX2}
    print(f"\n===== {k}  (12 crops 512x512, skip_top={skip}) =====")
    print(f"  拟合的仿射: "+" ".join(f"{'RGB'[c]}:a={ab[c][0]:.4f},b={ab[c][1]:+.3f}" for c in range(3)))
    print(f"  {'variant':22s} {'TOPIQ-FR':>9s} {'MANIQA':>8s} {'proxy p':>8s} {'Δp vs LQ':>10s}")
    base=None
    for nm,img in variants.items():
        gc=[G[y:y+512,x:x+512] for y,x in cs]
        lc=[img[y:y+512,x:x+512] for y,x in cs]
        _,f,n,p=eval_variant(nm,lc,gc)
        if base is None: base=p
        print(f"  {nm:22s} {f:9.5f} {n:8.5f} {p:8.4f} {100*(p-base)/base:+9.3f}%")
        RES[(k,nm)]=(f,n,p)
json.dump({f"{k}|{n}":v for (k,n),v in RES.items()}, open("/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone/iqa_res.json","w"), indent=1)
