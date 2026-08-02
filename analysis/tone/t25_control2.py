"""对照2：带限 + JPEG重编码（用验证集 LQ 同款量化表, 4:2:0），色调仍然零改动。"""
import numpy as np, cv2, io, struct
from PIL import Image
from common import *
def bandlimit(a):
    h,w=a.shape[:2]
    d=cv2.resize(a,(w//2,h//2),interpolation=cv2.INTER_AREA)
    return np.clip(cv2.resize(d,(w,h),interpolation=cv2.INTER_CUBIC),0,255).astype(np.uint8)
def qtabs(path):
    d=open(path,'rb').read(); i=2; out={}
    while i<len(d)-1:
        if d[i]!=0xFF: i+=1; continue
        m=d[i+1]
        if m in (0xD8,0xD9) or 0xD0<=m<=0xD7: i+=2; continue
        if m==0xDA: break
        L=struct.unpack('>H',d[i+2:i+4])[0]; pl=d[i+4:i+2+L]
        if m==0xDB:
            o=0
            while o<len(pl):
                tq=pl[o]&15; o+=1; out[tq]=list(np.frombuffer(pl[o:o+64],'u1').astype(int)); o+=64
        i+=2+L
    return out
QT=qtabs(PAIRS["case1"][0])
def jenc(a):
    b=io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", qtables=[QT[0],QT[1]], subsampling=2, optimize=False)
    b.seek(0); return np.asarray(Image.open(b).convert("RGB"))
def gdown(a,s):
    b=cv2.GaussianBlur(np.asarray(a,np.float32),(0,0),s,borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return b[s//2::s,s//2::s]
def aff(L,G): return [np.polyfit(L[...,c].ravel().astype(np.float64),G[...,c].ravel().astype(np.float64),1) for c in range(3)]
print("LQ_syn = JPEG(带限(GT))，量化表与验证集 LQ 完全相同，4:2:0，色调零改动")
print(f"{'img':7s} {'尺度':>8s} {'a_R':>8s} {'a_G':>8s} {'a_B':>8s} {'b_R':>8s} {'b_G':>8s} {'b_B':>8s}")
for k,(lp,gp) in PAIRS.items():
    G=imread(gp); Ls=jenc(bandlimit(G))
    for s in (1,8,32):
        A,B=(Ls,G) if s==1 else (gdown(Ls,s),gdown(G,s))
        ab=aff(A,B)
        print(f"{k:7s} {('原分辨率' if s==1 else f's={s}'):>8s} "+"".join(f"{ab[c][0]:8.4f}" for c in range(3))+"".join(f"{ab[c][1]:+8.3f}" for c in range(3)))
    print(f"        真实 LQ 原分辨率: "+"".join(f"{aff(imread(lp),G)[c][0]:8.4f}" for c in range(3)))
