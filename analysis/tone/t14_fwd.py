"""正向曲线 E[LQ|GT=v]（GT 是自变量），在抗混叠低通域里做，看退化到底做了什么。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=250)
def gdown(a,s):
    b=cv2.GaussianBlur(a.astype(np.float32),(0,0),s,borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return b[s//2::s,s//2::s]
CH="RGB"
vs=[0,4,8,12,16,20,24,28,32,40,48,64,80,96,128,160,192,224,240,252]
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    Lg,Gg=gdown(L,8),gdown(G,8)
    print(f"\n### {k}  正向 E[LQ|GT=v] - v  (s=8 抗混叠低通域) ###")
    print("GT v   "+"".join(f"{v:>6d}" for v in vs))
    for c in range(3):
        x=np.clip(np.round(Gg[...,c].ravel()),0,255).astype(int); y=Lg[...,c].ravel()
        cnt=np.bincount(x,minlength=256).astype(float); s_=np.bincount(x,weights=y,minlength=256)
        m=np.where(cnt>0,s_/np.maximum(cnt,1),np.nan)
        print(f"  {CH[c]}    "+"".join((f"{m[v]-v:>6.2f}" if cnt[v]>20 else "     ·") for v in vs))
    print("GT像素占比%"+"".join(f"{100*np.bincount(np.clip(np.round(Gg[...,0].ravel()),0,255).astype(int),minlength=256)[v]/Gg[...,0].size:>6.2f}" for v in vs))

# ICC profile 解析
import struct
def icc_info(path):
    d=open(path,'rb').read(); i=d.find(b'ICC_PROFILE\x00')
    if i<0: return None
    L=struct.unpack('>H',d[i-2:i])[0]; p=d[i+14:i+14+L-16]
    n=struct.unpack('>I',p[128:132])[0]; out={}
    for j in range(n):
        sig,off,sz=struct.unpack('>4sII',p[132+j*12:144+j*12])
        out[sig.decode()]= p[off:off+sz]
    def xyz(t):
        if t is None: return None
        return tuple(struct.unpack('>i',t[8+4*q:12+4*q])[0]/65536 for q in range(3))
    def txt(t):
        if t is None: return None
        if t[:4]==b'desc':
            ln=struct.unpack('>I',t[8:12])[0]; return t[12:12+ln].rstrip(b'\x00').decode('latin1')
        if t[:4]==b'mluc':
            ln=struct.unpack('>I',t[20:24])[0]; o=struct.unpack('>I',t[24:28])[0]
            return t[o:o+ln].decode('utf-16-be').rstrip('\x00')
        return t[:40]
    r,g,b,w = xyz(out.get('rXYZ')),xyz(out.get('gXYZ')),xyz(out.get('bXYZ')),xyz(out.get('wtpt'))
    trc=out.get('rTRC')
    trcinfo=None
    if trc:
        if trc[:4]==b'curv':
            n2=struct.unpack('>I',trc[8:12])[0]
            trcinfo=f"curv n={n2}" + (f" gamma={struct.unpack('>H',trc[12:14])[0]/256:.4f}" if n2==1 else "")
        elif trc[:4]==b'para':
            ft=struct.unpack('>H',trc[8:10])[0]
            par=[struct.unpack('>i',trc[12+4*q:16+4*q])[0]/65536 for q in range((len(trc)-12)//4)]
            trcinfo=f"para type={ft} params={[round(x,5) for x in par]}"
    return dict(desc=txt(out.get('desc')), tags=list(out.keys()), rXYZ=r,gXYZ=g,bXYZ=b,wtpt=w,trc=trcinfo)
print("\n\n### 原生测试图的 ICC ###")
for n in (1,64):
    info=icc_info(f"/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集/case{n}.jpg")
    print(f"case{n}:", info['desc'], "| tags", info['tags'])
    print("   rXYZ",[round(x,5) for x in info['rXYZ']],"gXYZ",[round(x,5) for x in info['gXYZ']],"bXYZ",[round(x,5) for x in info['bXYZ']],"wtpt",[round(x,5) for x in info['wtpt']])
    print("   TRC:",info['trc'])
# sRGB / DisplayP3 参考 primaries (Bradford-adapted to D50, 即 ICC 存的形式)
print("\n参考(D50适配后): sRGB rXYZ=(0.43607,0.22249,0.01392) gXYZ=(0.38515,0.71687,0.09708) bXYZ=(0.14307,0.06061,0.71410)")
print("参考(D50适配后): DisplayP3 rXYZ=(0.51512,0.24120,-0.00105) gXYZ=(0.29198,0.69224,0.04189) bXYZ=(0.15710,0.06657,0.78407)")
