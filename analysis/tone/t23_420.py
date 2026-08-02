"""4:2:0 单独贡献了多少损失？以及各 GT 的真实色度采样因子。"""
import numpy as np, cv2, struct
from common import *
def sof(path):
    d=open(path,'rb').read()
    if d[:2]!=b'\xff\xd8': return "PNG(全色度 4:4:4)"
    i=2
    while i<len(d)-1:
        if d[i]!=0xFF: i+=1; continue
        m=d[i+1]
        if m in (0xD8,0xD9) or 0xD0<=m<=0xD7: i+=2; continue
        if m==0xDA: break
        L=struct.unpack('>H',d[i+2:i+4])[0]; pl=d[i+4:i+2+L]
        if m in (0xC0,0xC1,0xC2):
            nc=pl[5]; cs=[(pl[6+c*3],pl[7+c*3]>>4,pl[7+c*3]&15) for c in range(nc)]
            samp="".join(f"id{a}:{h}x{v} " for a,h,v in cs)
            ratio="4:2:0" if cs[0][1]==2 and cs[0][2]==2 else ("4:4:4" if cs[0][1]==1 else "其他")
            return f"JPEG {samp}-> {ratio}"
        i+=2+L
    return "?"
print("=== 各文件真实容器与色度采样 ===")
for k,(lp,gp) in PAIRS.items():
    print(f"  {k}_lq: {sof(lp)}")
    print(f"  {k}_gt: {sof(gp)}")

M=np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
def rgb2ycc(a): o=a.astype(np.float64)@M.T; o[...,1:]+=128; return o
def ycc2rgb(a): a=a.astype(np.float64).copy(); a[...,1:]-=128; return a@np.linalg.inv(M).T
def sub420(y):
    h,w=y.shape
    d=cv2.resize(y,(w//2,h//2),interpolation=cv2.INTER_AREA)
    return cv2.resize(d,(w,h),interpolation=cv2.INTER_LINEAR)
print("\n=== 只对 GT 做 4:2:0 色度下采样+上采样，单独造成多少损失？ ===")
print(f"{'img':8s} {'LQ-GT 总MSE':>12s} {'仅420 的MSE':>12s} {'占比':>8s} {'仅420 PSNR':>11s}")
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    Gy=rgb2ycc(G); t=Gy.copy()
    t[...,1]=sub420(Gy[...,1]); t[...,2]=sub420(Gy[...,2])
    G420=np.clip(ycc2rgb(t),0,255)
    tot=np.mean((G.astype(np.float64)-L.astype(np.float64))**2)
    only=np.mean((G.astype(np.float64)-G420)**2)
    print(f"{k:8s} {tot:12.3f} {only:12.3f} {100*only/tot:7.2f}% {10*np.log10(255**2/only):10.2f} dB")
print("\n=== 反过来：LQ 的色度分辨率真的丢了吗？把 LQ 色度自己 420 往返，PSNR 越高说明本来就已带限 ===")
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    for nm,A in (("LQ",L),("GT",G)):
        Ay=rgb2ycc(A); t=Ay.copy(); t[...,1]=sub420(Ay[...,1]); t[...,2]=sub420(Ay[...,2])
        r=np.clip(ycc2rgb(t),0,255)
        # 只看色度平面自身的 PSNR
        pcb=10*np.log10(255**2/np.mean((Ay[...,1]-t[...,1])**2))
        pcr=10*np.log10(255**2/np.mean((Ay[...,2]-t[...,2])**2))
        print(f"  {k} {nm}: 色度自往返 Cb={pcb:.2f} dB  Cr={pcr:.2f} dB")
