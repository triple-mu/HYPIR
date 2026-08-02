"""任务3 决定性检验：把 gain map 施加到 SDR 基图上，看它造成的色调变化方向，
   与验证集观察到的 LQ->GT 方向对比。"""
import numpy as np, cv2, io, struct
from PIL import Image
TEST="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
def load_pair(n):
    d=open(f"{TEST}/case{n}.jpg",'rb').read()
    j=d.find(b'MPF\x00'); tiff=d[j+4:]; base=j+4
    bo='<' if tiff[:2]==b'II' else '>'
    ifd0=struct.unpack(bo+'I',tiff[4:8])[0]; nent=struct.unpack(bo+'H',tiff[ifd0:ifd0+2])[0]
    ent={}
    for q in range(nent):
        e=tiff[ifd0+2+q*12:ifd0+14+q*12]; tag,typ,cnt=struct.unpack(bo+'HHI',e[:8]); ent[tag]=(typ,cnt,e[8:12])
    off=struct.unpack(bo+'I',ent[0xB002][2])[0]
    a0,s0,d0,_,_=struct.unpack(bo+'IIIHH', tiff[off:off+16])
    a1,s1,d1,_,_=struct.unpack(bo+'IIIHH', tiff[off+16:off+32])
    sdr=np.asarray(Image.open(io.BytesIO(d[:s0])).convert("RGB"))
    gm =np.asarray(Image.open(io.BytesIO(d[base+d1:base+d1+s1])).convert("L"))
    return sdr, gm

GMAX=2.169925   # log2(4.5), 全部 9 张一致
def apply_gain(sdr, gm, w):
    """w=1 表示全量施加 (HDR headroom 2.148 st)；再用简单 Reinhard 压回 SDR。"""
    g = cv2.resize(gm, (sdr.shape[1], sdr.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    gain = 2.0**(g*GMAX*w)                                  # gamma=1, min=0
    lin = np.where(sdr/255.0<=0.04045,(sdr/255.0)/12.92,((sdr/255.0+0.055)/1.055)**2.4).astype(np.float32)
    hdr = lin*gain[...,None]
    return hdr
def enc(x):
    x=np.clip(x,0,1); return np.where(x<=0.0031308,x*12.92,1.055*x**(1/2.4)-0.055)*255.0
def reinhard(hdr, Lw):
    """把峰值 Lw 的 HDR 用 extended Reinhard 压回 [0,1]。"""
    return hdr*(1+hdr/(Lw**2))/(1+hdr)

for n in (1,38,66):
    sdr,gm = load_pair(n)
    print(f"\n===== case{n} =====")
    print(f"  SDR mean/ch={sdr.reshape(-1,3).mean(0).round(2)} std={sdr.reshape(-1,3).std(0).round(2)}")
    g=cv2.resize(gm,(sdr.shape[1],sdr.shape[0]),interpolation=cv2.INTER_LINEAR)
    y=cv2.cvtColor(sdr,cv2.COLOR_RGB2GRAY)
    print(f"  gain map: frac(gain=1)={100*(g==0).mean():.2f}%  与亮度相关 corr(gainmap, Y)={np.corrcoef(g[::8,::8].ravel(),y[::8,::8].ravel())[0,1]:+.4f}")
    for lo,hi in [(0,32),(32,64),(64,128),(128,192),(192,240),(240,256)]:
        m=(y>=lo)&(y<hi)
        if m.sum()<10000: continue
        print(f"    SDR Y∈[{lo:3d},{hi:3d}) n={100*m.mean():5.2f}%  平均线性增益={2.0**(g[m].astype(np.float32)/255*GMAX).mean():.4f}x")
    hdr=apply_gain(sdr,gm,1.0)
    Lw=2.0**2.148445
    out=np.clip(enc(reinhard(hdr,Lw)),0,255)
    d=out-sdr.astype(np.float64)
    print(f"  全量施加 gain map + Reinhard 压回 SDR 后，相对原 SDR:")
    print(f"    Δmean/ch={d.reshape(-1,3).mean(0).round(3)}  RMSE={np.sqrt((d**2).mean()):.3f}")
    for lo,hi in [(0,32),(32,64),(64,128),(128,192),(192,256)]:
        m=(y>=lo)&(y<hi)
        if m.sum()<10000: continue
        print(f"    SDR Y∈[{lo:3d},{hi:3d})  Δ(输出-原SDR) 均值={d[m].mean():+7.3f}")
