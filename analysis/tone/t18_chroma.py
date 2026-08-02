"""任务4/5：Y/Cb/Cr 分解，色度损失占比；白平衡偏移。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=220)
M=np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
def rgb2ycc(a):
    a=a.astype(np.float64); o=a@M.T; o[...,1:]+=128; return o
def ycc2rgb(a):
    a=a.copy().astype(np.float64); a[...,1:]-=128; return a@np.linalg.inv(M).T
def rms(e): return float(np.sqrt(np.mean(np.asarray(e,np.float64)**2)))
def psnr(e): return 10*np.log10(255**2/max(np.mean(np.asarray(e,np.float64)**2),1e-12))

print("=== 任务4：色度损失占比 ===")
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    Ly,Gy=rgb2ycc(L),rgb2ycc(G)
    d=Gy-Ly
    print(f"\n{k}: RGB 总 RMSE={rms(G.astype(float)-L.astype(float)):.4f}  PSNR={psnr(G.astype(float)-L.astype(float)):.3f} dB")
    print(f"   Y  RMSE={rms(d[...,0]):7.4f}   Cb RMSE={rms(d[...,1]):7.4f}   Cr RMSE={rms(d[...,2]):7.4f}")
    # RGB MSE 可由 YCbCr MSE 精确分解（M 非正交，用实际矩阵）
    Minv=np.linalg.inv(M)
    # 单独把 Y / Cb / Cr 之一替换成 GT 的，看 RGB MSE 降多少
    base=np.mean((G.astype(np.float64)-L.astype(np.float64))**2)
    for i,nm in enumerate(["Y","Cb","Cr"]):
        t=Ly.copy(); t[...,i]=Gy[...,i]
        r=ycc2rgb(t)
        mse=np.mean((G.astype(np.float64)-r)**2)
        print(f"   只把 {nm:2s} 换成 GT 的 -> RGB MSE {base:9.3f} -> {mse:9.3f}  (消除 {100*(1-mse/base):6.2f}%)  PSNR {psnr(G.astype(np.float64)-r):.3f} dB")
    t=Ly.copy(); t[...,1:]=Gy[...,1:]; r=ycc2rgb(t)
    print(f"   Cb+Cr 都换成 GT -> RGB MSE {np.mean((G.astype(np.float64)-r)**2):9.3f} (消除 {100*(1-np.mean((G.astype(np.float64)-r)**2)/base):6.2f}%) PSNR {psnr(G.astype(np.float64)-r):.3f} dB")
    t=Ly.copy(); t[...,0]=Gy[...,0]; r=ycc2rgb(t)

print("\n\n=== 任务5：白平衡/色偏 ===")
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    Ly,Gy=rgb2ycc(L),rgb2ycc(G)
    print(f"\n{k}:")
    print(f"  全图均值  LQ Y/Cb/Cr = {Ly.reshape(-1,3).mean(0).round(3)}   GT = {Gy.reshape(-1,3).mean(0).round(3)}   Δ = {(Gy-Ly).reshape(-1,3).mean(0).round(3)}")
    print(f"  RGB 均值  LQ = {L.reshape(-1,3).mean(0).round(3)}  GT = {G.reshape(-1,3).mean(0).round(3)}  Δ = {(G.astype(float)-L.astype(float)).reshape(-1,3).mean(0).round(3)}")
    # 按亮度分层的 Cb/Cr 偏移
    y=Ly[...,0]
    print("   按 LQ 亮度分层的 ΔCb / ΔCr / ΔY :")
    bins=[(0,8),(8,16),(16,32),(32,64),(64,96),(96,128),(128,160),(160,192),(192,224),(224,256)]
    for lo,hi in bins:
        m=(y>=lo)&(y<hi)
        if m.sum()<1000: continue
        dY=(Gy[...,0]-Ly[...,0])[m].mean(); dCb=(Gy[...,1]-Ly[...,1])[m].mean(); dCr=(Gy[...,2]-Ly[...,2])[m].mean()
        print(f"     Y∈[{lo:3d},{hi:3d}) n={100*m.mean():5.2f}%  ΔY={dY:+7.3f}  ΔCb={dCb:+7.3f}  ΔCr={dCr:+7.3f}")
    # 灰世界色温
    for nm,A in (("LQ",L),("GT",G)):
        mu=A.reshape(-1,3).mean(0)
        print(f"   {nm} 灰世界增益 (R/G, B/G) = ({mu[0]/mu[1]:.5f}, {mu[2]/mu[1]:.5f})")
