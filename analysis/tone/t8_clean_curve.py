"""在 8x 降采样域（LQ/GT 内容几乎相同、带限差异消失）求干净的传递曲线。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=250)

def down(a,f): return cv2.resize(a,(a.shape[1]//f,a.shape[0]//f),interpolation=cv2.INTER_AREA)

CH="RGB"
CURVES={}
for k,(lp,gp) in PAIRS.items():
    L,G = imread(lp), imread(gp)
    Ld,Gd = down(L,8).astype(np.float64), down(G,8).astype(np.float64)
    print(f"\n########## {k}  (8x down: {Ld.shape}) ##########")
    for c in range(3):
        x,y = Ld[...,c].ravel(), Gd[...,c].ravel()
        # 分箱条件均值（宽度 1 的箱，x 是浮点故用 round）
        xi=np.clip(np.round(x),0,255).astype(int)
        cnt=np.bincount(xi,minlength=256).astype(float)
        s  =np.bincount(xi,weights=y,minlength=256)
        m  =np.where(cnt>0,s/np.maximum(cnt,1),np.nan)
        CURVES[(k,c)]=(m,cnt)
        # 拟合 gamma:  y/255 = (x/255)^g
        ok=(cnt>30)&(np.arange(256)>2)&(np.arange(256)<253)
        v=np.arange(256)[ok]
        g_fit=np.polyfit(np.log(v/255.0), np.log(np.clip(m[ok],1e-3,None)/255.0),1)[0]
        a,b=np.polyfit(x,y,1)
        # 残差：1D 曲线解释了多少
        pred_lin = a*x+b
        pred_cur = np.interp(x, np.arange(256), np.nan_to_num(m, nan=np.arange(256).astype(float)))
        def rms(e): return float(np.sqrt((e**2).mean()))
        print(f" {CH[c]}: identity RMSE={rms(y-x):6.3f} | affine(a={a:.4f},b={b:+.3f}) RMSE={rms(y-pred_lin):6.3f} "
              f"| 1D-curve RMSE={rms(y-pred_cur):6.3f} | gamma_fit={g_fit:.4f}")

print("\n\n=== 干净传递曲线 f(v)=E[GT|LQ=v]（8x 降采样域），显示 f(v)-v ===")
vs=[0,2,4,8,12,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,250,255]
print("v        "+"".join(f"{v:>6d}" for v in vs))
for k in PAIRS:
    for c in range(3):
        m,cnt=CURVES[(k,c)]
        print(f"{k} {CH[c]}  "+"".join((f"{m[v]-v:>6.2f}" if cnt[v]>20 else "     ·") for v in vs))
print("\n=== 同上，显示 f(v) 绝对值 ===")
print("v        "+"".join(f"{v:>6d}" for v in vs))
for k in PAIRS:
    for c in range(3):
        m,cnt=CURVES[(k,c)]
        print(f"{k} {CH[c]}  "+"".join((f"{m[v]:>6.1f}" if cnt[v]>20 else "     ·") for v in vs))
np.save("clean_curves.npy", {f"{k}_{c}":CURVES[(k,c)] for k in PAIRS for c in range(3)}, allow_pickle=True)
