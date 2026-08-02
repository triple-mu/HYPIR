"""检验：LQ 是否 = P3->sRGB(GT)？或 GT = P3->sRGB(LQ)？"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=200, precision=6)

def prim2mat(xy, wp):
    x=np.array([p[0] for p in xy]); y=np.array([p[1] for p in xy])
    X=x/y; Y=np.ones(3); Z=(1-x-y)/y
    M=np.vstack([X,Y,Z])
    W=np.array([wp[0]/wp[1], 1.0, (1-wp[0]-wp[1])/wp[1]])
    S=np.linalg.solve(M, W)
    return M*S
D65=(0.3127,0.3290)
M_sRGB = prim2mat([(0.64,0.33),(0.30,0.60),(0.15,0.06)], D65)
M_P3   = prim2mat([(0.680,0.320),(0.265,0.690),(0.150,0.060)], D65)
P3_to_sRGB = np.linalg.inv(M_sRGB) @ M_P3
sRGB_to_P3 = np.linalg.inv(P3_to_sRGB)
print("P3->sRGB 线性矩阵:\n", P3_to_sRGB)
print("sRGB->P3 线性矩阵:\n", sRGB_to_P3)

LUT_lin = np.where(np.arange(256)/255.0<=0.04045, (np.arange(256)/255.0)/12.92,
                   ((np.arange(256)/255.0+0.055)/1.055)**2.4)
def enc(x):
    x=np.clip(x,0,1)
    return np.where(x<=0.0031308, x*12.92, 1.055*x**(1/2.4)-0.055)
def convert(img8, M):
    lin = LUT_lin[img8]                      # sRGB EOTF 解码
    out = lin @ M.T
    clipped = ((out<0)|(out>1)).any(-1)
    return (enc(out)*255.0), clipped

def rms(a,b): return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))
def gdown(a,s):
    b=cv2.GaussianBlur(np.asarray(a,np.float32),(0,0),s,borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return b[s//2::s,s//2::s]

for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    gp3, clip1 = convert(G, P3_to_sRGB)      # 假设 GT 是 P3，转成 sRGB 数值
    lp3, clip2 = convert(L, sRGB_to_P3)      # 假设 LQ 是 sRGB，转回 P3 数值
    # 低通比较，剔除高频损失
    s=8
    Ls,Gs = gdown(L,s), gdown(G,s)
    GtoS  = gdown(gp3,s); LtoP = gdown(lp3,s)
    print(f"\n{k} (s={s} 低通域 RMSE):")
    print(f"   identity            LQ vs GT           = {rms(Ls,Gs):7.3f}")
    print(f"   H1: LQ vs P3->sRGB(GT)                 = {rms(Ls,GtoS):7.3f}   (GT 转换后越界像素 {100*clip1.mean():.2f}%)")
    print(f"   H2: GT vs sRGB->P3(LQ)                 = {rms(Gs,LtoP):7.3f}   (LQ 转换后越界像素 {100*clip2.mean():.2f}%)")
    for c,nm in enumerate("RGB"):
        print(f"     {nm}: id={rms(Ls[...,c],Gs[...,c]):6.3f}  H1={rms(Ls[...,c],GtoS[...,c]):6.3f}  H2={rms(Gs[...,c],LtoP[...,c]):6.3f}")
    print(f"   零值占比: LQ={[round(100*(L[...,c]==0).mean(),3) for c in range(3)]}  "
          f"P3->sRGB(GT)={[round(100*(gp3[...,c]<0.5).mean(),3) for c in range(3)]}  GT={[round(100*(G[...,c]==0).mean(),3) for c in range(3)]}")
