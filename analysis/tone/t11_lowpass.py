"""正确的低通：高斯预滤波(sigma=s)后每 s 点抽样，抗混叠。对比 INTER_AREA 的混叠伪影。"""
import numpy as np, cv2
from common import *
def gdown(a, s):
    b = cv2.GaussianBlur(a.astype(np.float32), (0,0), s, borderType=cv2.BORDER_REFLECT)
    return b[s//2::s, s//2::s]
def adown(a,f): return cv2.resize(a,(a.shape[1]//f,a.shape[0]//f),interpolation=cv2.INTER_AREA).astype(np.float32)

for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    print(f"\n{k}:")
    for s in (2,4,8,16,32):
        La,Ga = adown(L,s), adown(G,s)
        Lg,Gg = gdown(L,s), gdown(G,s)
        ra=float(np.sqrt(((Ga-La)**2).mean())); rg=float(np.sqrt(((Gg-Lg)**2).mean()))
        # 高斯低通域的仿射
        aa=[np.polyfit(Lg[...,c].ravel(),Gg[...,c].ravel(),1) for c in range(3)]
        print(f"  s={s:2d}: INTER_AREA RMSE={ra:6.3f} | Gaussian RMSE={rg:6.3f} | "
              f"gauss affine a=[{aa[0][0]:.4f},{aa[1][0]:.4f},{aa[2][0]:.4f}] b=[{aa[0][1]:+.2f},{aa[1][1]:+.2f},{aa[2][1]:+.2f}]")
