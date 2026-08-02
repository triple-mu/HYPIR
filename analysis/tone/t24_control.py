"""对照实验：对 GT 只做「2x 下采样+上采样」这一个已知退化（色调完全不变），
   再拟合 gt ≈ a*lq+b。若 a 也 <1，则 FINDINGS#4 的 a<1 是回归稀释伪影而非真实色调差。"""
import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=200)
def bandlimit(a):
    h,w=a.shape[:2]
    d=cv2.resize(a,(w//2,h//2),interpolation=cv2.INTER_AREA)
    return cv2.resize(d,(w,h),interpolation=cv2.INTER_CUBIC)
def gdown(a,s):
    b=cv2.GaussianBlur(np.asarray(a,np.float32),(0,0),s,borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return b[s//2::s,s//2::s]
def aff(L,G):
    return [np.polyfit(L[...,c].ravel().astype(np.float64),G[...,c].ravel().astype(np.float64),1) for c in range(3)]
print("对照：LQ_syn = 带限(GT)，色调零改动。拟合 GT ≈ a*LQ_syn+b")
print(f"{'img':7s} {'尺度':>6s} {'a_R':>8s} {'a_G':>8s} {'a_B':>8s} {'b_R':>8s} {'b_G':>8s} {'b_B':>8s}")
for k,(lp,gp) in PAIRS.items():
    G=imread(gp); Ls=np.clip(bandlimit(G),0,255).astype(np.uint8)
    for s in (1,8,32):
        A,B=(Ls,G) if s==1 else (gdown(Ls,s),gdown(G,s))
        ab=aff(A,B)
        print(f"{k:7s} {('原分辨率' if s==1 else f's={s}'):>6s} "+"".join(f"{ab[c][0]:8.4f}" for c in range(3))+"".join(f"{ab[c][1]:+8.3f}" for c in range(3)))
print("\n对比：真实 LQ 的拟合值")
print(f"{'img':7s} {'尺度':>6s} {'a_R':>8s} {'a_G':>8s} {'a_B':>8s} {'b_R':>8s} {'b_G':>8s} {'b_B':>8s}")
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    for s in (1,8,32):
        A,B=(L,G) if s==1 else (gdown(L,s),gdown(G,s))
        ab=aff(A,B)
        print(f"{k:7s} {('原分辨率' if s==1 else f's={s}'):>6s} "+"".join(f"{ab[c][0]:8.4f}" for c in range(3))+"".join(f"{ab[c][1]:+8.3f}" for c in range(3)))
