import numpy as np, cv2
from common import *
np.set_printoptions(suppress=True, linewidth=220)
L=imread(PAIRS["case3"][0]); G=imread(PAIRS["case3"][1])
print("case3 LQ 左上角 8x8 RGB:"); print(L[:8,:8].reshape(8,-1))
print("case3 GT 左上角 8x8 RGB:"); print(G[:8,:8].reshape(8,-1))
print("\nLQ 边条(0:70) 各通道直方图 top10:")
for c,nm in enumerate("RGB"):
    v=L[:70,:,c].ravel(); h=np.bincount(v,minlength=256)
    top=np.argsort(h)[::-1][:10]
    print(f"  {nm}: 值={top} 占比%={np.round(100*h[top]/v.size,2)}")
print("GT 边条(0:70) 各通道唯一值:")
for c,nm in enumerate("RGB"):
    print(f"  {nm}: {np.unique(G[:70,:,c])}")
# YCbCr (JPEG/BT.601 full range) —— LQ 是 JPEG 解码得到的，libjpeg 用的就是这个
def rgb2ycc(a):
    a=a.astype(np.float64)
    Y = 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    Cb= 128 -0.168736*a[...,0]-0.331264*a[...,1]+0.5*a[...,2]
    Cr= 128 +0.5*a[...,0]-0.418688*a[...,1]-0.081312*a[...,2]
    return np.stack([Y,Cb,Cr],-1)
lb=rgb2ycc(L[:70]); gb=rgb2ycc(G[:70])
print(f"\n边条 YCbCr:  LQ mean={lb.reshape(-1,3).mean(0).round(3)} std={lb.reshape(-1,3).std(0).round(3)}")
print(f"            GT mean={gb.reshape(-1,3).mean(0).round(3)} std={gb.reshape(-1,3).std(0).round(3)}")
# 直接解 JPEG 的 YCbCr 平面（不做上采样/色转），看 LQ 边条编码里到底存了什么
import subprocess
