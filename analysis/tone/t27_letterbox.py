import numpy as np, cv2, glob, os
TEST="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
NATIVE={1,2,3,4,5,38,64,65,66}
hits=[]
for p in sorted(glob.glob(TEST+"/*.jpg"), key=lambda x:int(''.join(filter(str.isdigit,os.path.basename(x))))):
    n=int(''.join(filter(str.isdigit,os.path.basename(p))))
    a=cv2.imread(p,cv2.IMREAD_COLOR)
    rm=a.reshape(a.shape[0],-1).mean(1); rs=a.reshape(a.shape[0],-1).std(1)
    t=0
    while t<len(rm) and rm[t]<14 and rs[t]<10: t+=1
    b=0
    while b<len(rm) and rm[-1-b]<14 and rs[-1-b]<10: b+=1
    if t>=8 or b>=8:
        hits.append((n, t, b, n in NATIVE, np.round(a[:max(t,1)].reshape(-1,3).mean(0)[::-1],2)))
print(f"检测到上下暗边(行均值<14 且行 std<10, 连续>=8 行)的图: {len(hits)}/100")
for n,t,b,nat,mu in hits:
    print(f"  case{n:<4d} top={t:<4d} bottom={b:<4d} {'原生' if nat else '重编码'}  边条 RGB 均值={mu}")
