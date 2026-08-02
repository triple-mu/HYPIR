import numpy as np, cv2
from common import *
def down(a,f): return cv2.resize(a,(a.shape[1]//f,a.shape[0]//f),interpolation=cv2.INTER_AREA)
for k,(lp,gp) in PAIRS.items():
    L,G=imread(lp),imread(gp)
    f=8
    Ld,Gd=down(L,f),down(G,f)
    d = Gd.astype(np.float32)-Ld.astype(np.float32)
    dy = d.mean(2)
    vis = np.clip(dy*6+128,0,255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    pan = np.hstack([cv2.cvtColor(Ld,cv2.COLOR_RGB2BGR), cv2.cvtColor(Gd,cv2.COLOR_RGB2BGR), vis])
    cv2.imwrite(f"vis_{k}.png", pan)
    print(k, "wrote vis_%s.png (LQ | GT | (GT-LQ)*6+128 jet)"%k, "d stats mean=%.2f std=%.2f"%(dy.mean(), dy.std()))
