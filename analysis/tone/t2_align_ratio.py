import numpy as np, cv2
from common import *

for k,(lp,gp) in PAIRS.items():
    L, G = imread(lp).astype(np.float32), imread(gp).astype(np.float32)
    ly = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY); gy = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    # subpixel shift via phase correlation on a central crop
    h,w = ly.shape; cy,cx = h//2, w//2; s=1024
    a = ly[cy-s//2:cy+s//2, cx-s//2:cx+s//2].copy()
    b = gy[cy-s//2:cy+s//2, cx-s//2:cx+s//2].copy()
    win = cv2.createHanningWindow((s,s), cv2.CV_32F)
    (dx,dy), resp = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64), win.astype(np.float64))
    print(f"{k}: phaseCorr shift (dx,dy)=({dx:+.4f},{dy:+.4f}) resp={resp:.4f}")
