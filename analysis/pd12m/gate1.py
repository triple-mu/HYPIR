import pandas as pd, numpy as np, sys, os
sys.path.insert(0,"/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
from csig_data import hf_max
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
Image.MAX_IMAGE_PIXELS=None
d = pd.read_csv("fullset.csv")
def f(k):
    p=f"full/{k:04d}.jpg"
    try:
        im=Image.open(p); g=np.asarray(im.convert("L"))
        return hf_max(g), os.path.getsize(p)*8/(im.width*im.height)
    except Exception as e: return (np.nan, np.nan)
with ProcessPoolExecutor(20) as ex:
    r=list(ex.map(f, range(len(d))))
d["hfmax"]=[x[0] for x in r]; d["bpp_full"]=[x[1] for x in r]
d.to_csv("fullset_gate.csv", index=False)
print(d.groupby("grp").agg(n=("hfmax","size"), hf_med=("hfmax","median"),
    pass1=("hfmax", lambda s:(s>=0.005).mean()), bpp_med=("bpp_full","median")).to_string())
