"""对任意本地图片目录跑三闸门 + EXIF，输出 CSV 行。"""
import os, sys, io, json
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from csig_data import hf_max, ijg_quality

def probe(path):
    r = {"path": path, "size": os.path.getsize(path)}
    try:
        im = Image.open(path)
    except Exception as e:
        r["err"] = str(e); return r
    r["w"], r["h"] = im.width, im.height
    r["long"] = max(im.width, im.height)
    r["bpp"] = round(r["size"] * 8 / (im.width * im.height), 3)
    q = getattr(im, "quantization", None)
    r["ijg"] = ijg_quality(q[0]) if q else None
    ex = {}
    try:
        e = im.getexif()
        ex["Make"] = e.get(271); ex["Model"] = e.get(272); ex["Software"] = e.get(305)
        ifd = e.get_ifd(0x8769)
        ex["LensModel"] = ifd.get(0xA434)
        ex["ExifW"] = ifd.get(0xA002); ex["ExifH"] = ifd.get(0xA003)
    except Exception:
        pass
    r["exif"] = {k: v for k, v in ex.items() if v}
    try:
        g = np.asarray(im.convert("L"))
        r["hf_max"] = round(hf_max(g), 5)
    except Exception as e:
        r["err"] = str(e); return r
    r["pass"] = bool(r["long"] >= 3000 and r["hf_max"] >= 0.005 and r["bpp"] >= 2.0
                     and (r["ijg"] is None or r["ijg"] >= 93))
    return r

if __name__ == "__main__":
    d = sys.argv[1]
    files = sorted(os.listdir(d)) if os.path.isdir(d) else [d]
    for f in files:
        p = os.path.join(d, f) if os.path.isdir(d) else d
        if not os.path.isfile(p): continue
        print(json.dumps(probe(p), ensure_ascii=False, default=str))
