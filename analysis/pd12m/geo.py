"""EXIF GPS -> 国家/地区。Natural Earth 50m 多边形 + matplotlib.path 点在多边形内判定。"""
import os
import json, numpy as np
from matplotlib.path import Path
_P = None
# Natural Earth 1:50m 国界（公有领域），随脚本一起放在同目录
def _load(fp=os.path.join(os.path.dirname(__file__), "ne_50m_countries.geojson")):
    global _P
    if _P is not None: return _P
    g = json.load(open(fp)); out = []
    for f in g["features"]:
        nm = f["properties"]["NAME"]
        gm = f["geometry"]
        polys = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
        for p in polys:
            r = np.asarray(p[0], dtype=float)
            out.append((nm, r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(), Path(r)))
    _P = out; return out

def country(lat, lon):
    for nm, x0, x1, y0, y1, pa in _load():
        if x0 <= lon <= x1 and y0 <= lat <= y1 and pa.contains_point((lon, lat)):
            return nm
    return None
