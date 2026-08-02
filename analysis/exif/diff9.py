import sys, os, struct, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
names = ["case1","case2","case3","case4","case5","case38","case64","case65","case66"]
alld = {}
order = []
for nm in names:
    fp = TEST + "/" + nm + ".jpg"
    data = open(fp, "rb").read()
    segs, _ = scan_segments(data)
    sof = None
    tiff = None
    for m, name, off, pl in segs:
        if name.startswith("SOF") and sof is None and len(pl) >= 6:
            h, w = struct.unpack(">HH", pl[1:5]); sof = (w, h)
        if name == "APP1" and pl.startswith(b"Exif\x00\x00") and tiff is None:
            tiff = pl[6:]
    r = parse_tiff(tiff)
    d = {"_SOF": "%dx%d" % sof, "_filesize": len(data), "_app1len": len(tiff)+6}
    for ifdname, ents in r["ifds"].items():
        seen = {}
        for e in ents:
            k = ifdname + "." + e["name"]
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > 1: k += "#%d" % seen[k]
            if e["name"] == "MakerNote" and e["size"] > 32:
                v = "<%dB %r>" % (e["size"], bytes(e["raw"][:16]))
            else:
                v = e["value"]
                if isinstance(v, list) and len(v) > 8: v = str(v[:8])+"..."
            d[k] = "%s|%s" % (e["type"], v)
            if k not in order: order.append(k)
    for k in ("_SOF","_filesize","_app1len"):
        if k not in order: order.insert(0, k)
    alld[nm] = d

# 打印：只显示有差异的字段 + 全部字段
same, diff = [], []
for k in order:
    vals = [alld[n].get(k, "<MISSING>") for n in names]
    (same if len(set(vals)) == 1 else diff).append(k)

print("======== 9 张图完全一致的字段 (%d) ========" % len(same))
for k in same:
    print("  %-40s %s" % (k, alld[names[0]][k]))
print()
print("======== 有差异的字段 (%d) ========" % len(diff))
w = max(len(n) for n in names)
print("%-38s %s" % ("FIELD", "  ".join(n.ljust(24) for n in names)))
for k in diff:
    print("%-38s %s" % (k, "  ".join(str(alld[n].get(k,'<MISS>'))[:24].ljust(24) for n in names)))
