import os, sys, struct, glob, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
files = sorted(glob.glob(TEST + "/case*.jpg"), key=lambda p: int(re.search(r"case(\d+)", p).group(1)))
print("n files", len(files))

summary = []
for fp in files:
    data = open(fp, "rb").read()
    segs, err = scan_segments(data)
    apps = []
    sof = None
    for m, name, off, pl in segs:
        if 0xE0 <= m <= 0xEF:
            # 取标识符
            ident = pl.split(b"\x00")[0][:20]
            apps.append((name, len(pl), ident.decode("latin1", "replace")))
        if name.startswith("SOF") and sof is None:
            if len(pl) >= 6:
                prec = pl[0]
                h, w = struct.unpack(">HH", pl[1:5])
                nc = pl[5]
                comps = []
                for c in range(nc):
                    cid, hv, tq = pl[6 + c * 3:9 + c * 3]
                    comps.append((cid, hv >> 4, hv & 15, tq))
                sof = dict(marker=name, prec=prec, w=w, h=h, comps=comps)
    has_exif = any(a[2].startswith("Exif") for a in apps)
    rec = dict(file=os.path.basename(fp), size=len(data), sof=sof, apps=apps, has_exif=has_exif)
    summary.append(rec)

nex = sum(1 for r in summary if r["has_exif"])
print("with EXIF:", nex)
print("files with EXIF:", [r["file"] for r in summary if r["has_exif"]])
print()
# APP 段签名统计
from collections import Counter
c = Counter()
for r in summary:
    sig = "|".join("%s:%s" % (a[0], a[2]) for a in r["apps"])
    c[sig] += 1
for k, v in c.most_common():
    print(v, "  ", k)
print()
for r in summary:
    if r["has_exif"]:
        print(r["file"], r["size"], r["sof"]["w"], "x", r["sof"]["h"], r["sof"]["marker"], r["sof"]["comps"])
        for a in r["apps"]:
            print("   ", a)
json.dump([{k: v for k, v in r.items()} for r in summary], open("/tmp/seg_summary.json", "w"), default=str)
