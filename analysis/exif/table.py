import sys, os, struct, glob, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
files = sorted(glob.glob(TEST + "/case*.jpg"), key=lambda p: int(re.search(r"case(\d+)", p).group(1)))

def rat(v):
    if not v: return ""
    a, b = v[0]
    return "%g" % (a / b) if b else "inf"

rows = []
for fp in files:
    data = open(fp, "rb").read()
    segs, _ = scan_segments(data)
    sof = None
    app1exif = None
    for m, name, off, pl in segs:
        if name.startswith("SOF") and sof is None and len(pl) >= 6:
            h, w = struct.unpack(">HH", pl[1:5]); sof = (w, h)
        if name == "APP1" and pl.startswith(b"Exif\x00\x00") and app1exif is None:
            app1exif = pl[6:]
    if app1exif is None:
        continue
    r = parse_tiff(app1exif)
    d = {}
    for ifdname, ents in r["ifds"].items():
        for e in ents:
            k = e["name"]
            if k in d:
                k = k + "#%d" % sum(1 for x in d if x.startswith(e["name"]))
            d[k] = e
    def g(n, default=""):
        e = d.get(n)
        if e is None: return default
        return e["value"]
    base = os.path.basename(fp)
    rows.append(dict(
        file=base, filesize=len(data), sof="%dx%d" % sof,
        exif_wh="%sx%s" % (g("ExifImageWidth", [""])[0], g("ExifImageHeight", [""])[0]),
        ImageDescription=g("ImageDescription"), Make=g("Make"), Model=g("Model"),
        Software=g("Software"), DateTime=g("DateTime"), DTO=g("DateTimeOriginal"),
        DTD=g("DateTimeDigitized"), SubSec=g("SubSecTimeOriginal"),
        Orientation=g("Orientation", [""])[0],
        Exp=rat(g("ExposureTime")), F=rat(g("FNumber")), ISO=g("ISOSpeedRatings", [""])[0],
        FL=rat(g("FocalLength")), FL35=g("FocalLengthIn35mmFilm", [""])[0],
        Zoom=rat(g("DigitalZoomRatio")),
        ColorSpace=g("ColorSpace", [""])[0], Scene=g("SceneCaptureType", [""])[0],
        MeteringMode=g("MeteringMode", [""])[0], LightSource=g("LightSource", [""])[0],
        Bright=rat(g("BrightnessValue")), MaxAp=rat(g("MaxApertureValue")),
        ExifVer="".join(chr(c) for c in g("ExifVersion", [])),
        mn_tags=[k for k in d if k.startswith("MakerNote")],
        mn_marks=[bytes(d[k]["raw"]).split(b"\x00")[0].decode("latin1") if d[k]["type_id"]==7 else "" for k in d if k.startswith("MakerNote")],
        n_ifd0=len(r["ifds"].get("IFD0", [])), n_exif=len(r["ifds"].get("ExifIFD", [])),
        has_gps="GPSIFD" in r["ifds"], ifd1=r.get("ifd1_offset", 0),
        app1_len=len(app1exif) + 6,
    ))

hdr = ["file","filesize","sof","exif_wh","Make","Model","Software","DateTime","SubSec","Orientation",
       "Exp","F","ISO","FL","FL35","Zoom","ColorSpace","Scene","ImageDescription","ExifVer","n_ifd0","n_exif","has_gps","ifd1","app1_len"]
print("\t".join(hdr))
for r in rows:
    print("\t".join(str(r.get(h, "")) for h in hdr))
print()
for r in rows:
    print(r["file"], r["mn_tags"], r["mn_marks"])
json.dump(rows, open("/tmp/exif_rows.json","w"), default=str, ensure_ascii=False)
