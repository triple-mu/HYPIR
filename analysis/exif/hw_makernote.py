import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff, parse_ifd

T = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"

def walk(mt, endian, off, indent, seen, label="IFD"):
    if off in seen or off >= len(mt): return
    seen.add(off)
    ents, nxt = parse_ifd(mt, endian, off, {})
    print("%s%s@%d  (%d entries)" % (indent, label, off, len(ents)))
    subs = []
    for e in ents:
        raw = bytes(e["raw"])
        v = e["value"]
        extra = ""
        if e["type_id"] == 11 and e["count"] == 1:
            extra = "  float=%.4f" % v[0]
        if e["type_id"] == 5 and e["count"] == 1 and v[0][1]:
            extra = "  =%.6g" % (v[0][0] / v[0][1])
        if e["type_id"] == 7 and e["count"] > 1:
            extra = "  raw=%r" % raw[:24]
        if isinstance(v, list) and len(v) > 8: v = str(v[:8]) + "..."
        print("%s  tag=0x%04x %-9s cnt=%-4d val=%-26s%s" % (indent, e["tag"], e["type"], e["count"], str(v)[:26], extra))
        # 可能的子 IFD 指针：LONG 且值落在 blob 内且指向合理的 entry-count
        if e["type_id"] == 4 and e["count"] == 1:
            o = e["value"][0]
            if 8 < o < len(mt) - 6:
                c = struct.unpack(endian + "H", mt[o:o+2])[0]
                if 0 < c < 40 and o + 2 + 12*c + 4 <= len(mt):
                    subs.append((e["tag"], o))
    for tag, o in subs:
        walk(mt, endian, o, indent + "    ", seen, "SubIFD(from 0x%04x)" % tag)
    if nxt:
        walk(mt, endian, nxt, indent, seen, "NextIFD")

def do(nm):
    d = open(T + "/" + nm + ".jpg", "rb").read()
    segs, _ = scan_segments(d)
    print("#" * 25, nm)
    for m, name, off, pl in segs:
        if name == "APP1" and pl.startswith(b"Exif\x00\x00"):
            r = parse_tiff(pl[6:])
            for e in r["ifds"]["ExifIFD"]:
                if e["name"] == "MakerNote" and e["size"] > 100:
                    raw = bytes(e["raw"]); mt = raw[8:]
                    en = "<" if mt[:2] == b"II" else ">"
                    io = struct.unpack(en + "I", mt[4:8])[0]
                    print(" MakerNote %dB endian=%s" % (len(raw), en))
                    walk(mt, en, io, "  ", set())
            break

for nm in sys.argv[1:]:
    do(nm)
    print()
