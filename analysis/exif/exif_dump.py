import sys, os, struct, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff

def fmt(e):
    v = e["value"]
    if isinstance(v, list) and len(v) > 12:
        v = str(v[:12]) + "...(%d)" % len(v)
    return v

def show(fp):
    data = open(fp, "rb").read()
    segs, _ = scan_segments(data)
    print("######## %s (%d B) ########" % (os.path.basename(fp), len(data)))
    idx = 0
    for m, name, off, pl in segs:
        if name == "APP1" and pl.startswith(b"Exif\x00\x00"):
            idx += 1
            tiff = pl[6:]
            r = parse_tiff(tiff)
            print("-- APP1 Exif #%d @%d payload=%dB  byte_order=%s magic=%s ifd0_off=%d ifd1_off=%d" % (
                idx, off, len(pl), r.get("byte_order"), r.get("magic"), r.get("ifd0_offset"), r.get("ifd1_offset", 0)))
            for ifdname, ents in r["ifds"].items():
                print("  [%s]  %d entries" % (ifdname, len(ents)))
                for e in ents:
                    extra = ""
                    if e["name"] == "MakerNote" or e["size"] > 64:
                        extra = " rawhead=%r" % (e["raw"][:48],)
                        v = "<%d bytes>" % e["size"]
                    else:
                        v = fmt(e)
                    print("    %-8s %-28s %-10s cnt=%-6d off=%-8s %s%s" % (
                        e["tag_hex"], e["name"], e["type"], e["count"],
                        e["value_offset"], v, extra))
        elif name == "APP1":
            print("-- APP1 (non-Exif) @%d len=%d head=%r" % (off, len(pl), pl[:80]))

for fp in sys.argv[1:]:
    show(fp)
    print()
