import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments, parse_tiff, TYPE_SIZE

def get_app1(fp):
    data = open(fp, "rb").read()
    segs, _ = scan_segments(data)
    for m, name, off, pl in segs:
        if name == "APP1" and pl.startswith(b"Exif\x00\x00"):
            return pl[6:]
    return None

def hexdump(b, base=0, width=16, maxlen=4096):
    out = []
    for i in range(0, min(len(b), maxlen), width):
        ch = b[i:i+width]
        out.append("%08x  %-48s |%s|" % (base+i, " ".join("%02x"%x for x in ch),
                   "".join(chr(c) if 32<=c<127 else "." for c in ch)))
    return "\n".join(out)

fp = sys.argv[1]
tiff = get_app1(fp)
print("TIFF block len =", len(tiff))
r = parse_tiff(tiff)
endian = r["endian"]

# 覆盖图：哪些字节被 IFD 结构/值占用
cov = bytearray(len(tiff))
cov[0:8] = b"\x01"*8
for ifdname, ents in r["ifds"].items():
    if not ents: continue
    start = None
    # 找 IFD 起点：entry_offset of first - 2
    start = ents[0]["entry_offset"] - 2
    end = ents[-1]["entry_offset"] + 12 + 4
    for k in range(start, min(end, len(tiff))): cov[k] = 1
    for e in ents:
        if e["value_offset"] is not None:
            o = e["value_offset"]; s = e["size"]
            for k in range(o, min(o+s, len(tiff))): cov[k] = 1
gaps = []
i = 0
while i < len(tiff):
    if cov[i] == 0:
        j = i
        while j < len(tiff) and cov[j] == 0: j += 1
        gaps.append((i, j-i))
        i = j
    else:
        i += 1
print("未被引用的字节区间 (offset, len):", [(o,l) for o,l in gaps if l > 0])
for o, l in gaps:
    if l >= 4:
        print("  gap @%d len=%d: %r" % (o, l, tiff[o:o+min(l,200)]))

print()
print("=== IFD0 raw ===")
ifd0_off = r["ifd0_offset"]
n = struct.unpack(endian+"H", tiff[ifd0_off:ifd0_off+2])[0]
print(hexdump(tiff[ifd0_off:ifd0_off+2+12*n+4], ifd0_off))
print()
ex = [e for e in r["ifds"]["IFD0"] if e["name"]=="ExifIFDPointer"]
if ex:
    eo = ex[0]["value"][0]
    n2 = struct.unpack(endian+"H", tiff[eo:eo+2])[0]
    print("=== ExifIFD raw (%d entries) @%d ===" % (n2, eo))
    print(hexdump(tiff[eo:eo+2+12*n2+4], eo))
