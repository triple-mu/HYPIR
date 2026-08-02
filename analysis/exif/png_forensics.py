import sys, os, struct, zlib, glob, hashlib
from collections import Counter

V = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"

COLOR = {0:"Gray",2:"RGB",3:"Palette",4:"GrayA",6:"RGBA"}

def png_report(fp):
    d = open(fp, "rb").read()
    print("#### %s  (%d B)" % (os.path.basename(fp), len(d)))
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        print("   NOT PNG, magic=%r" % d[:8]); return
    i = 8
    idat = []
    chunks = []
    while i < len(d):
        ln = struct.unpack(">I", d[i:i+4])[0]
        typ = d[i+4:i+8]
        data = d[i+8:i+8+ln]
        crc = struct.unpack(">I", d[i+8+ln:i+12+ln])[0]
        ok = (zlib.crc32(typ+data) & 0xffffffff) == crc
        chunks.append((typ.decode("latin1"), ln, i, ok))
        if typ == b"IHDR":
            w,h,bd,ct,cm,fm,il = struct.unpack(">IIBBBBB", data)
            print("   IHDR %dx%d bitdepth=%d color=%d(%s) compression=%d filter=%d interlace=%d"
                  % (w,h,bd,ct,COLOR.get(ct,"?"),cm,fm,il))
            IH=(w,h,bd,ct,il)
        elif typ == b"IDAT":
            idat.append(data)
        elif typ in (b"tEXt", b"iTXt", b"zTXt"):
            print("   %s: %r" % (typ.decode(), data[:200]))
        elif typ == b"pHYs":
            px,py,u = struct.unpack(">IIB", data)
            print("   pHYs x=%d y=%d unit=%d (%.1f dpi)" % (px,py,u, px*0.0254 if u==1 else 0))
        elif typ == b"tIME":
            print("   tIME", struct.unpack(">HBBBBB", data))
        elif typ == b"gAMA":
            print("   gAMA", struct.unpack(">I", data)[0])
        elif typ == b"sRGB":
            print("   sRGB rendering intent =", data[0])
        elif typ == b"iCCP":
            nm = data.split(b"\x00")[0]
            print("   iCCP name=%r comp=%d len=%d" % (nm, data[len(nm)+1], ln))
        elif typ == b"cHRM":
            print("   cHRM", struct.unpack(">8I", data))
        elif typ == b"eXIf":
            print("   eXIf len=%d head=%r" % (ln, data[:40]))
        elif typ == b"IEND":
            pass
        i += 12 + ln
    print("   chunk order:", " ".join("%s(%d)" % (c[0], c[1]) for c in chunks[:8]),
          "... total IDAT=%d" % sum(1 for c in chunks if c[0]=="IDAT"))
    print("   all chunk types:", Counter(c[0] for c in chunks))
    bad = [c for c in chunks if not c[3]]
    print("   CRC bad:", bad)
    sizes = [c[1] for c in chunks if c[0]=="IDAT"]
    if sizes:
        print("   IDAT count=%d sizes: first=%d, uniq(non-last)=%s, last=%d, total=%d"
              % (len(sizes), sizes[0], sorted(set(sizes[:-1]))[:5], sizes[-1], sum(sizes)))
    raw = b"".join(idat)
    # zlib 头
    cmf, flg = raw[0], raw[1]
    flevel = flg >> 6
    print("   zlib CMF=0x%02x FLG=0x%02x  method=%d windowbits=%d FLEVEL=%d(%s) FDICT=%d"
          % (cmf, flg, cmf&15, (cmf>>4)+8, flevel,
             ["fastest","fast","default","maximum"][flevel], (flg>>5)&1))
    dec = zlib.decompressobj()
    out = dec.decompress(raw)
    w,h,bd,ct,il = IH
    nch = {0:1,2:3,3:1,4:2,6:4}[ct]
    bpp = max(1, nch*bd//8)
    stride = (w*nch*bd + 7)//8
    filt = Counter()
    p = 0
    for y in range(h):
        if p >= len(out): break
        filt[out[p]] += 1
        p += 1 + stride
    print("   decompressed=%d (expect %d)  filter type histogram: %s"
          % (len(out), h*(1+stride), dict(sorted(filt.items()))))
    # 重压缩比较：不同工具的压缩效率
    print("   IDAT bytes=%d  ratio=%.4f" % (len(raw), len(raw)/len(out)))
    return raw, out, IH

for fp in sorted(glob.glob(V+"/*")):
    d = open(fp,"rb").read()
    magic = "PNG" if d[:8]==b"\x89PNG\r\n\x1a\n" else ("JPEG" if d[:2]==b"\xff\xd8" else "?")
    if magic == "PNG":
        png_report(fp)
        print()
