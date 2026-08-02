import sys, os, struct, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import scan_segments

def icc_dump(p):
    hdr = p
    size, cmm, ver = struct.unpack(">I4sI", hdr[0:12])
    cls, space, pcs = hdr[12:16], hdr[16:20], hdr[20:24]
    dt = struct.unpack(">6H", hdr[24:36])
    sig, plat = hdr[36:40], hdr[40:44]
    flags, manu, model = struct.unpack(">I4s4s", hdr[44:56])
    rend = struct.unpack(">I", hdr[64:68])[0]
    creator = hdr[80:84]
    print("   ICC size=%d cmm=%r ver=0x%08x class=%r space=%r pcs=%r date=%s"%(size,cmm,ver,cls,space,pcs,dt))
    print("   platform=%r manuf=%r model=%r rendering=%d creator=%r"%(plat,manu,model,rend,creator))
    n = struct.unpack(">I", hdr[128:132])[0]
    print("   tags=%d"%n)
    for i in range(n):
        s, off, ln = struct.unpack(">4sII", hdr[132+i*12:144+i*12])
        raw = hdr[off:off+ln]
        val = ""
        if raw[:4] in (b"desc", b"mluc", b"text"):
            txt = re.findall(rb"[\x20-\x7e]{3,}", raw[8:])
            u = re.findall(rb"(?:[\x20-\x7e]\x00){3,}", raw[8:])
            val = str([t.decode() for t in txt][:4]) + str([t.decode("utf-16-be" if raw[8:10]!=b"\x00\x00" else "utf-16-le","replace") for t in u][:2] if u else "")
            if raw[:4]==b"mluc":
                cnt=struct.unpack(">I",raw[8:12])[0]
                rs=struct.unpack(">I",raw[12:16])[0]
                recs=[]
                for k in range(cnt):
                    lang,cty,l,o=struct.unpack(">2s2sII",raw[16+k*rs:16+k*rs+12])
                    recs.append((lang.decode(),cty.decode(),raw[o:o+l].decode("utf-16-be","replace")))
                val=str(recs)
        elif raw[:4]==b"XYZ ":
            v=struct.unpack(">3i",raw[8:20]); val=str([x/65536.0 for x in v])
        elif raw[:4]==b"para":
            val="para type=%d"%struct.unpack(">H",raw[8:10])[0]
        elif raw[:4]==b"curv":
            c=struct.unpack(">I",raw[8:12])[0]; val="curv n=%d"%c
        print("     %-6s off=%-6d len=%-6d type=%r %s"%(s.decode(),off,ln,raw[:4],val[:160]))

for fp in sys.argv[1:]:
    data = open(fp,"rb").read()
    segs,_ = scan_segments(data)
    print("======",os.path.basename(fp))
    seen=set()
    for m,name,off,pl in segs:
        if 0xE0<=m<=0xEF:
            if pl.startswith(b"ICC_PROFILE\x00"):
                key=("icc",len(pl))
                if key in seen: continue
                seen.add(key)
                print("  %s@%d ICC_PROFILE seq=%d/%d len=%d"%(name,off,pl[12],pl[13],len(pl)))
                icc_dump(pl[14:])
            elif pl.startswith(b"urn:iso:std:iso:ts:21496"):
                print("  %s@%d ISO21496 len=%d payload=%r"%(name,off,len(pl),pl[:min(120,len(pl))]))
            elif pl.startswith(b"ITUT35"):
                print("  %s@%d ITUT35 len=%d hex=%s"%(name,off,len(pl),pl[:60].hex()))
    print()
