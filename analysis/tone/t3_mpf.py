"""手工解析 JPEG APP 段 + MPF 索引，找 Ultra HDR gain map。"""
import sys, os, struct, glob

def segments(path):
    d = open(path,'rb').read()
    assert d[:2]==b'\xff\xd8'
    i=2; segs=[]
    while i < len(d)-1:
        if d[i]!=0xFF: i+=1; continue
        m=d[i+1]
        if m in (0xD8,0xD9) or 0xD0<=m<=0xD7: i+=2; continue
        if m==0xDA:
            segs.append((m,i,struct.unpack('>H',d[i+2:i+4])[0],b'')); break
        L=struct.unpack('>H',d[i+2:i+4])[0]
        segs.append((m,i,L,d[i+4:i+2+L]))
        i+=2+L
    return d, segs

def parse_mpf(payload, mpf_offset_in_file):
    """payload 以 'MPF\\x00' 开头，其后是 TIFF 头。返回 [(offset,size,attr),...] 绝对文件偏移。"""
    tiff = payload[4:]
    base = mpf_offset_in_file + 4          # MP Endian 起始，所有 offset 相对于它
    bo = '<' if tiff[:2]==b'II' else '>'
    ifd0 = struct.unpack(bo+'I', tiff[4:8])[0]
    n = struct.unpack(bo+'H', tiff[ifd0:ifd0+2])[0]
    entries={}
    for j in range(n):
        e = tiff[ifd0+2+j*12: ifd0+14+j*12]
        tag,typ,cnt = struct.unpack(bo+'HHI', e[:8]); val = e[8:12]
        entries[tag]=(typ,cnt,val)
    out=[]
    if 0xB002 in entries:
        typ,cnt,val = entries[0xB002]
        off = struct.unpack(bo+'I', val)[0]
        num = cnt//16
        for j in range(num):
            r = tiff[off+j*16: off+j*16+16]
            attr, size, dataoff, d1, d2 = struct.unpack(bo+'IIIHH', r)
            abs_off = base+dataoff if dataoff!=0 else 0
            out.append((abs_off, size, attr))
    ver = entries.get(0xB000); num_img = entries.get(0xB001)
    return out, entries

files = sorted(glob.glob(sys.argv[1]))
for f in files:
    d, segs = segments(f)
    names=[]
    mpf=None
    for m,off,L,pl in segs:
        if 0xE0<=m<=0xEF:
            tag = pl.split(b'\x00')[0][:16]
            names.append(f"APP{m-0xE0}:{tag.decode('latin1','replace')}({L})")
            if pl[:4]==b'MPF\x00': mpf=(pl, off+4)
    print(f"\n=== {os.path.basename(f)}  size={os.path.getsize(f)} ===")
    print("  APP:", " ".join(names) if names else "(none)")
    if mpf:
        imgs, ent = parse_mpf(mpf[0], mpf[1])
        print("  MPF tags:", {hex(t):(v[0],v[1]) for t,v in ent.items()})
        for j,(o,s,a) in enumerate(imgs):
            typ = a & 0xFFFFFF
            print(f"  img{j}: abs_off={o} size={s} attr=0x{a:08x} MPType=0x{typ:06x}")
            if o and s:
                sub = d[o:o+s]
                print(f"        magic={sub[:4].hex()} ", end="")
                # 解析子图 SOF 拿尺寸/通道
                i=2
                while i < len(sub)-1:
                    if sub[i]!=0xFF: i+=1; continue
                    mm=sub[i+1]
                    if mm in (0xC0,0xC1,0xC2):
                        Lh=struct.unpack('>H',sub[i+2:i+4])[0]
                        prec=sub[i+4]; hh=struct.unpack('>H',sub[i+5:i+7])[0]; ww=struct.unpack('>H',sub[i+7:i+9])[0]; nc=sub[i+9]
                        print(f"SOF{mm-0xC0} {ww}x{hh} ncomp={nc} prec={prec} ", end="")
                        break
                    if mm==0xDA: break
                    if mm in (0xD8,0xD9) or 0xD0<=mm<=0xD7: i+=2; continue
                    i+=2+struct.unpack('>H',sub[i+2:i+4])[0]
                # 子图 APP 段
                sn=[]
                try:
                    _,ss = None,None
                    i2=2
                    while i2 < len(sub)-1:
                        if sub[i2]!=0xFF: i2+=1; continue
                        m2=sub[i2+1]
                        if m2==0xDA: break
                        if m2 in (0xD8,0xD9) or 0xD0<=m2<=0xD7: i2+=2; continue
                        L2=struct.unpack('>H',sub[i2+2:i2+4])[0]
                        if 0xE0<=m2<=0xEF: sn.append(f"APP{m2-0xE0}:{sub[i2+4:i2+4+L2-2].split(b'*chr*')[0][:24]!r}"[:60])
                        i2+=2+L2
                except Exception as e: pass
                print("subAPP:", sn)
