import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import parse_ifd, TYPE_NAME

MP_TAGS = {0xb000:"MPFVersion",0xb001:"NumberOfImages",0xb002:"MPEntry",0xb003:"ImageUIDList",
           0xb004:"TotalFrames",0xb101:"MPIndividualNum",0xb201:"PanOrientation",
           0xb202:"PanOverlapH",0xb203:"PanOverlapV",0xb204:"BaseViewpointNum",
           0xb205:"ConvergenceAngle",0xb206:"BaselineLength",0xb207:"VerticalDivergence",
           0xb208:"AxisDistanceX",0xb209:"AxisDistanceY",0xb20a:"AxisDistanceZ",
           0xb20b:"YawAngle",0xb20c:"PitchAngle",0xb20d:"RollAngle"}

def find_all_segments(data):
    out = []
    n = len(data); i = 0
    while i < n-1:
        if data[i] != 0xFF: i += 1; continue
        m = data[i+1]
        if m in (0x00,0xFF) or 0xD0<=m<=0xD7: i += 1; continue
        if m in (0xD8,0xD9): out.append((m,i,b"")); i += 2; continue
        if i+4>n: break
        ln = struct.unpack(">H", data[i+2:i+4])[0]
        out.append((m,i,data[i+4:i+2+ln]))
        i = i+2+ln
        if m == 0xDA:
            j = i
            while j<n-1:
                if data[j]==0xFF and data[j+1] not in (0x00,0xFF) and not (0xD0<=data[j+1]<=0xD7): break
                j += 1
            i = j
    return out

def sof_at(data, off):
    """从 off 处开始（SOI），找第一个 SOFn，返回 (w,h,comps)"""
    n=len(data); i=off+2
    while i<n-1 and data[i]==0xFF:
        m=data[i+1]
        ln=struct.unpack(">H",data[i+2:i+4])[0]
        if m in (0xC0,0xC1,0xC2):
            pl=data[i+4:i+2+ln]
            h,w=struct.unpack(">HH",pl[1:5]); nc=pl[5]
            comps=tuple((pl[6+c*3],pl[7+c*3]>>4,pl[7+c*3]&15,pl[8+c*3]) for c in range(nc))
            return w,h,comps
        if m==0xDA: return None
        i=i+2+ln
    return None

for fp in sys.argv[1:]:
    data=open(fp,"rb").read()
    print("=== %s (%d B) ==="%(os.path.basename(fp),len(data)))
    segs=find_all_segments(data)
    for m,off,pl in segs:
        if 0xE0<=m<=0xEF and pl.startswith(b"MPF\x00"):
            tiff=pl[4:]
            base_in_file=off+4+4   # APP2 payload starts at off+4; TIFF header at off+4+4
            endian = "<" if tiff[0:2]==b"II" else ">"
            ifd_off=struct.unpack(endian+"I",tiff[4:8])[0]
            ents,nxt=parse_ifd(tiff,endian,ifd_off,MP_TAGS)
            print(" MPF APP2 @%d len=%d  endian=%s"%(off,len(pl),endian))
            for e in ents:
                print("   %-16s %-10s cnt=%-5d %s"%(e["name"],e["type"],e["count"],
                      e["value"] if e["name"]!="MPEntry" else ""))
                if e["name"]=="MPEntry":
                    raw=e["raw"]; k=len(raw)//16
                    for x in range(k):
                        attr,size,doff,d1,d2=struct.unpack(endian+"IIIHH",raw[x*16:x*16+16])
                        typ=attr&0xFFFFFF
                        real = doff if doff==0 else doff+base_in_file
                        s=sof_at(data,real) if real<len(data) and data[real:real+2]==b"\xff\xd8" else None
                        print("     img%d attr=0x%08x type=0x%06x size=%d dataoff=%d -> abs=%d SOF=%s"%(
                            x,attr,typ,size,doff,real,s[:2] if s else None))
    # 所有 SOI
    sois=[off for m,off,pl in segs if m==0xD8]
    print(" SOI offsets:",sois, " SOF:",[sof_at(data,o)[:2] if sof_at(data,o) else None for o in sois])
    eois=[off for m,off,pl in segs if m==0xD9]
    print(" EOI offsets:",eois, " tail after last EOI:", len(data)-(eois[-1]+2) if eois else None)
    print()
