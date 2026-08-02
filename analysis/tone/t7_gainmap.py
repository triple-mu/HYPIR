import struct, glob, os, io
import numpy as np
from PIL import Image
TEST="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
NATIVE=[1,2,3,4,5,38,64,65,66]
os.makedirs("gm", exist_ok=True)
def parse_iso(b):
    minv,wrv=struct.unpack('>HH',b[0:4]); fl=b[4]; o=5
    is_mc=(fl>>7)&1
    def rat(s=False):
        nonlocal o
        n=struct.unpack('>i' if s else '>I', b[o:o+4])[0]; d=struct.unpack('>I',b[o+4:o+8])[0]; o+=8
        return n/d
    bh=rat(); ah=rat(); out=[]
    for c in range(3 if is_mc else 1):
        out.append(dict(gmin=rat(True),gmax=rat(True),gamma=rat(),boff=rat(True),aoff=rat(True)))
    return dict(is_mc=is_mc, base_hr=bh, alt_hr=ah, ch=out)

for n in NATIVE:
    p=f"{TEST}/case{n}.jpg"; d=open(p,'rb').read()
    # MPF: 第二张子图
    j=d.find(b'MPF\x00'); tiff=d[j+4:]; base=j+4
    bo='<' if tiff[:2]==b'II' else '>'
    ifd0=struct.unpack(bo+'I',tiff[4:8])[0]; nent=struct.unpack(bo+'H',tiff[ifd0:ifd0+2])[0]
    ent={}
    for q in range(nent):
        e=tiff[ifd0+2+q*12:ifd0+14+q*12]; tag,typ,cnt=struct.unpack(bo+'HHI',e[:8]); ent[tag]=(typ,cnt,e[8:12])
    off=struct.unpack(bo+'I',ent[0xB002][2])[0]; num=ent[0xB002][1]//16
    imgs=[]
    for q in range(num):
        attr,size,dof,_,_=struct.unpack(bo+'IIIHH', tiff[off+q*16:off+q*16+16])
        imgs.append((base+dof if dof else 0, size, attr))
    gm_off, gm_size, gm_attr = imgs[1]
    sub=d[gm_off:gm_off+gm_size]
    im=Image.open(io.BytesIO(sub)); a=np.asarray(im)
    # 元数据
    k=sub.find(b'urn:iso:std:iso:ts:21496'); L=struct.unpack('>H',sub[k-2:k])[0]
    md=parse_iso(sub[k:k+L-2][28:])
    base_im=Image.open(io.BytesIO(d[:imgs[0][1] if imgs[0][1] else len(d)]))
    ch=md['ch'][0]
    # gain map 是灰度还是彩色？
    if a.ndim==3:
        mx=[float(np.abs(a[...,0].astype(int)-a[...,c].astype(int)).mean()) for c in (1,2)]
    else: mx=None
    print(f"case{n}: base={base_im.size} gainmap={im.size} mode={im.mode} attr=0x{gm_attr:08x} bytes={gm_size}")
    print(f"   ISO21496: multich={md['is_mc']} base_hr={md['base_hr']:.4f} alt_hr={md['alt_hr']:.6f}st ({2**md['alt_hr']:.3f}x) "
          f"gmin={ch['gmin']:+.4f} gmax={ch['gmax']:+.4f} gamma={ch['gamma']:.3f}")
    print(f"   gainmap px: mean={a.mean():.2f} std={a.std():.2f} min={a.min()} max={a.max()} "
          f"p1={np.percentile(a,1):.0f} p50={np.percentile(a,50):.0f} p99={np.percentile(a,99):.0f} "
          f"frac(==0)={100*(a==0).mean():.2f}% inter-ch MAD={mx}")
    np.save(f"gm/case{n}.npy", a)
    # ICC
    icc = d.find(b'ICC_PROFILE\x00')
    if icc>0:
        prof=d[icc+14: icc+14+struct.unpack('>H',d[icc-2:icc])[0]-16]
        desc=b'desc' in prof
        # 取 profile description
        print(f"   ICC {len(prof)}B cmm={prof[4:8]} class={prof[12:16]} space={prof[16:20]} pcs={prof[20:24]}")
