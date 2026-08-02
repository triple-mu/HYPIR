"""对全部 100 张测试图 + 3 张验证 LQ/GT：量化表指纹、APP 段、MPF/gain map 存在性。"""
import struct, glob, os, sys, json
import numpy as np

def scan(path):
    d=open(path,'rb').read()
    if d[:2]!=b'\xff\xd8': return dict(fmt='PNG' if d[:4]==b'\x89PNG' else 'other', size=len(d))
    i=2; apps=[]; qt={}; sof=None; comps=None; tail=None
    while i < len(d)-1:
        if d[i]!=0xFF: i+=1; continue
        m=d[i+1]
        if m in (0xD8,0xD9) or 0xD0<=m<=0xD7: i+=2; continue
        if m==0xDA:
            break
        L=struct.unpack('>H',d[i+2:i+4])[0]; pl=d[i+4:i+2+L]
        if 0xE0<=m<=0xEF:
            tag=pl.split(b'\x00')[0][:14].decode('latin1','replace')
            apps.append((f"APP{m-0xE0}",tag,L))
        elif m==0xDB:
            o=0
            while o < len(pl):
                pq,tq = pl[o]>>4, pl[o]&15; o+=1
                n=64*(2 if pq else 1)
                t=np.frombuffer(pl[o:o+n], '>u2' if pq else 'u1').astype(int); o+=n
                qt[tq]=t
        elif m in (0xC0,0xC1,0xC2):
            prec=pl[0]; hh=struct.unpack('>H',pl[1:3])[0]; ww=struct.unpack('>H',pl[3:5])[0]; nc=pl[5]
            sof=(f"SOF{m-0xC0}",ww,hh,nc)
            comps=[(pl[6+c*3], pl[7+c*3]>>4, pl[7+c*3]&15, pl[8+c*3]) for c in range(nc)]
        i+=2+L
    fp = "|".join(f"T{k}[dc={v[0]},mean={v.mean():.1f},max={v.max()}]" for k,v in sorted(qt.items()))
    has_gm = b'urn:iso:std:iso:ts:21496' in d
    n_iso  = d.count(b'urn:iso:std:iso:ts:21496')
    has_mpf= any(a[1]=='MPF' for a in apps)
    has_hw = b'\xff\xe7\xf2P' in d or b'HUAWEI\x00\x00II' in d
    nsoi = 0; j=0
    while True:
        j=d.find(b'\xff\xd8\xff', j)
        if j<0: break
        nsoi+=1; j+=1
    return dict(size=len(d), fp=fp, sof=sof, comps=comps,
                apps=[f"{a[0]}:{a[1]}({a[2]})" for a in apps],
                gainmap=has_gm, n_iso=n_iso, mpf=has_mpf, huawei=has_hw, nsoi=nsoi)

TEST="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL ="/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
rows={}
for p in sorted(glob.glob(TEST+"/*.jpg"), key=lambda x:int(''.join(filter(str.isdigit,os.path.basename(x))))):
    rows[os.path.basename(p)]=scan(p)
for p in sorted(glob.glob(VAL+"/*")):
    rows["VAL/"+os.path.basename(p)]=scan(p)

# 按量化表指纹分组
from collections import defaultdict
grp=defaultdict(list)
for k,v in rows.items():
    if 'fp' in v: grp[v['fp']].append(k)
print("=== 量化表指纹分组 ===")
for fp, ks in sorted(grp.items(), key=lambda x:-len(x[1])):
    print(f"\n[{len(ks)} 张] {fp}")
    print("  ", " ".join(sorted(ks, key=lambda s:(not s.startswith('VAL'), s))[:110]))
print("\n=== 特殊标记 ===")
sp=[k for k,v in rows.items() if v.get('gainmap') or v.get('mpf') or v.get('huawei')]
for k in sp:
    v=rows[k]
    print(f"{k:16s} size={v['size']:>9d} SOI={v['nsoi']} mpf={v['mpf']} gainmap={v['gainmap']}(x{v['n_iso']}) huawei={v['huawei']}")
    print(f"    apps: {' '.join(v['apps'])}")
print(f"\n有 gain map: {sum(1 for v in rows.values() if v.get('gainmap'))} / {len(rows)}")
print(f"有 MPF    : {sum(1 for v in rows.values() if v.get('mpf'))}")
json.dump({k:{kk:(list(map(int,vv)) if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} for k,v in rows.items()}, open('classify.json','w'), default=str, indent=0)
