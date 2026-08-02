"""扫描文件里所有 SOI/EOI，确认真实的子图边界；并 dump ISO21496 payload。"""
import sys, struct, os
f=sys.argv[1]; d=open(f,'rb').read()
print("filesize", len(d))
# 找所有 FFD8FF
pos=[]; i=0
while True:
    j=d.find(b'\xff\xd8\xff', i)
    if j<0: break
    pos.append(j); i=j+1
print("SOI-like positions:", pos[:20])
# 找 EOI
i=0; eoi=[]
while True:
    j=d.find(b'\xff\xd9', i)
    if j<0: break
    eoi.append(j); i=j+1
print("num FFD9:", len(eoi), "last few:", eoi[-5:])
# 逐个 SOI 尝试解析 SOF
for p in pos[:10]:
    sub=d[p:]
    i=2; info=None
    while i<len(sub)-1 and i < 200000:
        if sub[i]!=0xFF: i+=1; continue
        m=sub[i+1]
        if m in (0xC0,0xC1,0xC2):
            L=struct.unpack('>H',sub[i+2:i+4])[0]
            prec=sub[i+4]; hh=struct.unpack('>H',sub[i+5:i+7])[0]; ww=struct.unpack('>H',sub[i+7:i+9])[0]; nc=sub[i+9]
            comps=[]
            for c in range(nc):
                cid=sub[i+10+c*3]; hv=sub[i+11+c*3]; tq=sub[i+12+c*3]
                comps.append(f"id{cid}:{hv>>4}x{hv&15}q{tq}")
            info=f"SOF{m-0xC0} {ww}x{hh} nc={nc} [{' '.join(comps)}]"
            break
        if m==0xDA: break
        if m in (0xD8,0xD9) or 0xD0<=m<=0xD7: i+=2; continue
        i+=2+struct.unpack('>H',sub[i+2:i+4])[0]
    print(f"  @{p}: {info}")
# dump 所有 urn:iso APP2 内容
i=0
while True:
    j=d.find(b'urn:iso:std:iso', i)
    if j<0: break
    seglen=struct.unpack('>H', d[j-2:j])[0]
    print(f"\n  ISO21496 APP2 @ {j-4} len={seglen}: {d[j:j+seglen-2][:64]!r}")
    print(f"    hex: {d[j:j+seglen-2].hex()}")
    i=j+1
