"""提取 MPF 子图 + 解析 ISO 21496-1 gain map 元数据。"""
import os, struct, glob, json, io
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
OUT = "/home/ubuntu/workspace/contest/CSIG-2026/Moebius/analysis_tone/gm"
os.makedirs(OUT, exist_ok=True)

def find_segs(d, want):
    """返回 (marker, abs_offset, payload) 列表。"""
    i, out = 2, []
    n = len(d)
    while i < n - 1:
        if d[i] != 0xFF: i += 1; continue
        m = d[i+1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7: i += 2; continue
        if m == 0xD9: i += 2; continue
        if i+4 > n: break
        L = struct.unpack(">H", d[i+2:i+4])[0]
        pl = d[i+4:i+2+L]
        if m in want: out.append((m, i, pl))
        if m == 0xDA:
            i += 2 + L
            while i < n-1:
                if d[i] == 0xFF and d[i+1] != 0x00 and not (0xD0 <= d[i+1] <= 0xD7): break
                i += 1
            continue
        i += 2 + L
    return out

def mpf_entries(pl, seg_abs_off):
    tif = pl[4:]
    tif_base = seg_abs_off + 4 + 4   # MPF header 起点 = 段数据起点+4；偏移基准是 TIFF 头起点
    bo = "<" if tif[:2] == b"II" else ">"
    off = struct.unpack(bo+"I", tif[4:8])[0]
    n = struct.unpack(bo+"H", tif[off:off+2])[0]
    ent = {}
    for k in range(n):
        p = off + 2 + k*12
        tag, typ, cnt = struct.unpack(bo+"HHI", tif[p:p+8])
        ent[tag] = (typ, cnt, tif[p+8:p+12], p)
    imgs = []
    if 0xB002 in ent:
        typ, cnt, val, _ = ent[0xB002]
        eoff = struct.unpack(bo+"I", val)[0]
        for k in range(cnt // 16):
            p = eoff + k*16
            attr, size, doff, d1, d2 = struct.unpack(bo+"IIIHH", tif[p:p+16])
            imgs.append({"attr": hex(attr), "size": size, "off": doff})
    return imgs, bo, {hex(k): (v[0], v[1]) for k, v in ent.items()}, tif_base

def parse_iso21496(pl):
    """ISO/TS 21496-1 gain map metadata（APP2 'urn:iso:std:iso:ts:21496-1\\0' 之后）。
    结构（big endian）: version(u16) minimum_version(u16) channel_count(u8) flags(u8)
    denominators..., 然后每通道 gain_min/gain_max/gamma/base_offset/alt_offset。"""
    j = pl.find(b"\x00")
    body = pl[j+1:]
    return body.hex(), len(body)

res = {}
for f in sorted(glob.glob(TEST+"/*.jpg"), key=lambda p:int(''.join(c for c in os.path.basename(p) if c.isdigit()))):
    d = open(f, "rb").read()
    segs = find_segs(d, {0xE2, 0xE1})
    mpfs = [(m,i,pl) for m,i,pl in segs if pl.startswith(b"MPF\x00")]
    if not mpfs: continue
    name = os.path.basename(f)
    m, i, pl = mpfs[0]
    imgs, bo, tags, tif_base = mpf_entries(pl, i)
    isos = [pl2 for m2,i2,pl2 in segs if pl2.startswith(b"urn:iso:std:iso:ts:21496")]
    rec = {"file": f, "size": len(d), "mpf_tags": tags, "imgs": imgs,
           "iso21496": [parse_iso21496(x) for x in isos]}
    # 提取子图：第 1 张的 off=0 表示从 TIFF 头起点（实际是整文件起点），后续的 off 相对 TIFF 头
    subs = []
    for k, im in enumerate(imgs):
        start = 0 if im["off"] == 0 else tif_base + im["off"]
        blob = d[start:start+im["size"]]
        p = f"{OUT}/{name[:-4]}_sub{k}.jpg"
        open(p, "wb").write(blob)
        try:
            img = Image.open(p); img.load()
            subs.append({"path": p, "mode": img.mode, "size": img.size, "bytes": len(blob),
                         "magic": blob[:4].hex()})
        except Exception as e:
            subs.append({"path": p, "err": str(e), "bytes": len(blob), "magic": blob[:4].hex()})
    rec["subs"] = subs
    res[name] = rec
    print(name, [ (s.get("size"), s.get("mode"), s.get("bytes")) for s in subs], flush=True)

json.dump(res, open("/home/ubuntu/workspace/contest/CSIG-2026/Moebius/analysis_tone/t3b.json","w"), indent=1)
for k,v in res.items():
    print("==", k, "iso21496 payloads:", [(l, h[:200]) for h,l in v["iso21496"]])
