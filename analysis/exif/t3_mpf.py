"""扫全部测试图 + 验证集：JPEG APP 段清单、MPF 子图、XMP(Ultra HDR / GContainer) 标记。"""
import os, sys, struct, glob, json

def segments(path):
    d = open(path, "rb").read()
    assert d[:2] == b"\xff\xd8", path
    i = 2
    segs = []
    while i < len(d) - 1:
        if d[i] != 0xFF:
            i += 1; continue
        m = d[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2; continue
        if m == 0xDA:  # SOS -> 后面是熵编码数据
            L = struct.unpack(">H", d[i + 2:i + 4])[0]
            segs.append((m, i, L, d[i + 4:i + 4 + L - 2]))
            i += 2 + L
            # 跳到下一个非 RSTn 的 marker
            while i < len(d) - 1:
                if d[i] == 0xFF and d[i + 1] != 0x00 and not (0xD0 <= d[i + 1] <= 0xD7):
                    break
                i += 1
            continue
        if m == 0xD9:
            segs.append((m, i, 0, b"")); i += 2; continue
        L = struct.unpack(">H", d[i + 2:i + 4])[0]
        segs.append((m, i, L, d[i + 4:i + 4 + L - 2]))
        i += 2 + L
    return d, segs

def parse_mpf(payload):
    """payload 以 'MPF\\x00' 开头，之后是 TIFF 头 + MP Index IFD。"""
    tif = payload[4:]
    bo = "<" if tif[:2] == b"II" else ">"
    off = struct.unpack(bo + "I", tif[4:8])[0]
    n = struct.unpack(bo + "H", tif[off:off + 2])[0]
    entries = {}
    for k in range(n):
        p = off + 2 + k * 12
        tag, typ, cnt = struct.unpack(bo + "HHI", tif[p:p + 8])
        val = tif[p + 8:p + 12]
        entries[tag] = (typ, cnt, val)
    imgs = []
    if 0xB002 in entries:  # MP Entry
        typ, cnt, val = entries[0xB002]
        eoff = struct.unpack(bo + "I", val)[0]
        num = cnt // 16
        for k in range(num):
            p = eoff + k * 16
            attr, size, doff, d1, d2 = struct.unpack(bo + "IIIHH", tif[p:p + 16])
            imgs.append({"attr": hex(attr), "type": hex(attr & 0xFFFFFF), "size": size,
                         "data_off": doff})
    return imgs, list(map(hex, entries.keys()))

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
files = sorted(glob.glob(TEST + "/*.jpg"), key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
files += sorted(glob.glob(VAL + "/*"))

res = {}
for f in files:
    head = open(f, "rb").read(4)
    if head[:2] != b"\xff\xd8":
        res[f] = {"fmt": "notjpeg"}; continue
    d, segs = segments(f)
    apps = []
    mpf = None; xmp = []
    for m, i, L, pl in segs:
        if 0xE0 <= m <= 0xEF:
            tag = pl.split(b"\x00")[0][:24]
            apps.append((f"APP{m-0xE0}", tag.decode("latin1", "replace"), L))
            if pl.startswith(b"MPF\x00"):
                try:
                    mpf = parse_mpf(pl)
                except Exception as e:
                    mpf = ("err", str(e))
            if b"http://ns.adobe.com/xap" in pl[:40] or b"xmpmeta" in pl:
                xmp.append(pl.decode("latin1", "replace"))
    txt = " ".join(xmp)
    flags = {k: (k.lower() in txt.lower()) for k in
             ["GainMap", "hdrgm", "GContainer", "Item:Semantic", "RecoveryMap", "apple", "hdr"]}
    res[os.path.basename(f)] = {
        "size": os.path.getsize(f),
        "apps": apps,
        "mpf": mpf,
        "n_soi": d.count(b"\xff\xd8\xff"),
        "xmp_len": sum(len(x) for x in xmp),
        "xmp_flags": {k: v for k, v in flags.items() if v},
    }

json.dump(res, open("/home/ubuntu/workspace/contest/CSIG-2026/Moebius/analysis_tone/t3.json", "w"), indent=1)

# 摘要
withmpf = [k for k, v in res.items() if v.get("mpf")]
print("total", len(res), "with MPF:", withmpf)
for k in withmpf:
    v = res[k]
    print(" ", k, v["size"], "n_soi", v["n_soi"], "xmp", v["xmp_len"], v["xmp_flags"])
    print("     apps:", v["apps"])
    print("     mpf imgs:", v["mpf"][0] if isinstance(v["mpf"], tuple) else v["mpf"])
print("--- xmp non-empty:", [(k, v["xmp_len"], v["xmp_flags"]) for k, v in res.items() if v.get("xmp_len")])
print("--- multi-SOI:", [(k, v["n_soi"]) for k, v in res.items() if v.get("n_soi", 0) > 1])
