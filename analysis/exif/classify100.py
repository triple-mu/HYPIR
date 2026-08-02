import sys, os, struct, glob, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegmeta import MARKER_NAMES

TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"

def parse_top(data):
    """只解析第一帧（到第一个 SOS 为止），并额外记录文件里所有 SOI/EOI 位置。"""
    n = len(data); i = 2
    apps = []; dqt = []; sof = None; dht = 0; dri = None; sos_off = None
    while i < n - 1:
        if data[i] != 0xFF: break
        m = data[i+1]
        if m in (0xD8, 0xD9): i += 2; continue
        ln = struct.unpack(">H", data[i+2:i+4])[0]
        pl = data[i+4:i+2+ln]
        name = MARKER_NAMES.get(m, "0x%02X"%m)
        if 0xE0 <= m <= 0xEF:
            apps.append((name, ln, pl.split(b"\x00")[0][:24].decode("latin1","replace")))
        elif name == "DQT":
            p = 0
            while p < len(pl):
                pq, tq = pl[p] >> 4, pl[p] & 15
                if pq == 0:
                    t = list(pl[p+1:p+65]); p += 65
                else:
                    t = list(struct.unpack(">64H", pl[p+1:p+129])); p += 129
                dqt.append((tq, t))
        elif name.startswith("SOF"):
            prec = pl[0]; h, w = struct.unpack(">HH", pl[1:5]); nc = pl[5]
            comps = tuple((pl[6+c*3], pl[7+c*3]>>4, pl[7+c*3]&15, pl[8+c*3]) for c in range(nc))
            sof = (name, prec, w, h, comps)
        elif name == "DHT": dht += 1
        elif name == "DRI": dri = struct.unpack(">H", pl[:2])[0]
        elif name == "SOS":
            sos_off = i; break
        i = i + 2 + ln
    return apps, dqt, sof, dht, dri, sos_off

def dqtsig(dqt):
    return "|".join("T%d:dc=%d,mean=%.1f,max=%d,md5=%s" % (tq, t[0], sum(t)/64.0, max(t), hashlib.md5(bytes(t)).hexdigest()[:8]) for tq, t in dqt)

files = sorted(glob.glob(TEST+"/case*.jpg"), key=lambda p:int(re.search(r"case(\d+)",p).group(1)))
files += sorted(glob.glob(VAL+"/*_lq.jpg"))
rows = []
for fp in files:
    data = open(fp,"rb").read()
    apps, dqt, sof, dht, dri, sos = parse_top(data)
    nsoi = data.count(b"\xff\xd8\xff")
    rows.append((os.path.basename(fp), len(data), sof, dqtsig(dqt), dht, dri,
                 "|".join("%s(%d)"%(a[0],a[1]) for a in apps), nsoi))

from collections import Counter, defaultdict
c = Counter()
for r in rows: c[(r[3], r[4], r[5], r[6] if "Exif" not in r[6] else "EXIF-NATIVE")] += 1
print("=== 结构指纹分类 ===")
for k, v in c.most_common():
    print("n=%-4d DQT=%s  DHT=%d DRI=%s APPS=%s" % (v, k[0], k[1], k[2], k[3]))
print()
print("=== 每张 ===")
print("%-16s %-10s %-12s %-6s %-5s %-6s %s" % ("file","size","SOF","DRI","nSOI","DHT","apps"))
for r in rows:
    sof = r[2]
    print("%-16s %-10d %-12s %-6s %-5d %-6d %s" % (r[0], r[1], "%dx%d"%(sof[2],sof[3]), r[5], r[7], r[4], r[6][:90]))
