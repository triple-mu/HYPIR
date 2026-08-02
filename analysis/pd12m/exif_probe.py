"""HTTP Range 只取 JPEG 头 128KB -> 解 EXIF(Make/Model/GPS) + DQT(IJG quality) + 总字节数(bpp)。
比整图下载省 30-50 倍带宽，闸门 2/3 也能在这一步判掉。"""
import io, sys, csv, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ExifTags
sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
from csig_data import ijg_quality
Image.MAX_IMAGE_PIXELS = None
HEAD = 131072
S = requests.Session()
a = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=2)
S.mount("https://", a)

def dec(v):
    if isinstance(v, bytes):
        v = v.split(b"\x00")[0].decode("utf-8", "replace")
    return "".join(c if c.isprintable() else " " for c in str(v)).strip()[:120]

def rat(x):
    try: return float(x)
    except Exception: return None

def one(row):
    i, url, w, h = row
    r = {"i": i, "url": url, "w": w, "h": h}
    try:
        rp = S.get(url, headers={"Range": f"bytes=0-{HEAD-1}"}, timeout=45)
        buf = rp.content
        cr = rp.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1]) if "/" in cr else len(buf)
        r["bytes"] = total
        r["bpp"] = round(total * 8 / (w * h), 3)
        im = Image.open(io.BytesIO(buf + b"\xff\xd9"))
        ex = im.getexif()
        d = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        r["make"] = dec(d.get("Make", "")); r["model"] = dec(d.get("Model", ""))
        r["software"] = dec(d.get("Software", "")); r["dt"] = dec(d.get("DateTime", ""))
        try:
            ei = ex.get_ifd(0x8769)
            r["lens"] = dec(ei.get(42036, "")); r["fnum"] = rat(ei.get(33437))
            r["flen"] = rat(ei.get(37386)); r["iso"] = ei.get(34855)
        except Exception: pass
        try:
            g = ex.get_ifd(0x8825)
            if g and 2 in g and 4 in g:
                la = [float(x) for x in g[2]]; lo = [float(x) for x in g[4]]
                lat = la[0] + la[1] / 60 + la[2] / 3600
                lon = lo[0] + lo[1] / 60 + lo[2] / 3600
                if dec(g.get(1, "N")).upper().startswith("S"): lat = -lat
                if dec(g.get(3, "E")).upper().startswith("W"): lon = -lon
                r["lat"] = round(lat, 5); r["lon"] = round(lon, 5)
        except Exception: pass
        q = getattr(im, "quantization", None)
        if q:
            r["ijgq"] = ijg_quality(q[0]); r["qsum"] = int(sum(list(q[0])[:64]))
    except Exception as e:
        r["err"] = type(e).__name__
    return r

if __name__ == "__main__":
    src, dst, nth = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 64
    df = pd.read_csv(src)
    rows = list(zip(df.index, df.url, df.width, df.height))
    cols = ["i","url","w","h","bytes","bpp","make","model","software","dt","lens","fnum","flen","iso","lat","lon","ijgq","qsum","err"]
    t0 = time.time(); n = 0
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_ALL); wr.writeheader()
        with ThreadPoolExecutor(nth) as ex:
            for r in ex.map(one, rows):
                wr.writerow({k:(v.replace(chr(0)," ") if isinstance(v,str) else v) for k,v in r.items()}); n += 1
                if n % 500 == 0:
                    f.flush(); print(f"{n}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    print("done", n, time.time() - t0)
