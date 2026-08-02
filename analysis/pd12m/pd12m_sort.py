"""PD12M 按 EXIF 机型/地域分拣。**先 Range 抓 96KB 头，只把命中的整图下下来。**

为什么不直接下整图：候选平均 5.34 MB/张（实测 390 张 2.08 GB），
1.48M 候选全下 = 3.2 TB；而 EXIF(APP1) + DQT 都在文件头 96KB 内，
Range 抓头只要 142 GB，且**不落盘**。实测 64 线程 169 req/s，全量扫完约 2.4 小时。

三个子命令：
  scan   parquet -> 候选 URL -> Range 96KB -> {make/model/GPS国家/bpp} -> scan.csv
         闸门 3（bpp>=2）在这步就判完（Content-Range 白送总字节数）；
         闸门 2 的 DQT 有一半躺在 96KB 之外（华为/OPPO 的 APP 段能到 400KB），挪到 fetch 阶段
  fetch  scan.csv 按策略挑行 -> 下整图 -> 闸门 1（hf_max>=0.005）+ 闸门 2 -> keep.txt
  sort   对**已经下好**的目录直接按 EXIF 分拣（不联网），出 sorted.csv

用法：
  python pd12m_sort.py scan  <parquet_dir> <out.csv> [--threads 64] [--res-only]
  python pd12m_sort.py fetch <scan.csv> <img_dir> <keep.txt> [--policy cn|cn_geo|phone]
  python pd12m_sort.py sort  <img_dir> <out.csv>

实测产出率（n=11,181 分层抽样 + 431 张整图验证，2026-08）：
  几何候选 1,478,983
    中国品牌手机 3.80%  56,186  -> bpp>=2 45,863 -> 过闸门1&2(77.9%) **35,713**
    中国品牌+中国地域证据 0.77%  11,402 -> bpp>=2  9,539 -> 过闸门1&2 **7,428**
"""

import argparse, csv, glob, io, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image, ExifTags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csig_data import hf_max, ijg_quality

Image.MAX_IMAGE_PIXELS = None
HEAD = 98304          # 96KB：实测 64KB 已能解出 96.9% 的 Model，96KB 留余量
GEOJSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ne_50m_countries.geojson")
GEOJSON_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_50m_admin_0_countries.geojson")
GREATER_CN = {"China", "Hong Kong", "Taiwan", "Macao S.A.R"}

# 高产分辨率白名单：占候选 7.43%，含 62.3% 的中国品牌手机（精度 31.8%，相对基线 8.4x 富集）
# 4160x3120 / 3968x2976 / 4096x3072 三档精度 71-74%，几乎是中国机专属；4000x3000 只有 19% 但量大
RES_WHITELIST = {(4000, 3000), (3000, 4000), (4160, 3120), (3120, 4160),
                 (3968, 2976), (2976, 3968), (4096, 3072), (3072, 4096)}

# 中国品牌：Make 字段命中，或 Make 空/未知时 Model 命中厂商内部代号
CN_MAKE = re.compile(r"^(huawei|honor|xiaomi|redmi|poco|oppo|vivo|oneplus|realme|meizu|zte|nubia|"
                     r"lenovo|coolpad|gionee|smartisan|hisense|tcl|doogee|umidigi|blackview|cubot|"
                     r"ulefone|letv|leeco|infinix|tecno|itel|hmd)", re.I)
CN_MODEL = re.compile(r"^(cph\d|pd[a-z0-9]{4}|v\d{4}[a-z]|rmx\d|mi \d|mi note|mi max|mi a\d|redmi|poco|"
                      r"m2\d{3}[a-z0-9]|2[0-9]{5}[a-z0-9]+|[a-z]{3}-[atwlnc][lx]\d{2}|"
                      r"nth-|els-|vog-|lya-|ana-|ela-|tas-|noh-|jad-|alp-|clt-|eml-|war-|jsn-|pot-|"
                      r"mar-|lld-|dub-|jny-|brq-|nam-|kb2\d{3}|le2\d{3}|in2\d{3}|hd1\d{3}|gm1\d{3})", re.I)
OTHER_PHONE_MAKE = re.compile(r"^(apple|samsung|google|motorola|lg electronics|lge|nokia|htc|asus|"
                              r"sony ericsson|essential|fairphone|sharp|kyocera|blackberry)", re.I)
CAMERA_MAKE = re.compile(r"^(nikon|canon|sony|fujifilm|olympus|panasonic|pentax|leica|hasselblad|"
                         r"phase one|leaf|casio|kodak|epson|ricoh|sigma)", re.I)
# caption 里的中国线索（PD12M 的 caption 是机器生成，只有这些词能提供地域信号）
CN_CAPTION = re.compile(r"\b(china|chinese|beijing|shanghai|hangzhou|shenzhen|guangzhou|wenzhou|"
                        r"nanjing|suzhou|chengdu|hong kong|taiwan|taipei|macau|pagoda)\b", re.I)


def brand_class(make: str, model: str) -> str:
    """返回 cn / phone / camera / unknown。"""
    mk, md = (make or "").strip(), (model or "").strip()
    if CN_MAKE.match(mk):
        return "cn"
    if CAMERA_MAKE.match(mk):
        return "camera"
    if OTHER_PHONE_MAKE.match(mk):
        return "phone"
    if CN_MODEL.match(md):
        return "cn"
    return "unknown" if not md else "camera"


_POLY = None


def _polys():
    """Natural Earth 50m 国界，第一次用时自动下。bbox 预筛 + matplotlib 点在多边形内。"""
    global _POLY
    if _POLY is None:
        if not os.path.exists(GEOJSON):
            open(GEOJSON, "wb").write(requests.get(GEOJSON_URL, timeout=300).content)
        from matplotlib.path import Path
        _POLY = []
        for f in json.load(open(GEOJSON))["features"]:
            gm = f["geometry"]
            for p in (gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]):
                r = np.asarray(p[0], dtype=float)
                _POLY.append((f["properties"]["NAME"], r[:, 0].min(), r[:, 0].max(),
                              r[:, 1].min(), r[:, 1].max(), Path(r)))
    return _POLY


def country_of(lat, lon):
    for nm, x0, x1, y0, y1, pa in _polys():
        if x0 <= lon <= x1 and y0 <= lat <= y1 and pa.contains_point((lon, lat)):
            return nm
    return ""


def _txt(v) -> str:
    if isinstance(v, bytes):
        v = v.split(b"\x00")[0].decode("utf-8", "replace")
    return "".join(c if c.isprintable() else " " for c in str(v)).strip()[:120]


def candidates(parquet_dir: str, res_only: bool) -> pd.DataFrame:
    """几何闸门：jpeg + 长边>=3400 + 宽高比 1.28-1.40（手机 4:3 / 3:2 的并集）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["url", "caption", "width", "height", "mime_type"]).to_pandas()
        L, S = t[["width", "height"]].max(1), t[["width", "height"]].min(1)
        m = (t.mime_type == "image/jpeg") & (L >= 3400) & (L / S >= 1.28) & (L / S <= 1.40)
        if res_only:
            m &= pd.Series([(w, h) in RES_WHITELIST for w, h in zip(t.width, t.height)], index=t.index)
        out.append(t[m].drop(columns="mime_type"))
    return pd.concat(out, ignore_index=True)


_S = requests.Session()
_S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=2))


def app1_exif(b: bytes):
    """手工走 JPEG 段表取 Exif APP1。**不能用 Image.open**：华为/OPPO 的 APP 段常达 400KB，
    截断的头解不出 SOF，PIL 直接抛 Truncated File Read —— 实测这样会漏掉 8% 候选，
    而且漏的恰好是 Mate 50 Pro / OPPO A77 / vivo V2111 这类我们最想要的机型。"""
    i = 2
    while i + 4 <= len(b) and b[i] == 0xFF and b[i + 1] != 0xDA:
        n = int.from_bytes(b[i + 2:i + 4], "big")
        seg = b[i + 4:i + 2 + n]
        if b[i + 1] == 0xE1 and seg.startswith(b"Exif"):
            e = Image.Exif()
            e.load(seg)
            return e
        i += 2 + n
    return None


def probe_head(row):
    """Range 抓 96KB -> EXIF + 总字节（闸门 3）。闸门 2 的 DQT 有一半落在 96KB 之外，挪到 fetch 阶段判。"""
    url, w, h, cap = row
    r = {"url": url, "w": w, "h": h, "cn_caption": int(bool(CN_CAPTION.search(cap or "")))}
    try:
        rp = _S.get(url, headers={"Range": f"bytes=0-{HEAD-1}"}, timeout=45)
        total = int(rp.headers.get("Content-Range", "/0").split("/")[-1]) or len(rp.content)
        r["bpp"] = round(total * 8 / (w * h), 3)
        ex = app1_exif(rp.content)
        if ex is None:
            r["brand"] = "unknown"
            return r
        d = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        r["make"], r["model"] = _txt(d.get("Make", "")), _txt(d.get("Model", ""))
        r["dt"] = _txt(d.get("DateTime", ""))
        g = ex.get_ifd(0x8825)
        if g and 2 in g and 4 in g:
            la, lo = [float(x) for x in g[2]], [float(x) for x in g[4]]
            lat = la[0] + la[1] / 60 + la[2] / 3600
            lon = lo[0] + lo[1] / 60 + lo[2] / 3600
            if _txt(g.get(1, "N")).upper().startswith("S"):
                lat = -lat
            if _txt(g.get(3, "E")).upper().startswith("W"):
                lon = -lon
            if lat or lon:
                r["lat"], r["lon"] = round(lat, 5), round(lon, 5)
                r["country"] = country_of(lat, lon)
        r["brand"] = brand_class(r["make"], r["model"])
    except Exception as e:
        r["err"] = type(e).__name__
    return r


COLS = ["url", "w", "h", "bpp", "make", "model", "brand", "dt", "lat", "lon",
        "country", "cn_caption", "err"]


def cmd_scan(a):
    df = candidates(a.parquet_dir, a.res_only)
    done = set()
    if os.path.exists(a.out):                      # 断点续扫
        done = set(pd.read_csv(a.out, usecols=["url"]).url)
        df = df[~df.url.isin(done)]
    print(f"候选 {len(df)+len(done)}，待扫 {len(df)}", flush=True)
    _polys()
    rows = list(zip(df.url, df.width, df.height, df.caption))
    t0, n = time.time(), 0
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        if not done:
            w.writeheader()
        with ThreadPoolExecutor(a.threads) as ex:
            for r in ex.map(probe_head, rows):
                w.writerow(r)
                n += 1
                if n % 5000 == 0:
                    f.flush()
                    print(f"{n}/{len(rows)}  {n/(time.time()-t0):.0f} req/s", flush=True)
    print("scan done", n, f"{time.time()-t0:.0f}s")


POLICY = {
    # 只要中国品牌手机；最大召回，实测占候选 3.49%
    "cn":     lambda d: d.brand.eq("cn"),
    # 中国品牌手机 + 有中国地域证据（GPS 落大中华区 或 caption 有中国词）；实测 0.76%
    "cn_geo": lambda d: d.brand.eq("cn") & (d.country.isin(GREATER_CN) | d.cn_caption.eq(1)),
    # 任意手机 + 中国地域证据（想把 iPhone 在华拍的也收进来时用）；实测 0.90%
    "phone":  lambda d: d.brand.isin(["cn", "phone"]) & (d.country.isin(GREATER_CN) | d.cn_caption.eq(1)),
}


def cmd_fetch(a):
    d = pd.read_csv(a.scan, dtype={"country": str}).fillna({"country": "", "brand": "unknown", "cn_caption": 0})
    m = POLICY[a.policy](d) & (d.bpp >= 2.0)            # 闸门 3 在 scan 阶段已算好
    d = d[m].reset_index(drop=True)
    print(f"策略 {a.policy} + 闸门3 命中 {len(d)} 张，预计下载 {len(d)*5.34/1024:.1f} GB", flush=True)
    os.makedirs(a.img_dir, exist_ok=True)

    def get(t):
        i, url = t
        p = os.path.join(a.img_dir, f"{i:07d}.jpg")
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            try:
                open(p, "wb").write(_S.get(url, timeout=300).content)
            except Exception:
                return p, np.nan
        try:
            im = Image.open(p)
            q = getattr(im, "quantization", None)
            if q:                                       # 闸门 2：厂商自研表 -> None -> 放行
                iq = ijg_quality(q[0])
                if iq is not None and iq < 93:
                    return p, np.nan
            return p, hf_max(np.asarray(im.convert("L")))   # 闸门 1：原生锐度
        except Exception:
            return p, np.nan

    keep = 0
    with open(a.keep, "w") as f, ThreadPoolExecutor(a.threads) as ex:
        for p, hf in ex.map(get, enumerate(d.url)):
            if hf >= 0.005:
                f.write(p + "\n")
                keep += 1
            elif os.path.exists(p):
                os.remove(p)                            # 没过闸门的直接删，别占盘
    print(f"过三闸门 {keep}/{len(d)} = {keep/max(len(d),1)*100:.1f}%")


def cmd_sort(a):
    """对已下好的图直接分拣：EXIF 机型 + GPS 国家 + 三闸门，全部本地算。"""
    _polys()
    paths = sorted(glob.glob(os.path.join(a.img_dir, "**", "*.jp*g"), recursive=True))
    print(f"{len(paths)} 张", flush=True)

    def one(p):
        r = {"path": p}
        try:
            im = Image.open(p)
            r["w"], r["h"] = im.size
            r["bpp"] = round(os.path.getsize(p) * 8 / (im.width * im.height), 3)
            ex = im.getexif()
            d = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
            r["make"], r["model"] = _txt(d.get("Make", "")), _txt(d.get("Model", ""))
            r["dt"] = _txt(d.get("DateTime", ""))
            r["brand"] = brand_class(r["make"], r["model"])
            g = ex.get_ifd(0x8825)
            if g and 2 in g and 4 in g:
                la, lo = [float(x) for x in g[2]], [float(x) for x in g[4]]
                lat = la[0] + la[1] / 60 + la[2] / 3600
                lon = lo[0] + lo[1] / 60 + lo[2] / 3600
                if _txt(g.get(1, "N")).upper().startswith("S"):
                    lat = -lat
                if _txt(g.get(3, "E")).upper().startswith("W"):
                    lon = -lon
                if lat or lon:
                    r["lat"], r["lon"], r["country"] = round(lat, 5), round(lon, 5), country_of(lat, lon)
            q = getattr(im, "quantization", None)
            r["ijgq"] = ijg_quality(q[0]) if q else None
            r["hfmax"] = round(hf_max(np.asarray(im.convert("L"))), 6)
            r["pass3"] = int(r["hfmax"] >= 0.005 and r["bpp"] >= 2.0
                             and (r["ijgq"] is None or r["ijgq"] >= 93))
        except Exception as e:
            r["err"] = type(e).__name__
        return r

    with ThreadPoolExecutor(a.threads) as ex:
        rows = pd.DataFrame(list(ex.map(one, paths)))
    rows.to_csv(a.out, index=False)
    rows = rows.fillna({"brand": "unknown", "country": "", "pass3": 0})
    print(rows.brand.value_counts().to_string())
    print("\n中国品牌 %d，其中过三闸门 %d" % (rows.brand.eq("cn").sum(),
                                        int(rows.pass3[rows.brand.eq("cn")].sum())))
    print(rows[rows.brand.eq("cn")].model.value_counts().head(20).to_string())
    print("\nGPS 国家 top10:", rows[rows.country.ne("")].country.value_counts().head(10).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan");  s.add_argument("parquet_dir"); s.add_argument("out")
    s.add_argument("--threads", type=int, default=64)
    s.add_argument("--res-only", action="store_true", help="只扫高产分辨率白名单（省 93% 带宽，召回 57%%）")
    s.set_defaults(func=cmd_scan)
    f = sub.add_parser("fetch"); f.add_argument("scan"); f.add_argument("img_dir"); f.add_argument("keep")
    f.add_argument("--policy", default="cn_geo", choices=list(POLICY))
    f.add_argument("--threads", type=int, default=24)
    f.set_defaults(func=cmd_fetch)
    o = sub.add_parser("sort"); o.add_argument("img_dir"); o.add_argument("out")
    o.add_argument("--threads", type=int, default=16)
    o.set_defaults(func=cmd_sort)
    a = ap.parse_args()
    a.func(a)
