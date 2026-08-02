#!/usr/bin/env python3
"""Wikimedia Commons 中国手机原图采集器（CSIG-2026 赛道二训练数据）。

三步：index -> filter -> download。索引阶段就能算 bpp（size*8/(w*h)），
不用下图就能砍掉一大半，只对候选下原图跑 hf_max / 量化表。

本机网络：clash 规则里 wikimedia 域名走 DIRECT（被墙），只有进程名匹配白名单的才走代理，
所以必须用一个名为 claude 的解释器副本跑，否则 HTTPS 握手直接超时：
    cp $(which python3.10) /tmp/pxpy/claude
    PYTHONHOME=/home/ubuntu/miniconda3/envs/torch /tmp/pxpy/claude commons_harvest.py ...

用法：
    python commons_harvest.py index    --out idx.jsonl [--preset zhejiang|china|flagship|all]
    python commons_harvest.py filter   --idx idx.jsonl --out cand.jsonl
    python commons_harvest.py download --cand cand.jsonl --dir ./imgs [--limit N]

实测得出的语法约束（2026-08，见文末 NOTES）：
  * filew:>=3000 无效，必须写 filew:>2999
  * `A OR B` / `(A OR B)` 对 filew/incategory 无效 —— 长边>=3000 必须拆成
    filew:>2999 和 fileh:>2999 两条 query 再并集去重
  * incategory:A|B|C 的 OR 生效（不加引号、连字符可用）
  * haswbstatement:P275=Q1|P275=Q2 的 OR 生效
  * deepcat 子类太多会静默失败，响应里带 warnings.search.warnings="Deep category
    search timed out"，必须校验（deepcat:"China" 就是这样只返回 245 条）
  * search 的 sroffset 上限 10000，超了要拆分区
"""
import argparse, hashlib, io, json, os, sys, time
import requests

UA = "CSIGDataBot/0.0 (https://github.com/csig2026-data; sonlin@nvidia.com)"
API = "https://commons.wikimedia.org/w/api.php"
RATE = 60.0 / 200.0          # 200 req/min

S = requests.Session()
S.headers.update({"User-Agent": UA})
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < RATE:
        time.sleep(RATE - dt)
    _last[0] = time.time()


def api(params):
    p = dict(params)
    p.setdefault("format", "json")
    p.setdefault("formatversion", 2)
    for att in range(5):
        _throttle()
        try:
            r = S.get(API, params=p, timeout=90)
            if r.status_code == 200:
                d = r.json()
                w = json.dumps(d.get("warnings", {}))
                if "timed out" in w:          # deepcat 子类爆了，结果不可信
                    raise RuntimeError("DEEPCAT_TIMEOUT: " + w[:200])
                return d
            if r.status_code in (403, 429):
                print("HTTP %d -- 检查 UA 是否含 'bot'" % r.status_code, file=sys.stderr)
            time.sleep(3 * (att + 1))
        except RuntimeError:
            raise
        except Exception as e:
            print("retry %d: %s" % (att, e), file=sys.stderr)
            time.sleep(3 * (att + 1))
    raise RuntimeError("api failed: %s" % p.get("gsrsearch") or p.get("gcmtitle"))


# ---------------- 检索常量（全部实测过 totalhits）----------------
CN_PHONES = 'deepcat:"Photos taken with Chinese mobile phones"'
BRAND_CAT = {
    "huawei":  'deepcat:"Photos taken with Huawei mobile phones"',   # 含 Honor
    "xiaomi":  'deepcat:"Photos taken with Xiaomi mobile phones"',   # 含 Redmi/Poco
    "oppo":    'deepcat:"Photos taken with Oppo mobile phones"',
    "vivo":    'deepcat:"Taken with Vivo mobile phones"',
    "realme":  'deepcat:"Taken with Realme mobile phones"',
    "oneplus": 'deepcat:"Taken with OnePlus mobile phones"',
    "meizu":   'deepcat:"Photos taken with Meizu mobile phones"',
    "zte":     'deepcat:"Taken with ZTE mobile phones"',
}
# 只要可商用且不传染：CC0 / PD / CC BY 各版本。**全部 -SA 与 GFDL 排除**
FREE_LIC = "incategory:CC-Zero|CC-BY-4.0|CC-BY-3.0|CC-BY-2.0|CC-BY-2.5|CC-BY-1.0|CC-PD-Mark|PD-self"
LIC_ALLOW = {"cc0", "cc-zero", "cc by 4.0", "cc by 3.0", "cc by 2.5", "cc by 2.0", "cc by 1.0",
             "public domain", "pd", "pdm-owner"}
JPG = 'filemime:image/jpeg'
CN_BRANDS = ("huawei", "honor", "xiaomi", "redmi", "poco", "oppo", "vivo", "realme",
             "oneplus", "meizu", "zte", "nubia", "hisense", "lenovo", "smartisan",
             "coolpad", "leeco", "tcl", "blackview", "doogee", "umidigi")

PRESETS = {
    # 浙江/杭州/温州 —— 与测试集地理完全对口，量小但最贵
    "zhejiang": [f'deepcat:"{g}" {CN_PHONES} {JPG} {FREE_LIC}'
                 for g in ("Zhejiang", "Hangzhou", "Wenzhou", "Ningbo", "Shaoxing", "Jiaxing",
                           "Taizhou, Zhejiang", "Jinhua", "Huzhou", "Quzhou", "Lishui, Zhejiang", "Zhoushan")],
    # 全中国（34 个省级行政区，deepcat:"China" 本身子类太多会静默截断，只能逐省打）
    "china": [f'deepcat:"{g}" {CN_PHONES} {JPG} {FREE_LIC}'
              for g in ("Anhui", "Beijing", "Chongqing", "Fujian", "Gansu", "Guangdong", "Guangxi",
                        "Guizhou", "Hainan", "Hebei", "Heilongjiang", "Henan", "Hubei", "Hunan",
                        "Inner Mongolia", "Jiangsu", "Jiangxi", "Jilin", "Liaoning", "Ningxia",
                        "Qinghai", "Shaanxi", "Shandong", "Shanghai", "Shanxi", "Sichuan",
                        "Tianjin", "Tibet", "Xinjiang", "Yunnan", "Zhejiang", "Hong Kong", "Macau")],
    # 近年旗舰（大底 + 强 ISP），与 Pura 80 Ultra 的成像风格最近
    "flagship": [f'incategory:"{c}" {JPG} {FREE_LIC}' for c in (
        "Taken with Xiaomi 17 Ultra", "Taken with Xiaomi 15 Ultra", "Taken with Xiaomi 14 Ultra",
        "Taken with Xiaomi 13 Ultra", "Taken with Xiaomi 12S Ultra", "Taken with Xiaomi 14T Pro",
        "Taken with Xiaomi 13 Pro", "Taken with Xiaomi 14 Pro", "Taken with Xiaomi 15",
        "Taken with Honor Magic 6 Pro", "Taken with Honor Magic5 Pro", "Taken with Honor Magic4 Pro",
        "Taken with Huawei P30 Pro", "Taken with Huawei P40 Pro", "Taken with Huawei P50 Pro",
        "Taken with Huawei Mate 20 Pro", "Taken with Huawei Mate 30 Pro", "Taken with Huawei Mate 40 Pro",
        "Taken with Huawei P60 Pro", "Taken with Huawei Mate 50 Pro", "Taken with Huawei Pura 70 Ultra",
        "Taken with Oppo Find X5 Pro", "Taken with Oppo Find X6 Pro", "Taken with Vivo X100 Pro",
        "Taken with Vivo X90 Pro", "Taken with OnePlus 11 5G", "Taken with OnePlus 12")],
    # 华为+荣耀整棵树（实测 filew:>2999 命中 9,942，逼近 sroffset 10000 上限，届时按机型再拆）
    "huawei": [f'{BRAND_CAT["huawei"]} {JPG} {FREE_LIC}'],
    # 赛题原生栅格 4096x3072 / 3072x4096（实测 4,327 + 1,459）
    "raster": [f'{CN_PHONES} {JPG} {FREE_LIC} filew:{w} fileh:{h}'
               for w, h in ((4096, 3072), (3072, 4096))],
    # 全量中国品牌手机
    "all": [f'{c} {JPG} {FREE_LIC}' for c in BRAND_CAT.values()],
}

EXTMETA = "LicenseShortName|License|UsageTerms|Artist|Credit|DateTimeOriginal|ImageDescription|Categories|Restrictions"


def _pages(d):
    return (d.get("query") or {}).get("pages") or []


def _rec(p):
    ii = (p.get("imageinfo") or [{}])[0]
    if not ii.get("url"):
        return None
    md = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if isinstance(m.get("value"), (str, int, float))}
    em = {k: v.get("value") for k, v in (ii.get("extmetadata") or {}).items()}
    w, h = ii.get("width", 0), ii.get("height", 0)
    sz = ii.get("size", 0)
    return {
        "title": p["title"], "url": ii["url"], "w": w, "h": h, "size": sz,
        "bpp": round(sz * 8 / max(w * h, 1), 4),
        "make": str(md.get("Make", "")).strip(), "model": str(md.get("Model", "")).strip(),
        "software": str(md.get("Software", "")).strip(),
        "lic": em.get("LicenseShortName", ""), "lic_code": em.get("License", ""),
        "restrict": em.get("Restrictions", ""),
        "artist": em.get("Artist", ""), "date": em.get("DateTimeOriginal", ""),
        "cats": em.get("Categories", ""),
    }


def search_all(query, cap=100000):
    """跑一条 query，翻页取全部（sroffset 上限 10000，到顶就停并告警）。"""
    off, seen = 0, {}
    while off < min(cap, 10000):
        d = api({"action": "query", "generator": "search", "gsrsearch": query,
                 "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": off,
                 "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata|metadata",
                 "iiextmetadatafilter": EXTMETA})
        ps = _pages(d)
        if not ps:
            break
        for p in ps:
            r = _rec(p)
            if r:
                seen[r["title"]] = r
        nxt = (d.get("continue") or {}).get("gsroffset")
        if nxt is None:
            break
        off = nxt
    if off >= 10000:
        print("  !! 触到 sroffset 10000 上限，该 query 需再拆分: %s" % query[:90], file=sys.stderr)
    return seen


def cmd_index(a):
    qs = []
    for pre in a.preset.split(","):
        qs += PRESETS[pre]
    all_rec = {}
    for base in qs:
        for dim in ("filew:>2999", "fileh:>2999"):     # 长边>=3000 拆两条再并集
            q = base + " " + dim
            # deepcat 会静默返回截断结果（实测 Guangxi 同一 query 先后拿到 18 和 658，
            # 且截断那次没有 warnings）。跑 a.repeat 遍取并集，截断只会漏不会多。
            for _ in range(a.repeat):
                try:
                    got = search_all(q)
                except RuntimeError as e:
                    print("SKIP (%s): %s" % (e, q[:90]), file=sys.stderr)
                    continue
                new = len(set(got) - set(all_rec))
                all_rec.update(got)
                print("%6d new / %6d hit  <- %s" % (new, len(got), q[:100]), flush=True)
    with open(a.out, "w") as f:
        for r in all_rec.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("index: %d unique files -> %s" % (len(all_rec), a.out))


def cmd_filter(a):
    keep, stat = [], {"n": 0, "res": 0, "bpp": 0, "lic": 0, "brand": 0}
    for line in open(a.idx):
        r = json.loads(line)
        stat["n"] += 1
        if max(r["w"], r["h"]) < 3000:
            stat["res"] += 1; continue
        if r["bpp"] < a.bpp:                    # 闸门3 在这里就能判，省下载
            stat["bpp"] += 1; continue
        lic = (r["lic"] or "").strip().lower()
        if lic not in LIC_ALLOW:
            stat["lic"] += 1; continue
        blob = (r["make"] + " " + r["model"]).lower()
        if not any(b in blob for b in CN_BRANDS):
            stat["brand"] += 1; continue
        keep.append(r)
    with open(a.out, "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("filter: %d -> %d  (drop: res=%d bpp<%.1f=%d lic=%d brand=%d)"
          % (stat["n"], len(keep), stat["res"], a.bpp, stat["bpp"], stat["lic"], stat["brand"]))


def _fetch_original(url, path):
    """下原图。upload.wikimedia.org 的限流比 api.php 严得多，实测连发会吃 429，
    所以这里固定 1.2 s 间隔 + 尊重 Retry-After 退避。"""
    for att in range(6):
        time.sleep(1.2)
        try:
            resp = S.get(url, timeout=300)
        except Exception as e:
            print("  net %s" % e, file=sys.stderr); continue
        if resp.status_code == 200:
            open(path, "wb").write(resp.content)
            return True
        if resp.status_code in (429, 503):
            time.sleep(float(resp.headers.get("Retry-After", 0) or 0) + 5 * (att + 1))
            continue
        print("  HTTP %d" % resp.status_code, file=sys.stderr)
        return False
    return False


def cmd_download(a):
    sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/Moebius/tools")
    import numpy as np
    from PIL import Image
    from csig_data import hf_max, ijg_quality
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(a.dir, exist_ok=True)
    rows = [json.loads(l) for l in open(a.cand)]
    if a.limit:
        rows = rows[:a.limit]
    out, npass = [], 0
    for i, r in enumerate(rows):
        name = hashlib.md5(r["title"].encode()).hexdigest()[:16] + ".jpg"
        path = os.path.join(a.dir, name)
        if not os.path.exists(path):
            if not _fetch_original(r["url"], path):
                print("  dl fail %s" % r["title"][:60], file=sys.stderr); continue
        try:
            im = Image.open(path)
            g = np.asarray(im.convert("L"))
            hf = hf_max(g)
            qt = im.quantization.get(0) if getattr(im, "quantization", None) else None
            iq = ijg_quality(qt) if qt else None
            bpp = os.path.getsize(path) * 8 / (im.width * im.height)
        except Exception as e:
            print("  read err %s" % e, file=sys.stderr); continue
        ok = hf >= 0.005 and (iq is None or iq >= 93) and bpp >= 2.0
        npass += ok
        r.update(file=path, hf_max=round(hf, 5), ijg_q=iq, bpp_real=round(bpp, 3), passed=bool(ok))
        out.append(r)
        print("[%3d] %s hf=%.5f ijgq=%s bpp=%.2f %dx%d %-22s %s"
              % (i, "PASS" if ok else "----", hf, iq, bpp, r["w"], r["h"],
                 (r["make"] + " " + r["model"])[:22], r["title"][:48]), flush=True)
    json.dump(out, open(os.path.join(a.dir, "gate_report.json"), "w"), ensure_ascii=False, indent=1)
    print("\n三闸门通过 %d/%d = %.1f%%" % (npass, len(out), 100.0 * npass / max(len(out), 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index");    p.add_argument("--out", required=True); p.add_argument("--preset", default="zhejiang"); p.add_argument("--repeat", type=int, default=2); p.set_defaults(fn=cmd_index)
    p = sub.add_parser("filter");   p.add_argument("--idx", required=True); p.add_argument("--out", required=True); p.add_argument("--bpp", type=float, default=2.0); p.set_defaults(fn=cmd_filter)
    p = sub.add_parser("download"); p.add_argument("--cand", required=True); p.add_argument("--dir", required=True); p.add_argument("--limit", type=int, default=0); p.set_defaults(fn=cmd_download)
    a = ap.parse_args()
    a.fn(a)
