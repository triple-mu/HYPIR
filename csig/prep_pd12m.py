"""PD12M 手机 ISP 域内层：元数据筛选 -> 下载 -> 三闸门锐度筛 -> file_list。

为什么要这一层（占配比 30%）：4KLSDB 虽然 100% >=3840px，但它 49.6% 是人像、
中文/亚洲线索仅 1.7%，且最锐 patch 的高频能量比真手机 ISP 高 6.57 倍 —— 拿它训出来的模型
会幻想出手机根本不会产生的精细纹理，而 TOPIQ-FR 权重是 MANIQA 的 3 倍，这些纹理是净扣分。
PD12M 是唯一同时满足「手机 ISP(~90% 真 EXIF) + 12MP 4:3 + 3.11 bpp + 可商用(CDLA-Permissive-2.0)」的源。

用法:
    python prep_pd12m.py index                  # 只读元数据出候选 URL
    python prep_pd12m.py fetch [--limit N]      # 下载 + 筛选
"""
import argparse, glob, io, os, re, sys
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csig_data import hf_max, ijg_quality

CSIG = "/root/.cache/huggingface/csig"
META = os.path.join(CSIG, "data/pd12m/metadata")
OUT = os.path.join(CSIG, "data/pd12m_hr")
URLS = os.path.join(CSIG, "data/pd12m_urls.txt")

MIN_LONG = 3400
AR_LO, AR_HI = 1.28, 1.40          # 12MP 手机出图的 4:3（含少量 3:2 边缘）
DROP = re.compile(r"\b(portrait|man|woman|person|people|girl|boy|face|painting|drawing|"
                  r"illustration|engraving|sketch|manuscript|coin|stamp|map|document|"
                  r"museum|artwork|sculpture of a)\b", re.I)
KEEP = re.compile(r"\b(building|architecture|facade|skyscraper|tower|city|urban|street|"
                  r"skyline|window|balcony|roof|bridge|house|plant|flower|leaf|leaves|tree|"
                  r"foliage|garden|branch|petal|blossom|grass|park|forest|sign|shop|store|"
                  r"boat|ship|vehicle|bicycle|mountain|landscape)\b", re.I)


def do_index():
    fs = sorted(glob.glob(os.path.join(META, "*.parquet")))
    print("元数据分片 %d 个" % len(fs), flush=True)
    seen = kept = 0
    with open(URLS, "w") as fh:
        for i, f in enumerate(fs):
            t = pq.read_table(f, columns=["url", "caption", "width", "height", "mime_type"])
            for r in t.to_pylist():
                seen += 1
                if r["mime_type"] != "image/jpeg":
                    continue
                w, h = r["width"] or 0, r["height"] or 0
                if max(w, h) < MIN_LONG or min(w, h) < 512:
                    continue
                ar = max(w, h) / max(min(w, h), 1)
                if not (AR_LO <= ar <= AR_HI):
                    continue
                cap = r["caption"] or ""
                if DROP.search(cap) or not KEEP.search(cap):
                    continue
                fh.write(r["url"] + "\n"); kept += 1
            if (i + 1) % 25 == 0:
                print("  %d/%d 片, 扫 %d 存 %d" % (i + 1, len(fs), seen, kept), flush=True)
    print("合计: 扫 %d 行, 候选 %d 条 (%.2f%%) -> %s" % (seen, kept, 100.0 * kept / seen, URLS), flush=True)


def one(args):
    idx, url = args
    import urllib.request
    from PIL import Image
    dst = os.path.join(OUT, "pd_%07d.jpg" % idx)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        return dst
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "csig-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            b = r.read()
        im = Image.open(io.BytesIO(b))
        if min(im.size) < 512 or max(im.size) < 3000:
            return None
        if hf_max(np.asarray(im.convert("L"))) < 0.005:      # 闸门1: 谱截止
            return None
        q = getattr(im, "quantization", None)                # 闸门2: 量化表(仅 IJG 族)
        if q:
            iq = ijg_quality(q[0])
            if iq is not None and iq < 93:
                return None
        if len(b) * 8 / (im.width * im.height) < 2.0:        # 闸门3: bpp
            return None
        with open(dst, "wb") as f:
            f.write(b)
        return dst
    except Exception:
        return None


def do_fetch(limit, workers):
    os.makedirs(OUT, exist_ok=True)
    urls = [l.strip() for l in open(URLS) if l.strip()]
    if limit:
        urls = urls[:limit]
    print("候选 %d 条, %d 线程" % (len(urls), workers), flush=True)
    kept = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, enumerate(urls))):
            if r:
                kept.append(r)
            if (i + 1) % 500 == 0:
                print("  %d/%d 尝试, 通过 %d (%.1f%%)" %
                      (i + 1, len(urls), len(kept), 100.0 * len(kept) / (i + 1)), flush=True)
    with open(os.path.join(CSIG, "data/pd12m_list.txt"), "w") as f:
        f.write("\n".join(kept) + "\n")
    print("完成: 尝试 %d, 通过三闸门 %d (%.1f%%)" %
          (len(urls), len(kept), 100.0 * len(kept) / max(len(urls), 1)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["index", "fetch"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    do_index() if a.mode == "index" else do_fetch(a.limit, a.workers)
