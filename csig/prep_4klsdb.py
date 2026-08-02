"""从 4KLSDB parquet 抽 HR 图 -> 内容过滤 -> 三闸门锐度筛选 -> 出 file_list。

4KLSDB 的内容分布与赛题严重错配：人像 49.6% vs 我们 ~0%，中文/亚洲线索仅 1.7%。
所以先按 caption 做内容过滤（丢人像、保建筑/植被/街景/招牌），再过锐度三闸门。
"""
import io, os, re, sys, glob
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# 丢：人像/人物主体。留：建筑/植被/城市/街景/招牌/器物
DROP = re.compile(r"\b(portrait|man|woman|person|people|girl|boy|face|model|couple|"
                  r"child|kid|bride|groom|hand|selfie|posing|wearing)\b", re.I)
KEEP = re.compile(r"\b(building|architecture|facade|skyscraper|tower|city|urban|street|"
                  r"skyline|window|balcony|roof|bridge|plant|flower|leaf|leaves|tree|"
                  r"foliage|garden|branch|petal|sign|signage|shop|store|statue|sculpture|"
                  r"boat|ship|vehicle|bicycle)\b", re.I)


def do_shard(path):
    from csig_data import hf_max, ijg_quality
    import numpy as np
    from PIL import Image
    kept, seen = [], 0
    t = pq.read_table(path, columns=["hr", "caption", "cogvlm_caption", "width", "height"])
    for row in t.to_pylist():
        seen += 1
        cap = (row.get("caption") or "") + " " + (row.get("cogvlm_caption") or "")
        if DROP.search(cap) or not KEEP.search(cap):
            continue
        b = row["hr"]["bytes"]
        if not b:
            continue
        try:
            im = Image.open(io.BytesIO(b))
            if min(im.size) < 512 or max(im.size) < 3000:
                continue
            # 三闸门
            if hf_max(np.asarray(im.convert("L"))) < 0.005:
                continue
            q = getattr(im, "quantization", None)
            if q:
                iq = ijg_quality(q[0])
                if iq is not None and iq < 93:
                    continue
            if len(b) * 8 / (im.width * im.height) < 2.0:
                continue
        except Exception:
            continue
        dst = os.path.join(OUT, "%s_%06d.jpg" % (os.path.basename(path)[9:14], len(kept)))
        with open(dst, "wb") as f:
            f.write(b)
        kept.append(dst)
    return os.path.basename(path), seen, kept


if __name__ == "__main__":
    EVAL_SHARDS = ("00118", "00119")  # 评估专用，绝不进训练
    shards = [f for f in sorted(glob.glob(os.path.join(SRC, "*.parquet")))
              if not any(e in f for e in EVAL_SHARDS)]
    print("分片 %d 个" % len(shards), flush=True)
    total_seen, allk = 0, []
    with ProcessPoolExecutor(10) as ex:
        for name, seen, kept in ex.map(do_shard, shards):
            total_seen += seen; allk += kept
            print("  %s: 扫 %d 存 %d" % (name, seen, len(kept)), flush=True)
    with open(os.path.join(os.path.dirname(OUT), "4klsdb_list.txt"), "w") as f:
        f.write("\n".join(allk) + "\n")
    print("合计: 扫 %d 张，通过内容+锐度筛选 %d 张 (%.1f%%)"
          % (total_seen, len(allk), 100.0 * len(allk) / max(total_seen, 1)), flush=True)
