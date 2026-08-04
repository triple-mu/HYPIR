"""把 LSDIR 切成 512x512 patch，出官方 README 那个格式的 parquet。

官方 up/main:README.md 的 Train 一节给的复现配方就是这个：把 LSDIR 切成 512x512 patch
放进一个目录，再写一个两列（image_path / prompt）的 parquet 喂给 RealESRGANDataset。
这一步不能省 —— realesrgan.py:117 在 crop_type: none 下会 assert 图片恰好是
out_size x out_size，喂原图直接崩。

源用 HF 的 danjacobellis/LSDIR（未 gated，195 个 parquet，84,991 张 train，97.8 GB）。
切法是非重叠 512（step 512，残边丢弃，任一边不足 512 的图跳过），不做任何内容筛选 ——
官方没有筛，我们也不筛。

    python csig/prep_lsdir.py <lsdir_raw目录> <patch输出目录> <parquet输出路径>
"""
import io
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt import PROMPT

S = 512
WORKERS = int(os.environ.get("WORKERS", "48"))


def do_shard(src, out_root):
    kept = []
    pf = pq.ParquetFile(src)
    # 逐 batch 读：单个分片 ~500 MB，整片 to_pylist 会让 48 个进程一起吃掉几十 GB
    for batch in pf.iter_batches(batch_size=8, columns=["path", "image"]):
        for row in batch.to_pylist():
            stem = os.path.splitext(os.path.basename(row["path"]))[0]
            try:
                im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if w < S or h < S:
                continue
            # 分桶存：一个目录塞 30 万个文件虽然能用，但任何 ls / glob 都会变得很难受
            sub = os.path.join(out_root, stem[:4])
            os.makedirs(sub, exist_ok=True)
            for j, top in enumerate(range(0, h - S + 1, S)):
                for i, left in enumerate(range(0, w - S + 1, S)):
                    dst = os.path.join(sub, "%s_%d_%d.png" % (stem, j, i))
                    im.crop((left, top, left + S, top + S)).save(dst, "PNG")
                    kept.append(dst)
    return os.path.basename(src), len(kept), kept


def main():
    raw_dir, out_root, out_parquet = sys.argv[1], sys.argv[2], sys.argv[3]
    shards = sorted(glob.glob(os.path.join(raw_dir, "data", "train-*.parquet")))
    assert shards, "没找到 train-*.parquet，检查 %s" % raw_dir
    os.makedirs(out_root, exist_ok=True)
    print("分片 %d 个，worker %d" % (len(shards), WORKERS), flush=True)

    paths, done = [], 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for name, n, ps in ex.map(do_shard, shards, [out_root] * len(shards)):
            paths.extend(ps)
            done += 1
            print("  [%3d/%3d] %s -> %d patch (累计 %d)" % (done, len(shards), name, n, len(paths)),
                  flush=True)

    import polars as pl
    paths.sort()
    pl.from_dict({"image_path": paths, "prompt": [PROMPT] * len(paths)}).write_parquet(out_parquet)
    print("共 %d 个 patch，file_list 写到 %s" % (len(paths), out_parquet))


if __name__ == "__main__":
    main()
