"""把一个散图目录切成 512 patch，并产出官方格式的 parquet。

prep_lsdir.py 是从 parquet 读源图的；PD12M / ShopSign 这类源筛完是散文件，用这个。
切块方式与 prep_lsdir.py 完全一致（均匀铺满、残边不丢），保证两边产出可比。

    python csig/prep_patches.py <图片目录> <输出patch目录> <输出parquet> [--limit N]

--limit 按文件名排序取前 N 张，确定性，用于控制混合配比。
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

S = 512
WORKERS = int(os.environ.get("WORKERS", "48"))


def starts(size):
    """一维上的等距起点，正好盖住 [0, size)。与 prep_lsdir.py 逐字一致。"""
    n = max(1, -(-size // S))          # ceil(size / S)
    if n == 1:
        return [0]
    return [round(i * (size - S) / (n - 1)) for i in range(n)]


def cut(args):
    src, outdir = args
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    out = []
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if w < S or h < S:
            return out
        base = os.path.splitext(os.path.basename(src))[0]
        sub = os.path.join(outdir, base[:6])
        os.makedirs(sub, exist_ok=True)
        for j, top in enumerate(starts(h)):
            for i, left in enumerate(starts(w)):
                dst = os.path.join(sub, "%s_%d_%d.png" % (base, j, i))
                if not os.path.exists(dst):
                    im.crop((left, top, left + S, top + S)).save(dst, "PNG")
                out.append(dst)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("out_parquet")
    ap.add_argument("--limit", type=int, default=0, help="按文件名排序取前 N 张源图，用于控配比")
    a = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = sorted(f for f in os.listdir(a.src_dir) if f.lower().endswith(exts))
    if a.limit:
        files = files[:a.limit]
    files = [os.path.join(a.src_dir, f) for f in files]
    print("源图 %d 张 -> %s" % (len(files), a.out_dir), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)

    paths = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(cut, [(f, a.out_dir) for f in files], chunksize=8)):
            paths.extend(r)
            if (i + 1) % 500 == 0:
                print("  %d/%d 图, %d patch" % (i + 1, len(files), len(paths)), flush=True)

    import pandas as pd
    pd.DataFrame({"image_path": paths, "prompt": [""] * len(paths)}).to_parquet(a.out_parquet)
    print("完成: %d 图 -> %d patch (%.2f/图), parquet -> %s"
          % (len(files), len(paths), len(paths) / max(len(files), 1), a.out_parquet), flush=True)


if __name__ == "__main__":
    main()
