"""按纹理密度在 **patch 级** 筛训练索引。

为什么必须在 patch 级筛（这是 mix 实验暴露的方法论错误）：
csig/screen_dir.py 的三闸门在**整图**上测 hf_max，PD12M 的 4K 原生图全部过闸，
但切成 512 patch 后实测 46% 落在平坦区（Laplacian 方差 <100，LSDIR 只有 1%），
中位纹理密度差 14 倍。训练用的是 patch，闸门却测整图 —— 粒度脱节。
结果是 mix@3000 的 MANIQA 掉 0.0528（t=-9.83），感知分 -0.0925（t=-3.05）。

为什么按纹理密度筛能涨分：MANIQA（权重 4.477）是无参考指标，奖励"输出锐利、
细节丰富"，与 GT 是否忠实无关。低纹理 GT 教会模型少动手，直接压 MANIQA。
旁证：ECCV 2024《Rethinking SR from Training Data》实测 1.2M 张筛到 259K 反而 +0.10 dB。

    # 第一步：算方差并缓存（只需跑一次，几分钟）
    python csig/filter_patches.py stats 索引.parquet 方差缓存.npz
    # 第二步：按分位数出新索引（秒级，可反复试不同阈值）
    python csig/filter_patches.py apply 索引.parquet 方差缓存.npz 出.parquet --drop-frac 0.30
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

WORKERS = int(os.environ.get("WORKERS", "48"))


def lap_var(path):
    """Laplacian 方差 —— 标准的纹理/清晰度度量，对平坦区敏感。"""
    import cv2
    from PIL import Image
    try:
        g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return -1.0


def do_stats(a):
    df = pd.read_parquet(a.parquet)
    paths = df["image_path"].tolist()
    print("算 %d 个 patch 的 Laplacian 方差, %d 进程" % (len(paths), WORKERS), flush=True)
    out = np.empty(len(paths), dtype=np.float64)
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, v in enumerate(ex.map(lap_var, paths, chunksize=256)):
            out[i] = v
            if (i + 1) % 50000 == 0:
                print("  %d/%d" % (i + 1, len(paths)), flush=True)
    np.savez_compressed(a.cache, var=out, n=len(paths))
    ok = out[out >= 0]
    print("完成。分位: p05=%.0f p25=%.0f p50=%.0f p75=%.0f p95=%.0f  读失败 %d"
          % (*np.percentile(ok, [5, 25, 50, 75, 95]), (out < 0).sum()), flush=True)


def do_apply(a):
    df = pd.read_parquet(a.parquet)
    var = np.load(a.cache)["var"]
    assert len(var) == len(df), "缓存与 parquet 行数不一致：%d vs %d" % (len(var), len(df))
    if a.thresh is not None:
        th = a.thresh
    else:
        th = float(np.percentile(var[var >= 0], 100 * a.drop_frac))
    keep = var >= th
    sub = df[keep].reset_index(drop=True)
    sub.to_parquet(a.out)
    src = sub["image_path"].str.rsplit("/", n=1).str[-1].str.rsplit("_", n=2).str[0]
    src0 = df["image_path"].str.rsplit("/", n=1).str[-1].str.rsplit("_", n=2).str[0]
    print("阈值 Laplacian方差 >= %.1f (剔除最低 %.0f%%)" % (th, 100 * a.drop_frac))
    print("  patch %d -> %d (%.1f%%)" % (len(df), len(sub), 100 * len(sub) / len(df)))
    print("  源图  %d -> %d (%.1f%%)" % (src0.nunique(), src.nunique(),
                                        100 * src.nunique() / src0.nunique()))
    print("  保留部分方差中位 %.0f（原 %.0f）" % (np.median(var[keep]), np.median(var[var >= 0])))
    print("  -> %s" % a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("parquet"); s.add_argument("cache")
    p = sub.add_parser("apply")
    p.add_argument("parquet"); p.add_argument("cache"); p.add_argument("out")
    p.add_argument("--drop-frac", type=float, default=0.30, help="剔除方差最低的这一比例")
    p.add_argument("--thresh", type=float, default=None, help="直接给绝对阈值，覆盖 --drop-frac")
    a = ap.parse_args()
    (do_stats if a.cmd == "stats" else do_apply)(a)


if __name__ == "__main__":
    main()
