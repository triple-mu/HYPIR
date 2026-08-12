"""按配比把多个 patch parquet 合成一份训练索引。

为什么按**源图**分层而不是随机抽 patch：3000 步只消费 0.35 epoch，
唯一非零的收益机制是「图像级多样性」（见 docs/DATASETS.md）。随机抽 patch 会把
同一张源图的 patch 拆散，等于在同一批内容上重复采样；按源图整取才是真的换分布。
取源图时按文件名排序取前 N 个，确定性可复现。

    python csig/mix_parquet.py 出.parquet base.parquet add.parquet:0.30 [more.parquet:0.1 ...]

比例是「该源在最终池中的目标占比」，base（第一个）不设比例、全量保留，
其余按 base 的量反推各自需要多少 patch。
"""
import argparse
import os
import sys

import pandas as pd


def src_key(paths):
    """从 patch 路径反推源图名：<dir>/<base>_<row>_<col>.png -> <base>"""
    return paths.str.rsplit("/", n=1).str[-1].str.rsplit("_", n=2).str[0]


def take_by_source(df, n_target):
    """按源图整取，直到 patch 数刚好够 n_target。返回 (子集, 用了几张源图)。"""
    key = src_key(df["image_path"])
    counts = key.value_counts().sort_index()          # 按源图名排序，确定性
    keep, acc = [], 0
    for s, c in counts.items():
        if acc >= n_target:
            break
        keep.append(s)
        acc += c
    return df[key.isin(keep)], len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("base", help="基准源 parquet，全量保留")
    ap.add_argument("adds", nargs="+", help="附加源，格式 path.parquet:比例（如 0.30）")
    a = ap.parse_args()

    base = pd.read_parquet(a.base)
    n_base = len(base)
    parts = [base]
    print("base %-46s %7d patch (%d 源图)" % (
        os.path.basename(a.base), n_base, src_key(base["image_path"]).nunique()))

    specs = []
    for s in a.adds:
        path, _, frac = s.rpartition(":")
        specs.append((path, float(frac)))
    tot_frac = sum(f for _, f in specs)
    if tot_frac >= 1.0:
        sys.exit("附加源比例之和 %.2f 必须 < 1.0（base 要占剩下的）" % tot_frac)

    # base 占 (1 - Σfrac)，由此反推总量
    total = n_base / (1.0 - tot_frac)
    for path, frac in specs:
        want = int(round(total * frac))
        df = pd.read_parquet(path)
        sub, n_src = take_by_source(df, want)
        parts.append(sub)
        print("add  %-46s %7d patch (%d 源图, 目标 %d, 池中共 %d)" % (
            os.path.basename(path), len(sub), n_src, want, len(df)))

    mix = pd.concat(parts, ignore_index=True)
    mix.to_parquet(a.out)
    print("\n合计 %d patch -> %s" % (len(mix), a.out))
    for path, _ in [(a.base, 0)] + specs:
        tag = os.path.basename(path).split("_")[0]
        n = mix["image_path"].str.contains("/%s" % tag).sum() if tag else 0
        print("  %-20s %7d (%.1f%%)" % (tag, n, 100.0 * n / len(mix)))


if __name__ == "__main__":
    main()
