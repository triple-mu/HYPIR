"""把各源的 file_list 按配比合并成训练用的单一 file_list。

HYPIR 的 dataset 只吃一个 file_list，所以「配比」通过**按权重重复条目**实现
（过采样小源），而不是加权采样器 —— 这样不用改 HYPIR 的 dataloader。

配比的依据（见 ../README.md 第四节）：
  4KLSDB   通用大盘，唯一 100% >=3840px 的大规模语料，但 49.6% 是人像、中文线索仅 1.7%
  PD12M    手机 ISP 层，~90% 真手机 EXIF、bpp 3.11，用来纠正 4KLSDB 高频尾部偏高 6.57 倍的偏差
  HDR+     计算摄影锚点，Google Pixel 真多帧融合管线成品
  ShopSign 中文店招层，国产机(小米/OPPO/vivo/华为)拍的中文招牌，别处拿不到
  CASIA10K 同上，补充

用法:
    python build_mix.py --out /path/train_list.txt \\
        --src 4klsdb=/path/4klsdb_list.txt:0.55 \\
        --src pd12m=/path/pd12m_list.txt:0.30 \\
        --src hdrplus=/path/hdrplus_list.txt:0.08 \\
        --src shopsign=/path/shopsign_list.txt:0.07 \
        --profile hdrplus=night_hdr

输出仍兼容纯路径；带 profile 的条目写成 ``path<TAB>profile``，Dataset 会优先
使用该标签，避免把 HDR+ 成品随机套入普通白天链路。
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True,
                    help="名字=list路径:权重，可重复")
    ap.add_argument("--profile", action="append", default=[],
                    help="名字=ordinary|night_hdr；未指定的源由训练时 auto 采样")
    ap.add_argument("--total", type=int, default=0,
                    help="目标总条目数；0 表示按最大源自动定（不缩小任何源）")
    ap.add_argument("--seed", type=int, default=20260802)
    a = ap.parse_args()

    profiles = {}
    for item in a.profile:
        name, value = item.split("=", 1)
        if value not in ("ordinary", "night_hdr"):
            raise ValueError("未知 profile: %s" % value)
        profiles[name] = value

    srcs = {}
    for s in a.src:
        name, rest = s.split("=", 1)
        path, w = rest.rsplit(":", 1)
        items = [line.strip() for line in open(path) if line.strip()]
        assert items, "空 list: %s" % path
        srcs[name] = {"items": items, "w": float(w), "profile": profiles.get(name)}
    unknown_profiles = set(profiles) - set(srcs)
    if unknown_profiles:
        raise ValueError("--profile 引用了不存在的源: %s" % sorted(unknown_profiles))

    wsum = sum(v["w"] for v in srcs.values())
    for v in srcs.values():
        v["w"] /= wsum

    # 自动定总量：让「重复倍数最小的那个源」恰好重复 1 次，避免任何源被下采样丢信息
    total = a.total or max(int(round(len(v["items"]) / v["w"])) for v in srcs.values())

    rng = random.Random(a.seed)
    out = []
    print("%-10s %8s %8s %8s %9s" % ("源", "原始张数", "目标条目", "重复倍数", "实际占比"))
    print("-" * 50)
    for name, v in srcs.items():
        want = int(round(total * v["w"]))
        reps = want / len(v["items"])
        picked = []
        while len(picked) < want:
            pool = v["items"][:]
            rng.shuffle(pool)
            picked += pool[:want - len(picked)]
        if v["profile"]:
            picked = [item.split("\t", 1)[0] + "\t" + v["profile"] for item in picked]
        out += picked
        print("%-10s %8d %8d %8.2f %8.1f%%" %
              (name, len(v["items"]), len(picked), reps, 100.0 * len(picked) / total))

    rng.shuffle(out)
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("-" * 50)
    print("合计 %d 条 -> %s" % (len(out), a.out))
    print("注：一张 4096x3072 可切 48 个不重叠 512 crop，故 %d 条 ≈ %.0f 万 crop 的采样空间"
          % (len(out), len(set(out)) * 48 / 1e4))


if __name__ == "__main__":
    main()
