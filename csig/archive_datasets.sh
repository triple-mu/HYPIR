#!/bin/bash
# 归档「不可再生」的数据集元信息，不备份图片本身。
#
# 原则：图片几百 GB 且可由脚本 + URL 清单重建；真正丢不起的是「选中了哪些」——
# 即 URL 清单、patch 索引 parquet、混合配比。这几样加起来只有几十 MB。
# 筛选参数记在 docs/DATASETS.md，脚本在 git 里，两者合起来才能复现。
#
#   bash csig/archive_datasets.sh [输出目录]
set -eu
CSIG=${CSIG:-/root/.cache/huggingface/csig}
OUT=${1:-$CSIG/archive}
STAMP=$(date +%Y%m%d)
mkdir -p "$OUT"
TAR="$OUT/datasets_meta_$STAMP.tar.gz"

cd "$CSIG/data"
FILES=()
for f in pd12m_urls.txt lsdir_512_nulltxt.parquet mix_lsdir_pd12m.parquet; do
    [ -f "$f" ] && FILES+=("$f")
done
# 各源筛选后的 file_list（screen_dir.py 的产物）
for f in *_list.txt; do [ -f "$f" ] && FILES+=("$f"); done

if [ ${#FILES[@]} -eq 0 ]; then echo "没有可归档的元信息文件"; exit 1; fi

# 附一份清单：每个文件的行数与 sha256，便于日后校验是不是同一份
{
    echo "# 数据集元信息归档 $STAMP"
    echo "# 筛选参数见 docs/DATASETS.md；生成脚本见 csig/prep_*.py 与 csig/screen_dir.py"
    echo
    for f in "${FILES[@]}"; do
        printf "%-40s %12s 行  sha256=%s\n" "$f" \
            "$(wc -l < "$f" 2>/dev/null || echo -)" \
            "$(sha256sum "$f" | cut -c1-16)"
    done
} > "$OUT/MANIFEST_$STAMP.txt"

tar czf "$TAR" "${FILES[@]}" -C "$OUT" "MANIFEST_$STAMP.txt"
echo "已归档 -> $TAR  ($(du -h "$TAR" | cut -f1))"
cat "$OUT/MANIFEST_$STAMP.txt"
