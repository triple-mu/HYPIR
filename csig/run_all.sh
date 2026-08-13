#!/bin/bash
# CSIG-2026 赛道二训练数据集：一键复现。
# 说明与实测数字见同目录 README.md。
#
#   bash run_all.sh /path/to/data_root
#
# 分步跑（推荐，各源互不依赖，可并行）：
#   STEPS=4klsdb bash run_all.sh /path/to/data_root
set -e
ROOT=${1:?用法: bash run_all.sh <data_root>}
STEPS=${STEPS:-all}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT"/{logs,lists}
run() { [ "$STEPS" = all ] || [ "$STEPS" = "$1" ]; }

# ---- L1 通用大盘：4KLSDB（CC BY 4.0）----------------------------------------
# 全库 206 分片 / 879 GB。按「每 GB 域内产出」排序取高产片，实测 21.9 张/GB vs 全下 9.2 张/GB。
# 00118/00119 专供评估集，训练时由 prep_4klsdb.py 内部排除。
if run 4klsdb; then
  SH=""
  for i in 00118 00119 00120 00121 00122 00123 00124 00125 00126 00127 00128 00129 \
           00130 00131 00132 00133 00134 00135 00136 00137 00138 00139 00140 00141 \
           00142 00143 00144 00145 00146 00147 00148 00149 00150 00151 00152 00153 \
           00154 00164 00176 00178; do
    SH="$SH --include data/train_x4-$i-of-00206.parquet"
  done
  hf download SingleBicycle/4KLSDB --repo-type dataset --local-dir "$ROOT/4klsdb" \
      $SH --include metadata.jsonl 2>&1 | tail -2
  $PY "$HERE/prep_4klsdb.py" "$ROOT/4klsdb/data" "$ROOT/4klsdb_hr" \
      2>&1 | tee "$ROOT/logs/prep_4klsdb.log" | tail -3
  mv "$ROOT/4klsdb_list.txt" "$ROOT/lists/4klsdb.txt" 2>/dev/null || true
fi

# ---- L2 手机 ISP 层：PD12M（CDLA-Permissive-2.0，可商用）---------------------
# 1240 万行 -> 60 万候选（jpeg + 长边>=3400 + 宽高比 1.28-1.40 + caption 内容过滤）
# -> 边下边过三闸门，实测通过率 ~61%。
if run pd12m; then
  hf download Spawning/PD12M --repo-type dataset --local-dir "$ROOT/pd12m" \
      --include "metadata/*.parquet" 2>&1 | tail -1
  CSIG=$ROOT $PY "$HERE/prep_pd12m.py" index 2>&1 | tail -2
  CSIG=$ROOT $PY "$HERE/prep_pd12m.py" fetch --limit 15000 --workers 32 \
      2>&1 | tee "$ROOT/logs/pd12m.log" | tail -2
  cp "$ROOT/data/pd12m_list.txt" "$ROOT/lists/pd12m.txt" 2>/dev/null || true
fi

# ---- L3a 计算摄影锚点：Google HDR+（CC BY-SA ⚠️ 传染性未决，占比宜低）--------
if run hdrplus; then
  $PY "$HERE/dl_hdrplus.py" "$ROOT/hdrplus" 2>&1 | tail -2
  $PY "$HERE/screen_dir.py" "$ROOT/hdrplus" "$ROOT/lists/hdrplus.txt" --min-long 3000
fi

# ---- L3b 中文店招：ShopSign（需人工下载，见 README）--------------------------
if run shopsign && [ -d "$ROOT/ShopSign_1265" ]; then
  $PY "$HERE/screen_dir.py" "$ROOT/ShopSign_1265" "$ROOT/lists/shopsign.txt" \
      --copy-to "$ROOT/shopsign_hr"
fi

# ---- L3c 中文场景文字：CASIA-10K（需人工下载，见 README）---------------------
if run casia && [ -d "$ROOT/CASIA-10k" ]; then
  $PY "$HERE/screen_dir.py" "$ROOT/CASIA-10k" "$ROOT/lists/casia10k.txt" \
      --copy-to "$ROOT/casia_hr"
fi

# ---- 评估集：从 4KLSDB 的 00118/00119 造 held-out 配对 -----------------------
if run eval; then
  $PY "$HERE/build_eval.py" "$ROOT/4klsdb/data" "$ROOT/eval_p0" 2>&1 | tail -3
fi

# ---- 合并配比 ---------------------------------------------------------------
if run mix; then
  ARGS=""
  PROFILE_ARGS=""
  for kv in 4klsdb:0.55 pd12m:0.30 hdrplus:0.08 shopsign:0.05 casia10k:0.02; do
    n=${kv%%:*}; w=${kv##*:}
    [ -s "$ROOT/lists/$n.txt" ] && ARGS="$ARGS --src $n=$ROOT/lists/$n.txt:$w"
  done
  [ -s "$ROOT/lists/hdrplus.txt" ] && PROFILE_ARGS="--profile hdrplus=night_hdr"
  $PY "$HERE/build_mix.py" --out "$ROOT/lists/train_mix.txt" \
      $PROFILE_ARGS $ARGS
fi

echo "完成。训练用 file_list: $ROOT/lists/train_mix.txt"
