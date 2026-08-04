#!/bin/bash
# 扫训练快照：对每个采样点的 raw 与 EMA 各跑一遍 eval_bench，分摊到多张卡。
#
#   GPUS=5,6,7 STRIDE=1000 LIMIT=300 KIND=both bash csig/sweep_snapshots.sh $CSIG/out/taesd_main
#
# MAX_STEP 限定上界，配合「已评过就跳过」可以两趟扫出疏密不同的曲线：
#   STRIDE=250 MAX_STEP=3000 ...   # 峰值区间密扫
#   STRIDE=1000 ...                # 远端粗扫，前一趟覆盖过的自动跳过
#
# KIND=both|ema|raw。默认 both —— 历史上出现过 raw 反超 EMA 之后又掉回去的振荡，
# 只看其中一个会把振荡误判成收敛。先用 KIND=ema 粗扫定位峰值、再对峰值附近补 raw，
# 能省一半机时。
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.." || exit 1

OUT=${1:?用法: bash csig/sweep_snapshots.sh <训练输出目录>}
GPUS=${GPUS:-5,6,7}
STRIDE=${STRIDE:-1000}
MAX_STEP=${MAX_STEP:-0}          # 0 = 不限
LIMIT=${LIMIT:-300}
VAE=${VAE:-taesd}
KIND=${KIND:-both}
case "$KIND" in both) KINDS="raw ema";; ema) KINDS="ema";; raw) KINDS="raw";;
    *) echo "KIND 只能是 both|ema|raw"; exit 1;; esac
NAME=$(basename "$OUT")

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}

JOBS=$(mktemp)
i=0
for d in $(ls -d "$OUT"/snapshots/step-* 2>/dev/null | sort); do
    step=$(basename "$d" | sed 's/step-//')
    [ $((10#$step % STRIDE)) -eq 0 ] || continue
    [ "$MAX_STEP" -gt 0 ] && [ $((10#$step)) -gt "$MAX_STEP" ] && continue
    for kind in $KINDS; do
        [ "$kind" = raw ] && w="$d/state_dict.pth" || w="$d/ema_state_dict.pth"
        [ -f "$w" ] || continue
        tag="${NAME}_$((10#$step))_${kind}"
        # 已经评过就跳过，方便中断后接着扫
        [ -f "$CSIG/out/bench_$tag.json" ] && continue
        echo "${GPU_ARR[$((i % NG))]}|$w|$tag"
        i=$((i + 1))
    done
done > "$JOBS"

echo "待评 $(wc -l < "$JOBS") 份权重，$NG 张卡并行，每份 $LIMIT 对"
tr '|' ' ' < "$JOBS" | xargs -P "$NG" -L1 bash -c '
    CUDA_VISIBLE_DEVICES=$0 python csig/eval_bench.py --weight "$1" --vae '"$VAE"' \
        --tag "$2" --limit '"$LIMIT"' > "$CSIG/logs/bench_$2.log" 2>&1 \
        && echo "  完成 $2" || echo "  失败 $2（看 $CSIG/logs/bench_$2.log）"
'
rm -f "$JOBS"
echo "全部完成，出表: python csig/bench_table.py"
