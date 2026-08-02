#!/bin/bash
# 训练看门狗：机器是共享的，docker 重启/外部 kill 都会打断长跑任务。
# 被杀后自动从最近 checkpoint 续训。
#   GPUS=0,1,4,5 CFG=configs/csig_train.yaml bash csig/run_train.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.." || exit 1
CFG=${CFG:-configs/csig_train.yaml}
GPUS=${GPUS:-0,1}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=$(grep -oP '^output_dir:\s*\K\S+' "$CFG")
TARGET=$(grep -oP '^max_train_steps:\s*\K[0-9]+' "$CFG")
mkdir -p "$OUT" "$CSIG/logs"
LOG=$CSIG/logs/$(basename "$OUT").log
WD=$CSIG/logs/$(basename "$OUT")_watchdog.log

for i in $(seq 1 200); do
    LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -n "$LAST" ] && [ "$LAST" -ge "$TARGET" ]; then
        echo "[watchdog] $(date '+%F %T') 已达 $TARGET 步，完成" >> "$WD"; break
    fi
    RUN=$CSIG/logs/run_$(basename "$OUT").yaml
    if [ -n "$LAST" ]; then
        sed "s|^resume_from_checkpoint: .*|resume_from_checkpoint: $OUT/checkpoint-$LAST|" "$CFG" > "$RUN"
    else
        sed "s|^resume_from_checkpoint: .*|resume_from_checkpoint: ~|" "$CFG" > "$RUN"
    fi
    echo "[watchdog] $(date '+%F %T') 第 $i 次启动 resume=${LAST:-无} GPU=$GPUS" >> "$WD"
    env CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes "$N" \
        --mixed_precision fp16 train.py --config "$RUN" >> "$LOG" 2>&1
    echo "[watchdog] $(date '+%F %T') 退出码 $? (最近 ckpt: ${LAST:-无})" >> "$WD"
    sleep 30
done
