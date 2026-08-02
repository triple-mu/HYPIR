#!/bin/bash
# 训练看门狗：机器是共享的，docker 重启/外部 kill 都会打断长跑任务。
# 被杀后自动从最近 checkpoint 续训。
#   GPUS=0,1,4,5 CFG=configs/csig_train.yaml bash csig/run_train.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$HERE/.." || exit 1
CFG=${CFG:-configs/csig_train.yaml}
NGPU=${NGPU:-4}

# 选卡：机器是共享的，必须避开别人。判据比"当前显存<2GB"更严——
# 连续采样 3 次（间隔 2 s），三次都空闲才算，避开正在启动的任务。
pick_gpus() {
    local n=$1 i
    local -A busy
    for i in 1 2 3; do
        while IFS=, read -r idx used util; do
            used=${used// /}; used=${used//MiB/}; util=${util// /}; util=${util//%/}
            [ "$used" -gt 1024 ] || [ "$util" -gt 5 ] && busy[$idx]=1
        done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
        [ $i -lt 3 ] && sleep 2
    done
    local free=()
    for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        [ -z "${busy[$idx]}" ] && free+=("$idx")
    done
    (IFS=,; echo "${free[*]:0:$n}")
}

if [ -z "$GPUS" ]; then
    GPUS=$(pick_gpus "$NGPU")
    echo "[watchdog] 自动选卡: ${GPUS:-无}" >&2
fi
N=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
if [ "$N" -lt 1 ]; then echo "[watchdog] 没有空闲 GPU，退出" >&2; exit 1; fi
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
