#!/usr/bin/env bash
# nsys / ncu 采样封装。载荷是 csig_bench/profile_runner.py，它用 cudaProfilerStart/Stop
# 把窗口收在预热之后的稳态迭代上，所以这里两个工具都要开「从 API 起采」的开关：
#     nsys  --capture-range=cudaProfilerApi --capture-range-end=stop
#     ncu   --profile-from-start off
#
# 用法：
#     csig_bench/profile.sh nsys  [profile_runner.py 的参数...]
#     csig_bench/profile.sh ncu   [...]
#     csig_bench/profile.sh both  [...]
#
# 例：
#     csig_bench/profile.sh nsys --backend cuda --mode infer
#     csig_bench/profile.sh nsys --backend cuda --mode graph   # 抓 CUDA Graph
#     csig_bench/profile.sh ncu  --backend cuda --mode tile --iters 1
#
# 环境变量：
#     PY        python 解释器（默认 python）
#     NSYS/NCU  工具路径（默认 nsys / ncu）。注意 nsys 2026.4.1 在某些容器里注入失效——
#               能生成 .nsys-rep 但里面没有任何 CUDA kernel 数据，且带 --capture-range
#               时会在收尾段错误；换 2024.6.2 即可（CUDA 12.x 的 toolkit 里自带一份）
#     OUTDIR    产物目录（默认 prof/）
#     TAG       文件名后缀（默认由 --backend/--mode 拼出）
#     NCU_SET   ncu 的 metric 集合（默认 base；full 慢很多）
set -euo pipefail

TOOL="${1:?用法: profile.sh nsys|ncu|both [profile_runner.py 参数...]}"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
NSYS="${NSYS:-nsys}"
NCU="${NCU:-ncu}"
OUTDIR="${OUTDIR:-$REPO_ROOT/prof}"
PAYLOAD="$REPO_ROOT/csig_bench/profile_runner.py"
mkdir -p "$OUTDIR"

# 从参数里抠出 backend/mode 拼默认文件名
be="auto"; mode="infer"; prev=""
for a in "$@"; do
    case "$prev" in --backend) be="$a";; --mode) mode="$a";; esac
    prev="$a"
done
TAG="${TAG:-${be}_${mode}}"

run_nsys() {
    # --cuda-graph-trace=node：把图内的每个 kernel 单独记下来，否则整张图只显示成一个节点
    echo "== nsys -> $OUTDIR/nsys_$TAG.nsys-rep =="
    "$NSYS" profile \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --trace=cuda,nvtx,cublas,cudnn,osrt \
        --cuda-graph-trace=node \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output="$OUTDIR/nsys_$TAG" \
        "$PY" "$PAYLOAD" "$@"
    "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
        --output "$OUTDIR/nsys_$TAG" "$OUTDIR/nsys_$TAG.nsys-rep" >/dev/null 2>&1 || true
}

run_ncu() {
    echo "== ncu -> $OUTDIR/ncu_$TAG.ncu-rep =="
    "$NCU" --profile-from-start off \
        --set "${NCU_SET:-base}" \
        --nvtx \
        --graph-profiling node \
        --force-overwrite \
        --export "$OUTDIR/ncu_$TAG" \
        "$PY" "$PAYLOAD" "$@"
    "$NCU" --import "$OUTDIR/ncu_$TAG.ncu-rep" --csv --page raw \
        > "$OUTDIR/ncu_$TAG.csv" 2>/dev/null || true
}

case "$TOOL" in
    nsys) run_nsys "$@";;
    ncu)  run_ncu "$@";;
    both) run_nsys "$@"; run_ncu "$@";;
    *) echo "未知工具: $TOOL（可选 nsys / ncu / both）" >&2; exit 1;;
esac
echo "产物在 $OUTDIR/"
