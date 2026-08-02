# CSIG-2026 训练环境（容器 sglang-diffusion-triplemu-training）
# 用法: source /workspace/csig/HYPIR/csig/env.sh
source /workspace/.torch/bin/activate
export CSIG=/root/.cache/huggingface/csig
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/workspace/csig/HYPIR:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
# 容器未获 NVSwitch 多播(nvidia-caps)权限，NCCL 的 NVLS 在 3 卡以上报 CUDA 401
export NCCL_NVLS_ENABLE=0
# inductor 缓存放挂载卷：看门狗重启后编译 200s -> 66s
export TORCHINDUCTOR_CACHE_DIR=$CSIG/.inductor
# 压缩 allocated -> nvidia-smi 之间的碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
