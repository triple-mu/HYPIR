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

# 判别器骨干（open_clip 的 ConvNext + timm 的 forward_intermediates）对这两个版本敏感：
# 新版改了特征抽取，decoder 通道会变成 384（期望 768），四个 rank 同时炸在第一个 D 步。
# 报错信息离根因很远，且两台机器的 venv 是各自装的，很容易只在一台上对。这里当场拦下。
csig_check_versions() {
    python - <<'PY'
import sys
want = {"open_clip_torch": ("open_clip", "2.31.0"), "timm": ("timm", "1.0.15")}
bad = []
for pkg, (mod, ver) in want.items():
    got = __import__(mod).__version__
    if got != ver:
        bad.append("  %s: 期望 %s, 实际 %s   ->  pip install %s==%s" % (pkg, ver, got, pkg, ver))
if bad:
    sys.exit("[env] 判别器依赖版本不对，训练必炸：\n" + "\n".join(bad))
PY
}
