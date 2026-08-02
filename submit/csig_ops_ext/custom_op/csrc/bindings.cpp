#include <torch/library.h>

// 仅声明 schema；数学语义、形状约定与 sm70/sm86 实现细节见各 .cu 文件顶部注释。
// 这四个算子一一对应 model_dir/runner.py 里的 triton_* 实现，用于逐算子对拍与替换实验。
TORCH_LIBRARY(custom_op, m)
{
    // NHWC 上的 GroupNorm(+可选 SiLU)。x 必须 channels_last；act: 0=无、1=SiLU。
    // mean/rstd 按 (n, group) 统计，全程 fp32 累加，写回时一次舍入。
    m.def("group_norm_act(Tensor x, Tensor weight, Tensor bias, int num_groups, float eps, int act) -> Tensor");
    // (..., 2D) -> (..., D)：h * 0.5 * g * (1 + erf(g / sqrt(2)))，erf 精确版 gelu（非 tanh 近似）。
    m.def("geglu(Tensor y2) -> Tensor");
    // z = mean + exp(0.5 * clamp(logvar, -30, 20)) * noise。
    // moments (N,8,H,W) channels_last，noise (N,4,H,W) 必须 NCHW（保持 torch.randn 原始布局，
    // 使随机数流与 eager 一致），输出 (N,4,H,W) channels_last。
    m.def("sample_latent(Tensor moments, Tensor noise) -> Tensor");
    // 单 (batch,head)、非因果、head_dim=512：softmax(q @ kᵀ / sqrt(512)) @ v。
    // q/k/v (1,N,512)，最后一维连续、行 stride 可为 512 或 1536（融合 QKV 的 chunk 视图）。
    m.def("attn_d512(Tensor q, Tensor k, Tensor v) -> Tensor");
}
