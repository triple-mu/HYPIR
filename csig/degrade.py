"""CSIG-2026 赛道二的退化模型 —— 合成训练数据用。

⚠️ 这份配方推翻了 ../../FINDINGS.md §5 原来那条「2x area 下采样 + cubic 上采样」。
实测（12 个验证集 crop，TOPIQ-FR + MANIQA 代理分，LQ 基线 p_rel ≡ 1.000）：

    GT 重采样 2x -> p_rel 7.360      <- 旧配方，几乎等于 GT（天花板是 8.336）
    GT 重采样 6x -> p_rel 1.514
    GT 重采样 8x -> p_rel 0.708
    GT 高斯 σ=3.5 -> p_rel 1.131
    GT 高斯 σ=4.0 -> p_rel 0.750

真实 LQ 的感知强度相当于 **≈7x 重采样 / σ≈3.6 高斯**，不是 2x。按旧配方合成的训练对，
其 LQ 输入的 p_rel 是 7.36 —— 已经接近 GT，模型会学成近似恒等映射。

旧结论错在哪：「LQ 自往返 2x 得 46-51 dB」只证明 LQ **至少**被 2x 带限了，是下界不是辨识。
一张被 7x 糊过的图，2x 往返当然也几乎无损。

## 配方

    LQ = JPEG(clip(a ⊙ (k ⊛ GT) + b, 0, 255), quality=95, subsampling=4:2:0)

k 的实测性质（六路取证 + 本人复核）：
  - 重尾：单极点 1/(1+(f/f0)^2) 到四次之间，纯高斯被拒绝（logRMSE 0.078-0.74 vs 1.33-5.47）
  - 各向异性：长短轴比 1.2-1.4，方向逐图随机
  - 图内空间变化 ±40%（σ 的 p10/p90）
  - σ ∈ [1.85, 4.06]，逐图不同

**没有任何 LTI 核能配平**：最优 KxK 核（K 开到 31）的天花板是 26/35/28 dB，
而同流程反解已知合成退化能到 47.6 dB —— 20 dB 的结构性落差说明存在显著非 LTI 成分。
所以这里刻意做成**随机化**的，不要钉死一组参数。

用法：
    from degrade import degrade, DegradeConfig
    lq = degrade(gt_bgr_uint8, rng=np.random.default_rng(0))
"""

import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 实测标定的参数范围
#
# SIGMA_RANGE 是按**感知分**标定的，不是按频谱拟合的 σ 直接搬。原因：重尾极点滤波器
# 在同一 σ 下比高斯更狠，且 SPATIAL_VAR 的两档混合会让感知强度进一步偏向模糊那一档。
# 标定过程（pole=1、各向同性、无空间变化、无仿射，12 个验证 crop）：
#     σ=1.8 -> p_rel 2.419    σ=2.4 -> 1.492    σ=2.8 -> 1.091    σ=3.2 -> 0.803
# 即 σ≈2.9 对应真实 LQ 的 p_rel=1.000。下面的区间以此为中心展开，
# 跨度对应真实数据的逐图差异（case2 明显更轻，case1 最重）。
SIGMA_RANGE = (1.85, 3.45)     # 感知标定后的等效 σ；实测合成分布中位 p_rel≈1.0
ANISO_RANGE = (1.0, 1.45)      # 长短轴比，实测 1.2-1.4
POLE_RANGE = (0.9, 2.1)        # 极点阶数；1=单极点，2=四次。实测落在这之间
SPATIAL_VAR = 0.4              # 图内 σ 的相对变化幅度，实测 ±40%
AFFINE_A = (1.00, 1.10)        # 逐通道增益，实测 a ∈ [1.00, 1.10]
AFFINE_B = (-14.0, 0.0)        # 逐通道偏置，实测 b ∈ [-13.5, 0]
JPEG_QUALITY = 95              # 字节级确认：128 个量化系数与 IJG q95 全等
P_AFFINE = 0.6                 # 施加色调偏移的概率（实测三张图里只有 case3 显著）


def _sigma_to_f0(sigma: float) -> float:
    """把等效高斯 σ 折算成 -6 dB 截止频率（cyc/px）。

    高斯的功率响应 exp(-(2πfσ)^2)，半功率点 2πfσ = sqrt(ln2) => f = 0.1325/σ。
    实测 σ=1.85-4.06 对应 f0=0.046-0.100，与此式一致。
    """
    return 0.1325 / sigma


def _lowpass_fft(x: torch.Tensor, sigma: float, aniso: float, theta: float,
                 pole: float) -> torch.Tensor:
    """各向异性、重尾低通。x: [1,3,H,W] float32 [0,255]。

    H(f) = 1 / (1 + (f'/f0)^2)^pole，f' 是按 (aniso, theta) 做椭圆缩放后的频率。
    pole=1 是单极点（Lorentzian，重尾），pole=2 是四次。纯高斯 exp() 已被实测拒绝。
    """
    _, _, h, w = x.shape
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]        # [-0.5, 0.5) cyc/px
    fx = torch.fft.fftfreq(w, device=x.device)[None, :]
    c, s = np.cos(theta), np.sin(theta)
    # 旋转到主轴，长轴方向的截止频率放大 aniso 倍（即那个方向更清晰）
    u = (fx * c + fy * s) / aniso
    v = -fx * s + fy * c
    f0 = _sigma_to_f0(sigma)
    resp = 1.0 / (1.0 + ((u * u + v * v) / (f0 * f0))) ** pole
    return torch.fft.ifft2(torch.fft.fft2(x) * resp).real


def degrade(gt_bgr: np.ndarray, rng: np.random.Generator,
            device: str = "cuda") -> np.ndarray:
    """GT (BGR uint8 HxWx3) -> LQ (BGR uint8 HxWx3)。

    三段：各向异性重尾低通（含图内空间变化）-> 逐通道仿射+clip -> JPEG q95 4:2:0。
    """
    x = torch.from_numpy(gt_bgr).to(device).permute(2, 0, 1)[None].float()

    # 1) 低通。图内空间变化用两档模糊 + 平滑随机掩码混合实现（实测 σ 的 p10/p90 差 ±40%）
    sigma = rng.uniform(*SIGMA_RANGE)
    aniso = rng.uniform(*ANISO_RANGE)
    theta = rng.uniform(0, np.pi)
    pole = rng.uniform(*POLE_RANGE)
    lo = _lowpass_fft(x, sigma * (1 - SPATIAL_VAR), aniso, theta, pole)
    hi = _lowpass_fft(x, sigma * (1 + SPATIAL_VAR), aniso, theta, pole)
    m = torch.from_numpy(rng.random((1, 1, 8, 8)).astype(np.float32)).to(device)
    m = F.interpolate(m, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    x = lo * (1 - m) + hi * m

    # 2) 逐通道仿射 + clip（clip 到 0 会产生黑位压死，实测 case3 有 4.5-7.8% 像素死在 0）
    if rng.random() < P_AFFINE:
        a = torch.tensor(rng.uniform(*AFFINE_A, size=3), device=device,
                         dtype=torch.float32).view(1, 3, 1, 1)
        b = torch.tensor(rng.uniform(*AFFINE_B, size=3), device=device,
                         dtype=torch.float32).view(1, 3, 1, 1)
        x = x * a + b
    x = x.clamp(0, 255).round()

    bgr = x[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    # 3) JPEG q95 4:2:0（cv2 默认就是 4:2:0，且与 PIL/libjpeg 输出逐字节相同）
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
