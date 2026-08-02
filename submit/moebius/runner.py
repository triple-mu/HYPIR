"""Moebius 推理 Runner（CSIG-2026 赛题二提交入口）。

自包含实现了 Moebius（hustvl，ECCV'26，0.22B 轻量扩散模型）的全部模型定义
（LλMI UNet + SD/SDXL 的 AutoencoderKL + DDIM），推理路径只依赖 torch。
没有 diffusers / transformers / timm / einops / numpy / yaml，也没有 text encoder
（Moebius 的条件是一张 20x3072 的可学习 ID 表，且在导出期已被吸收成常量）。

本文件与权重同放在 model_dir/ 下：
    model_dir/
    ├── runner.py             本文件
    ├── moebius_weights.pth   fp16 打包权重（BN 已折叠、cross-λ 常量已预计算）
    └── config.json           可选；缺省走本文件的 DEFAULTS

用法：
    # 1. 一次性导出打包权重
    python model_dir/runner.py --export --src . --dst model_dir

    # 2. 推理
    from runner import Runner
    runner = Runner("model_dir")
    out = runner.infer(tile)    # 评测入口：单个 512x512 分块，[-1,1] -> 同形状
    out = runner.enhance(img)   # 整图入口：分块 -> 逐块调 infer -> 高斯融合

infer 与 enhance 的分工：赛题只在 512x512 上量 infer 的时延，故 infer 是一条形状恒定、
无 RNG 的最短路径；整图所需的短边保护、滑窗、融合都在 enhance 里，且 enhance 逐 tile
回调 infer，两条路径共用同一段模型链。

分辨率约束：cross-λ 的 rel_pos_emb 形状 (4096/1024/256, 10, k) 与 64x64 latent 死绑，
故 patch_size 恒为 512，不是可调旋钮——改它要重训。
"""

import argparse
import json
import math
import os
import time

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Callable, Dict, List, Optional, Tuple, Union

WEIGHTS_FILE = "moebius_weights.pth"
VAE_SCALING_FACTOR = 0.13025  # SDXL VAE（Moebius 用的这份），不是 SD2.1 的 0.18215
LATENT_SIZE = 64              # 512 // 8，见文件头的分辨率约束
NUM_COND_TOKENS = 10          # num_embeddings // 2，cross-λ 的 m

DEFAULTS = {
    "weights": WEIGHTS_FILE,
    "patch_size": 512,
    "stride": 256,
    "seed": 231,
    # sdedit: 从加噪的 LQ latent 出发跑 DDIM；onestep: 固定 timestep 单步（微调后的形态）。
    # ⚠️ 预训练权重是 inpainting 的，零训练做不出画质增强——实测见 ../FINDINGS.md：
    # 全网格最优只有 p_rel 1.037（HYPIR 是 2.65），mask=0 时严格退化为恒等重建。
    # 要真正出图必须先微调，那时 mode 应切到 onestep。
    "mode": "sdedit",
    "steps": 4,          # sdedit 的 DDIM 步数
    "t_start": 250,      # sdedit 的起始 timestep
    "model_t": 200,      # onestep 的固定 timestep
    "mask_value": 1.0,   # 常量 mask 通道的取值
    "unet_channels_last": True,   # LayerNorm 的 permute 在 NHWC 下是零拷贝视图
    "vae_channels_last": False,   # 纯 PyTorch 下 VAE 实测 NCHW 更快
}


def _load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Section 1: 融合算子（先只有 eager 实现，Triton/CUDA 后端从这里接入）
# ---------------------------------------------------------------------------

def group_norm_act(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                   num_groups: int, eps: float, act: Optional[str]) -> torch.Tensor:
    y = F.group_norm(x, num_groups, weight, bias, eps)
    return F.silu(y) if act == "silu" else y


def sample_latent(moments: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """等价 DiagonalGaussianDistribution.sample()。"""
    mean, logvar = moments.chunk(2, dim=1)
    return mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * noise


def silu_glu(x: torch.Tensor) -> torch.Tensor:
    """GLUMBConv 的门控：(B, 2h, H, W) -> (B, h, H, W)。"""
    h, gate = x.chunk(2, dim=1)
    return h * F.silu(gate)


def attn_d512(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """VAE mid block 的单头 head_dim=512 注意力。"""
    return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]


def gn_act(norm: nn.GroupNorm, x: torch.Tensor,
           act: Optional[str] = "silu") -> torch.Tensor:
    return group_norm_act(x, norm.weight, norm.bias, norm.num_groups, norm.eps, act)


# ---------------------------------------------------------------------------
# Section 2: 共享视觉模块（UNet 与 VAE 都用）
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int = 320,
                       max_period: int = 10000) -> torch.Tensor:
    # 等价 diffusers get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int = 320, dim: int = 1280) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class ResnetBlock2D(nn.Module):
    """VAE 的 resnet（普通 3x3 conv，无 temb）。UNet 用的是 DWResnetBlock2D。"""

    def __init__(self, in_ch: int, out_ch: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(gn_act(self.norm1, x))
        h = self.conv2(gn_act(self.norm2, h))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    """UNet 用 padding=1；VAE encoder 用 padding=0 + 前置非对称 pad (0,1,0,1)。"""

    def __init__(self, ch: int, padding: int = 1) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


_CONVT_TAPS = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def fold_upsample_conv(conv: nn.Conv2d,
                       channels_last: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 nearest 上采样 + 3x3 conv(pad=1) 折成等价的 ConvTranspose2d(k=4, s=2, p=1)。

    逐轴推导（索引映射可分离）：上采样后 u[m] = x[m // 2]，故
        输出 P=2t   : dp=-1 取源 t-1（权重 W0）；dp=0,1 取源 t（权重 W1+W2）
        输出 P=2t+1 : dp=-1,0 取源 t（权重 W0+W1）；dp=1 取源 t+1（权重 W2）
    ConvTranspose 的 kp = P - 2i + 1 与 P 的奇偶自动对齐，于是
        V[0]=W2, V[1]=W1+W2, V[2]=W0+W1, V[3]=W0
    两端越界项在两种写法下同为 0，边界也等价。float64 实测相对误差 2e-16。
    乘加数 9*C*O*4HW -> 16*C*O*HW，即 4/9。
    """
    W = conv.weight.detach()  # (out, in, 3, 3)
    o, c = W.shape[:2]
    v = torch.zeros(o, c, 4, 4, dtype=W.dtype, device=W.device)
    for kp, sp in _CONVT_TAPS.items():
        for kq, sq in _CONVT_TAPS.items():
            v[:, :, kp, kq] = W[:, :, sp, :][:, :, :, sq].sum((2, 3))
    v = v.transpose(0, 1).contiguous()  # ConvTranspose 权重布局是 (in, out, kH, kW)
    if channels_last:
        v = v.contiguous(memory_format=torch.channels_last)
    return v, conv.bias.detach()


class Upsample2D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self._convt = None  # Runner 预折叠出的 (weight, bias)，见 fold_upsample_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._convt is not None:
            return F.conv_transpose2d(x, self._convt[0], self._convt[1], stride=2, padding=1)
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class VAEAttention(nn.Module):
    """VAE mid block attention：单头 512、qkv 带 bias、GroupNorm 前置、残差连接。"""

    def __init__(self, ch: int = 512, eps: float = 1e-6) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, ch, eps=eps)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = nn.ModuleList([nn.Linear(ch, ch)])
        self._qkv_w = None  # 融合 QKV 权重/偏置（Runner 预 cat）
        self._qkv_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        # 4D GN 与原 3D 写法数学相同；channels_last 下 permute+reshape 是零拷贝 view
        hs = gn_act(self.group_norm, x, act=None).permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self._qkv_w is not None:
            q, k, v = F.linear(hs, self._qkv_w, self._qkv_b).chunk(3, dim=-1)
        else:
            q, k, v = self.to_q(hs), self.to_k(hs), self.to_v(hs)
        hs = attn_d512(q, k, v)
        hs = self.to_out[0](hs)
        return hs.reshape(b, h, w, c).permute(0, 3, 1, 2) + residual


# ---------------------------------------------------------------------------
# Section 3: AutoencoderKL（SDXL VAE，全部 GroupNorm eps=1e-6）
#
# 这一节与 HYPIR 方案的实现逐字相同：SDXL VAE 与 SD2.1 VAE 结构一致，实测两边
# state_dict 的 248 个 key 逐条匹配（无 missing / extra / shape mismatch）。
# ---------------------------------------------------------------------------

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, add_downsample: bool = True) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_ch, out_ch),
                                      ResnetBlock2D(out_ch, out_ch)])
        self.downsamplers = nn.ModuleList(
            [Downsample2D(out_ch, padding=0)]) if add_downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, add_upsample: bool = True) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_ch, out_ch),
                                      ResnetBlock2D(out_ch, out_ch),
                                      ResnetBlock2D(out_ch, out_ch)])
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class VAEMidBlock(nn.Module):
    def __init__(self, ch: int = 512) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([VAEAttention(ch)])
        self.resnets = nn.ModuleList([ResnetBlock2D(ch, ch), ResnetBlock2D(ch, ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class VAEEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownEncoderBlock2D(128, 128),
            DownEncoderBlock2D(128, 256),
            DownEncoderBlock2D(256, 512),
            DownEncoderBlock2D(512, 512, add_downsample=False),
        ])
        self.mid_block = VAEMidBlock(512)
        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6)
        self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(gn_act(self.conv_norm_out, x))


class VAEDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid_block = VAEMidBlock(512)
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(512, 512),
            UpDecoderBlock2D(512, 512),
            UpDecoderBlock2D(512, 256),
            UpDecoderBlock2D(256, 128, add_upsample=False),
        ])
        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(gn_act(self.conv_norm_out, x))


class AutoencoderKLLite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))


# ---------------------------------------------------------------------------
# Section 4: Moebius 的 LλMI 基础层
# ---------------------------------------------------------------------------

class DWConv2d(nn.Module):
    """timm DepthwiseSeparableConv 的推理形态。

    原结构是 conv_dw -> BN+ReLU -> conv_pw -> BN(无激活) -> (+skip if in==out)。
    两个 BN 在导出期已精确折进各自前面的 conv（那两个 conv 原本无 bias），
    所以这里只剩两次带 bias 的卷积和一个 ReLU。
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        self.conv_dw = nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch)
        self.conv_pw = nn.Conv2d(in_ch, out_ch, 1)
        self.has_skip = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_pw(F.relu(self.conv_dw(x)))
        return h + x if self.has_skip else h


class LayerNormC(nn.Module):
    """对 NCHW 的通道维做 LayerNorm。

    channels_last 下 permute(0,2,3,1) 是零拷贝连续视图，可直接喂 F.layer_norm——
    这也是 UNet 默认走 channels_last 的理由之一。
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.empty(dim))
        self.normalized_shape = (dim,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.permute(0, 2, 3, 1)
        v = F.layer_norm(v, self.normalized_shape, self.weight, self.bias, self.eps)
        return v.permute(0, 3, 1, 2)


class SelfLambda(nn.Module):
    """LλMI 的自注意力：多查询 λ 层。

    把 K/V 压成固定大小的 (dim_k, dim_v) 矩阵，复杂度 O(N·dk·dv)，没有 N^2 项。
    norm_q / norm_v 两个 BatchNorm 在导出期已折进 to_q / to_v。

    pos_conv 原为 Conv3d(1, dim_k, (1,15,15))：深度维核长为 1，即对 V 的每个通道切片
    独立做一次 2D 卷积。这里等价改写成 Conv2d 作用在 (B*dim_v, 1, H, W) 上，权重只是
    squeeze 掉那个恒为 1 的维度。改写的理由不是速度（开 cudnn.benchmark 后两者打平），
    而是 rank-5 权重会让 model.to(memory_format=channels_last) 直接抛 RuntimeError。
    """

    def __init__(self, dim: int, dim_k: int, heads: int, r: int = 15) -> None:
        super().__init__()
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim // heads
        self.to_q = nn.Conv2d(dim, dim_k * heads, 1)
        self.to_k = nn.Conv2d(dim, dim_k, 1, bias=False)
        self.to_v = nn.Conv2d(dim, self.dim_v, 1)
        self.pos_conv = nn.Conv2d(1, dim_k, r, padding=r // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n, dk, dv = h * w, self.dim_k, self.dim_v
        q = self.to_q(x).reshape(b, self.heads, dk, n)
        k = self.to_k(x).reshape(b, dk, n).softmax(dim=-1)
        v = self.to_v(x)
        lam_c = torch.einsum("bkm,bvm->bkv", k, v.reshape(b, dv, n))
        y = torch.einsum("bhkn,bkv->bhvn", q, lam_c)
        # λp：逐 V 通道切片的局部位置核
        lam_p = self.pos_conv(v.reshape(b * dv, 1, h, w)).reshape(b, dv, dk, n)
        y = y + torch.einsum("bhkn,bvkn->bhvn", q, lam_p)
        return y.reshape(b, self.heads * dv, h, w)


class CrossLambda(nn.Module):
    """LλMI 的交叉注意力。

    条件是一张常量 ID 表（Moebius 没有 text encoder），所以整条 K/V 支路都是常量：
    lambda_c 与 v_const 在导出期算好，to_k / to_v / norm_v / encoder_hid_proj /
    embedding_layer 全部从权重文件里消失。运行时只剩 Q 的投影和两次小 matmul。

    Yp 走重排式：先把 Q 与 rel_pos_emb 在 dim_k 上收缩，再与 v_const 在 m 上收缩。
    乘加数从 (n·m·k·v + h·n·k·v) 降到 2·h·n·k·m，且不物化 (b, n, k, v) 的中间量。
    原实现里的 rel_pos_emb[n, m] 是 meshgrid 恒等索引，等于参数本身，gather 已删除。
    """

    def __init__(self, dim: int, dim_k: int, heads: int, n: int,
                 m: int = NUM_COND_TOKENS) -> None:
        super().__init__()
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim // heads
        self.to_q = nn.Conv2d(dim, dim_k * heads, 1)
        self.register_buffer("rel_pos_emb", torch.empty(n, m, dim_k))
        self.register_buffer("lambda_c", torch.empty(dim_k, self.dim_v))
        self.register_buffer("v_const", torch.empty(self.dim_v, m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.to_q(x).reshape(b, self.heads, self.dim_k, h * w)
        y = torch.einsum("bhkn,kv->bhvn", q, self.lambda_c)
        a = torch.einsum("bhkn,nmk->bhnm", q, self.rel_pos_emb)
        y = y + torch.einsum("bhnm,vm->bhvn", a, self.v_const)
        return y.reshape(b, self.heads * self.dim_v, h, w)


class GLUMBConv(nn.Module):
    """SANA 的 MixFFN，替代 BasicTransformerBlock 的 FeedForward。

    原实现把三个卷积裹在 ConvLayer 里，但 norm 全是 None、act 只有 inverted_conv 的
    SiLU 生效，所以这里直接用裸 Conv2d（point_conv 无 bias）。
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.inverted_conv = nn.Conv2d(dim, hidden * 2, 1)
        self.depth_conv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.point_conv = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depth_conv(F.silu(self.inverted_conv(x)))
        return self.point_conv(silu_glu(h))


class MixTransformerBlock(nn.Module):
    """LλMI block：self-λ + cross-λ + MixFFN，三段都是 pre-norm 残差。"""

    def __init__(self, dim: int, heads: int, dim_k: int, n: int,
                 mlp_ratio: float = 2.5) -> None:
        super().__init__()
        self.norm1 = LayerNormC(dim)
        self.attn1 = SelfLambda(dim, dim_k, heads)
        self.norm2 = LayerNormC(dim)
        self.attn2 = CrossLambda(dim, dim_k, heads, n)
        self.norm3 = LayerNormC(dim)
        self.ff = GLUMBConv(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        return x + self.ff(self.norm3(x))


class MixTransformer2DModel(nn.Module):
    """UNet 里的 transformer 包装：GroupNorm -> 1x1 proj_in -> block -> 1x1 proj_out -> 残差。

    原实现在 proj_in 之后 flatten 成 (B, N, C)，但内部每个子层第一件事又都 reshape 回
    NCHW，只有三个 LayerNorm 真需要通道在最后一维。这里全程保持 NCHW，省掉每个 block
    约 6 次全尺寸拷贝。
    """

    def __init__(self, dim: int, heads: int, n: int, mlp_ratio: float = 2.5) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.transformer_blocks = nn.ModuleList(
            [MixTransformerBlock(dim, heads, dim // heads, n, mlp_ratio)])
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(gn_act(self.norm, x, act=None))
        for block in self.transformer_blocks:
            h = block(h)
        return self.proj_out(h) + x


class DWResnetBlock2D(nn.Module):
    """UNet 的 resnet：3x3 conv 全部换成深度可分离卷积。eps 恒为 UNet 的 norm_eps=1e-5。"""

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int = 1280,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=eps)
        self.conv1 = DWConv2d(in_ch, out_ch)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=eps)
        self.conv2 = DWConv2d(out_ch, out_ch)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        # temb 传进来时已经过 SiLU（整网只算一次）；onestep 模式下它被折进
        # conv1.conv_pw.bias，此处传 None
        h = self.conv1(gn_act(self.norm1, x))
        if temb is not None:
            h = h + self.time_emb_proj(temb)[:, :, None, None]
        h = self.conv2(gn_act(self.norm2, h))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class DWMixTFDownBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, heads: int = 8,
                 add_downsample: bool = True) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([DWResnetBlock2D(in_ch, out_ch),
                                      DWResnetBlock2D(out_ch, out_ch)])
        self.attentions = nn.ModuleList(
            [MixTransformer2DModel(out_ch, heads, n) for _ in range(2)])
        self.downsamplers = nn.ModuleList(
            [Downsample2D(out_ch, padding=1)]) if add_downsample else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor],
                skips: List[torch.Tensor]) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            x = attn(resnet(x, temb))
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x


class DWMixTFUpBlock2D(nn.Module):
    """skip 通道遵循 diffusers 的约定：最后一个 resnet 吃下一 stage 的通道数。"""

    def __init__(self, out_ch: int, prev_ch: int, in_ch: int, n: int, heads: int = 8,
                 add_upsample: bool = True) -> None:
        super().__init__()
        resnets = []
        for i in range(3):
            skip_ch = in_ch if i == 2 else out_ch
            res_in = prev_ch if i == 0 else out_ch
            resnets.append(DWResnetBlock2D(res_in + skip_ch, out_ch))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(
            [MixTransformer2DModel(out_ch, heads, n) for _ in range(3)])
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor],
                skips: List[torch.Tensor]) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            x = torch.cat([x, skips.pop()], dim=1)
            x = attn(resnet(x, temb))
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


# ---------------------------------------------------------------------------
# Section 5: Moebius UNet
# ---------------------------------------------------------------------------

class MoebiusUNetLite(nn.Module):
    """Moebius 学生 UNet（226M）：3 个 stage、没有 mid_block、全部注意力换成 λ 层。

    输入 [B, 9, 64, 64] = 4(noisy latent) + 1(mask) + 4(masked latent)，输出 [B, 4, 64, 64]
    的 ε 预测。latent 尺寸恒为 64——见文件头的分辨率约束。
    """

    CH = (320, 640, 1280)

    def __init__(self) -> None:
        super().__init__()
        c0, c1, c2 = self.CH
        n0 = LATENT_SIZE ** 2
        n1 = (LATENT_SIZE // 2) ** 2
        n2 = (LATENT_SIZE // 4) ** 2
        self.conv_in = DWConv2d(9, c0)
        self.time_embedding = TimestepEmbedding(c0, 1280)
        self.down_blocks = nn.ModuleList([
            DWMixTFDownBlock2D(c0, c0, n0),
            DWMixTFDownBlock2D(c0, c1, n1),
            DWMixTFDownBlock2D(c1, c2, n2, add_downsample=False),
        ])
        self.up_blocks = nn.ModuleList([
            DWMixTFUpBlock2D(c2, c2, c1, n2),
            DWMixTFUpBlock2D(c1, c2, c0, n1),
            DWMixTFUpBlock2D(c0, c1, c0, n0, add_upsample=False),
        ])
        self.conv_norm_out = nn.GroupNorm(32, c0, eps=1e-5)
        self.conv_out = DWConv2d(c0, 4)

    def forward(self, sample: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_in(sample)
        skips = [x]
        for block in self.down_blocks:
            x = block(x, temb, skips)
        for block in self.up_blocks:
            x = block(x, temb, skips)
        return self.conv_out(gn_act(self.conv_norm_out, x))


# ---------------------------------------------------------------------------
# Section 6: 扩散调度（scaled_linear beta schedule，ε 预测）
# ---------------------------------------------------------------------------

def make_alphas_cumprod(num_steps: int = 1000, beta_start: float = 0.00085,
                        beta_end: float = 0.012) -> torch.Tensor:
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps,
                           dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def ddpm_pred_x0(sample: torch.Tensor, eps: torch.Tensor,
                 alpha_prod_t: float) -> torch.Tensor:
    """等价 DDPMScheduler.step(...).pred_original_sample（ε 预测、无 clip/threshold）。"""
    return (sample - (1.0 - alpha_prod_t) ** 0.5 * eps) / alpha_prod_t ** 0.5


def ddim_timesteps(t_start: int, steps: int) -> List[int]:
    """DDIM 的 leading spacing，从 t_start 均匀取 steps 个（含 t_start，降序）。"""
    if steps <= 1:
        return [t_start]
    stride = t_start / float(steps)
    return [int(round(t_start - i * stride)) for i in range(steps)]


# ---------------------------------------------------------------------------
# Section 7: 滑窗分块、高斯融合与 wavelet 颜色校正
# ---------------------------------------------------------------------------

def _gaussian_probs(n: int, midpoint: float) -> torch.Tensor:
    var = 0.01
    i = torch.arange(n, dtype=torch.float64)
    return torch.exp(-(i - midpoint) ** 2 / (n * n) / (2 * var)) / math.sqrt(2 * math.pi * var)


def gaussian_weights(size: int) -> torch.Tensor:
    """峰值 15.9、角落 2.3e-10，跨 11 个数量级——所以下游必须用 fp32 累加。"""
    return torch.outer(_gaussian_probs(size, size / 2.0),
                       _gaussian_probs(size, (size - 1) / 2.0))


def coverage_count(h: int, w: int, starts_h: List[int], starts_w: List[int],
                   size: int) -> torch.Tensor:
    """count = Σ_tiles outer(y, x) = outer(Σ shift(y), Σ shift(x))，滑窗是笛卡尔积可分离。

    probs 先舍入到 fp32 再升 fp64，与逐 tile 用 fp32 weights 累加的结果对齐；
    否则分子丢了角落贡献而分母没丢，tile 角落会系统性偏暗。
    """
    y = _gaussian_probs(size, size / 2.0).float().double()
    x = _gaussian_probs(size, (size - 1) / 2.0).float().double()
    cov_y = torch.zeros(h, dtype=torch.float64)
    cov_x = torch.zeros(w, dtype=torch.float64)
    for s in starts_h:
        cov_y[s:s + size] += y
    for s in starts_w:
        cov_x[s:s + size] += x
    return torch.outer(cov_y, cov_x)


def sliding_windows(h: int, w: int, tile_size: int,
                    tile_stride: int) -> List[Tuple[int, int, int, int]]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    return [(hi, hi + tile_size, wi, wi + tile_size) for hi in hi_list for wi in wi_list]


def wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = torch.tensor([[0.0625, 0.125, 0.0625],
                           [0.125, 0.25, 0.125],
                           [0.0625, 0.125, 0.0625]], dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def wavelet_lowpass(image: torch.Tensor, levels: int = 5) -> torch.Tensor:
    for i in range(levels):
        image = wavelet_blur(image, 2 ** i)
    return image


def wavelet_reconstruction(content_feat: torch.Tensor,
                           style_feat: torch.Tensor) -> torch.Tensor:
    """逐级分解的 telescoping 等价形式：高频 = 原图 − 低通。"""
    return content_feat - wavelet_lowpass(content_feat) + wavelet_lowpass(style_feat)


# ---------------------------------------------------------------------------
# Section 8: Runner（赛题入口，签名对齐官方样例）
# ---------------------------------------------------------------------------

def build_model(cls, sd: Dict[str, torch.Tensor], dtype: torch.dtype,
                device: torch.device) -> nn.Module:
    with torch.device("meta"):
        model = cls()
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.config = {**DEFAULTS, **_load_config(model_dir)}

        # WARNING: All models must be using fp16 precision for fair comparisons
        self.weight_dtype = torch.float16

        cfg = self.config
        self.patch_size = cfg["patch_size"]
        self.stride = cfg["stride"]
        self.seed = cfg["seed"]
        self.mode = cfg["mode"]
        self.model_t = cfg["model_t"]
        assert self.mode in ("sdedit", "onestep"), self.mode
        assert self.patch_size == LATENT_SIZE * 8, "cross-λ 的 rel_pos_emb 与 512x512 死绑"

        sd = torch.load(os.path.join(model_dir, cfg["weights"]), map_location="cpu",
                        weights_only=True)
        self.vae = build_model(AutoencoderKLLite, sd["vae"], self.weight_dtype, self.device)
        self.unet = build_model(MoebiusUNetLite, sd["unet"], self.weight_dtype, self.device)
        del sd

        torch.backends.cudnn.benchmark = True  # 形状恒定，让 cuDNN 选最优算法

        alphas = make_alphas_cumprod()
        with torch.no_grad():
            self._setup_schedule(alphas, cfg)
            # QKV 融合：VAE attention 的 q/k/v 读同一输入，预 cat 成单 GEMM
            for m in self.vae.modules():
                if isinstance(m, VAEAttention):
                    m._qkv_w = torch.cat([m.to_q.weight, m.to_k.weight, m.to_v.weight])
                    m._qkv_b = torch.cat([m.to_q.bias, m.to_k.bias, m.to_v.bias])
            self._mask = torch.full((1, 1, LATENT_SIZE, LATENT_SIZE), float(cfg["mask_value"]),
                                    device=self.device, dtype=self.weight_dtype)

        # fork_rng 隔离预热消耗的 randn，保证后续输出与不预热时一致
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(self.seed)
            shape = (1, 4, LATENT_SIZE, LATENT_SIZE)
            # 热路径的两处噪声常驻：形状恒定、逐 tile 相同，infer 因此完全无 RNG
            self._enc_noise = torch.randn(shape, device=self.device, dtype=self.weight_dtype)
            self._init_noise = torch.randn(shape, device=self.device, dtype=self.weight_dtype)
            self._set_memory_format(cfg["unet_channels_last"], cfg["vae_channels_last"])
            self._tile_weights = gaussian_weights(self.patch_size).to(
                self.device, torch.float32)[None, None]
            self._warmup()

    def _setup_schedule(self, alphas: torch.Tensor, cfg: dict) -> None:
        """把 timestep 相关的量在加载期全部算成常量。

        onestep：timestep 恒定，各 resnet 的 time_emb_proj 输出是逐通道常量，语义与 bias
        相同，折进 conv1.conv_pw.bias 后整条 timestep 通路从前向删除。
        sdedit：每步一组常量，预计算成 temb 表，逐步取用（那 15 个 add 在 latent 分辨率，
        合计不到 0.1 ms，不值得为多步再做折叠）。
        """
        def temb_of(t: int) -> torch.Tensor:
            tt = torch.full((1,), t, dtype=torch.long, device=self.device)
            emb = timestep_embedding(tt).to(self.weight_dtype)
            return F.silu(self.unet.time_embedding(emb))

        if self.mode == "onestep":
            self.alpha_prod_t = float(alphas[self.model_t])
            temb = temb_of(self.model_t)
            for m in self.unet.modules():
                if isinstance(m, DWResnetBlock2D):
                    m.conv1.conv_pw.bias.add_(m.time_emb_proj(temb).flatten())
            self._temb_table = None
            self._schedule = None
            return

        ts = ddim_timesteps(cfg["t_start"], cfg["steps"])
        self._temb_table = [temb_of(t) for t in ts]
        # 每步的 (alpha_t, alpha_prev)；最后一步的 prev 用 alpha_0 之前的 1.0
        self._schedule = []
        for i, t in enumerate(ts):
            a_t = float(alphas[t])
            a_prev = float(alphas[ts[i + 1]]) if i + 1 < len(ts) else 1.0
            self._schedule.append((a_t, a_prev))
        self._a_start = float(alphas[ts[0]])

    def _set_memory_format(self, unet_cl: bool, vae_cl: bool) -> None:
        """UNet 与 VAE 分开选布局。

        UNet 走 channels_last：LayerNormC 的 permute 在 NHWC 下是零拷贝视图。
        VAE 走 NCHW：纯 PyTorch 下实测更快（差距全在 VAE）。
        中间只需在 (1,4,64,64) 的 latent 上转一次布局，32 KB，代价可忽略。
        """
        self._unet_mf = torch.channels_last if unet_cl else torch.contiguous_format
        self._vae_mf = torch.channels_last if vae_cl else torch.contiguous_format
        with torch.no_grad():
            for model, cl, mf in ((self.unet, unet_cl, self._unet_mf),
                                  (self.vae, vae_cl, self._vae_mf)):
                model.to(memory_format=mf)
                # 上采样折叠的权重布局跟随模型，换布局要重折一次
                for m in model.modules():
                    if isinstance(m, Upsample2D):
                        m._convt = fold_upsample_conv(m.conv, cl)

    def _warmup(self, iters: int = 3) -> float:
        """按当前布局跑热路径，返回后半程的平均墙钟秒数（顺带完成 cuDNN 试跑）。"""
        x = torch.zeros(1, 3, self.patch_size, self.patch_size,
                        dtype=self.weight_dtype, device=self.device)
        x = x.contiguous(memory_format=self._vae_mf)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            for _ in range(iters):
                self._infer_tile(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                self._infer_tile(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    @staticmethod
    def _resize_at_least(imgs: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = imgs.size()
        if h == w:
            new_h, new_w = size, size
        elif h < w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size
        return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)

    def _denoise(self, z_lq: torch.Tensor) -> torch.Tensor:
        """latent 去噪。z_lq 已乘 scaling_factor，[1,4,64,64] weight_dtype -> 同形状。"""
        if self.mode == "onestep":
            eps = self.unet(torch.cat([z_lq, self._mask, z_lq], dim=1))
            return ddpm_pred_x0(z_lq, eps, self.alpha_prod_t)

        z = self._a_start ** 0.5 * z_lq + (1.0 - self._a_start) ** 0.5 * self._init_noise
        for i, (a_t, a_prev) in enumerate(self._schedule):
            eps = self.unet(torch.cat([z, self._mask, z_lq], dim=1), self._temb_table[i])
            x0 = (z - (1.0 - a_t) ** 0.5 * eps) / a_t ** 0.5
            z = a_prev ** 0.5 * x0 + (1.0 - a_prev) ** 0.5 * eps
        return z

    @torch.no_grad()
    def _infer_tile(self, x: torch.Tensor) -> torch.Tensor:
        """单 tile 模型链：VAE encode -> 重参数化 -> 去噪 -> VAE decode。

        x: [1, 3, patch, patch] weight_dtype -> [1, 3, patch, patch] fp32。
        全程无 RNG、无形状分支，输入形状恒定。
        """
        z = sample_latent(self.vae.encode_moments(x), self._enc_noise) * VAE_SCALING_FACTOR
        z = self._denoise(z.to(self.weight_dtype).contiguous(memory_format=self._unet_mf))
        z = (z / VAE_SCALING_FACTOR).to(self.weight_dtype).contiguous(memory_format=self._vae_mf)
        return self.vae.decode(z).float()

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """赛题评测入口：单个 patch_size x patch_size 分块的推理。

        Args:
            image_tensor: [1, 3, patch, patch] tensor in [-1, 1] range。
            prompt: 仅为对齐赛题接口签名，不参与计算——Moebius 没有 text encoder，
                条件 embedding 在导出期就已被吸收成 cross-λ 的常量。

        Returns:
            output_tensor: 同形状 tensor in [-1, 1] range。

        整图请走 enhance()；这里对非分块尺寸的输入兜底转发过去。
        """
        if tuple(image_tensor.shape[-2:]) != (self.patch_size, self.patch_size):
            return self.enhance(image_tensor)

        ref = image_tensor.to(device=self.device, dtype=torch.float32)
        x = ref.to(self.weight_dtype).contiguous(memory_format=self._vae_mf)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            x = self._infer_tile(x)
        x = x.contiguous()  # wavelet 的 replicate pad 需要 NCHW
        return wavelet_reconstruction(x, ref).clamp(-1, 1)

    @torch.no_grad()
    def enhance(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """整图入口：短边保护 -> 像素域滑窗 -> 逐 tile 调 infer -> 高斯融合。

        Args:
            image_tensor: [1, 3, H, W] tensor in [-1, 1] range.

        Returns:
            output_tensor: [1, 3, H, W] tensor in [-1, 1] range.

        每个 tile 恒为 patch_size x patch_size，故不需要 pad 到 8 的倍数——滑窗的末块
        自动贴边。颜色校正已在 infer 内逐 tile 做过（局部低频对齐局部 LQ），这里不再做
        整图 wavelet。
        """
        assert len(image_tensor) == 1, "整图路径按 batch=1 设计"
        size, stride = self.patch_size, self.stride
        x = image_tensor.to(device=self.device, dtype=torch.float32)
        h0, w0 = x.shape[2:]

        # 短边不足一个 tile 时先放大。用严格小于：恰好等于 patch_size 时已够铺满一个
        # tile，再插值一次是纯浪费（antialias 的 bicubic 同尺寸也不是恒等变换）
        if min(h0, w0) < size:
            x = self._resize_at_least(x, size=size)
        h1, w1 = x.shape[2:]

        indices = sliding_windows(h1, w1, size, stride)
        if len(indices) == 1:
            out = self.infer(x)
        else:
            # 融合缓冲恒用 fp32：高斯权重跨 11 个数量级，fp16 下角落会下溢为 0，
            # 而 count 在 fp64 下算出、包含这些贡献，二者不一致会让 tile 角落系统性偏暗
            out = torch.zeros((1, 3, h1, w1), dtype=torch.float32, device=self.device)
            weights = self._tile_weights
            starts_h = sorted({hi for hi, _, _, _ in indices})
            starts_w = sorted({wi for _, _, wi, _ in indices})
            count = coverage_count(h1, w1, starts_h, starts_w, size).to(
                self.device, torch.float32)[None, None]
            for hi, hi_end, wi, wi_end in indices:
                tile = self.infer(x[..., hi:hi_end, wi:wi_end])
                out[..., hi:hi_end, wi:wi_end] += tile * weights
            out /= count

        if out.shape[2:] != (h0, w0):
            out = F.interpolate(out, size=(h0, w0), mode="bicubic", antialias=True)
        return out.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Section 9: --export 权重打包
#
# 读原始的 Moebius UNet + SDXL VAE checkpoint，在 float64 下折叠 BatchNorm、预计算
# cross-λ 常量，转 fp16 存成单文件。
# ---------------------------------------------------------------------------

_BN_EPS = 1e-5  # timm BatchNormAct2d 的默认 eps


def fold_bn(w: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mean: torch.Tensor,
            var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 BatchNorm 折进前面那个无 bias 的 conv/linear。在 float64 下算。

    y = γ·(Wx − μ)/√(σ²+ε) + β = (W·s)x + (β − μ·s)，  s = γ/√(σ²+ε)
    """
    s = gamma.double() / (var.double() + _BN_EPS).sqrt()
    shape = (-1,) + (1,) * (w.dim() - 1)
    return w.double() * s.reshape(shape), beta.double() - mean.double() * s


def _export_unet(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    src = {k[len("diff_model."):]: v for k, v in raw.items() if k.startswith("diff_model.")}
    out: Dict[str, torch.Tensor] = {}
    used = set()

    def take(key: str) -> torch.Tensor:
        used.add(key)
        return src[key]

    def bn(prefix: str) -> Tuple[torch.Tensor, ...]:
        return tuple(take("%s.%s" % (prefix, s))
                     for s in ("weight", "bias", "running_mean", "running_var"))

    # 1) DWConv2d：两个 BN 分别折进 conv_dw / conv_pw
    for key in [k for k in src if k.endswith(".conv_dw.weight")]:
        p = key[:-len(".conv_dw.weight")]
        out["%s.conv_dw.weight" % p], out["%s.conv_dw.bias" % p] = fold_bn(
            take("%s.conv_dw.weight" % p), *bn("%s.bn1" % p))
        out["%s.conv_pw.weight" % p], out["%s.conv_pw.bias" % p] = fold_bn(
            take("%s.conv_pw.weight" % p), *bn("%s.bn2" % p))

    # 2) self-λ：norm_q/norm_v 折进 to_q/to_v；to_k 原本就无 norm 无 bias；
    #    pos_conv 的 Conv3d 权重 (k,1,1,r,r) squeeze 成 Conv2d 的 (k,1,r,r)
    for key in [k for k in src if k.endswith(".attn1.to_q.weight")]:
        p = key[:-len(".to_q.weight")]
        out["%s.to_q.weight" % p], out["%s.to_q.bias" % p] = fold_bn(
            take("%s.to_q.weight" % p), *bn("%s.norm_q" % p))
        out["%s.to_v.weight" % p], out["%s.to_v.bias" % p] = fold_bn(
            take("%s.to_v.weight" % p), *bn("%s.norm_v" % p))
        out["%s.to_k.weight" % p] = take("%s.to_k.weight" % p).double()
        out["%s.pos_conv.weight" % p] = take("%s.pos_conv.weight" % p).double().squeeze(2)
        out["%s.pos_conv.bias" % p] = take("%s.pos_conv.bias" % p).double()

    # 3) cross-λ：K/V 支路整条是常量，预计算成 lambda_c 与 v_const
    #    cond 用 background 的 id（前 10 个），与 pipeline 的 input_ids 一致
    emb = raw["embedding_layer.weight"][:NUM_COND_TOKENS].double()
    hidden = F.linear(emb, take("encoder_hid_proj.weight").double(),
                      take("encoder_hid_proj.bias").double())  # (10, 768)
    for key in [k for k in src if k.endswith(".attn2.to_q.weight")]:
        p = key[:-len(".to_q.weight")]
        out["%s.to_q.weight" % p], out["%s.to_q.bias" % p] = fold_bn(
            take("%s.to_q.weight" % p), *bn("%s.norm_q" % p))
        kk = F.linear(hidden, take("%s.to_k.weight" % p).double()).t()   # (dk, 10)
        vv = F.linear(hidden, take("%s.to_v.weight" % p).double()).t()   # (dv, 10)
        g, b, m, v = bn("%s.norm_v" % p)
        s = g.double() / (v.double() + _BN_EPS).sqrt()
        v_const = vv * s[:, None] + (b.double() - m.double() * s)[:, None]
        out["%s.lambda_c" % p] = kk.softmax(dim=-1) @ v_const.t()        # (dk, dv)
        out["%s.v_const" % p] = v_const
        # rel_pos_emb 的最后一维 dim_u 恒为 1；原实现的 rel_pos_emb[n,m] 是恒等索引
        out["%s.rel_pos_emb" % p] = take("%s.rel_pos_emb" % p).double().squeeze(-1)

    # 4) GLUMBConv：拆掉 ConvLayer 的 .conv 包装
    for key in [k for k in src if ".ff." in k and ".conv." in k]:
        out[key.replace(".conv.", ".")] = take(key).double()

    # 5) 其余直通（norm / proj_in / proj_out / time_emb_proj / conv_shortcut /
    #    downsamplers / upsamplers / conv_norm_out / time_embedding）
    for key in src:
        if key in used or key.endswith("num_batches_tracked"):
            continue
        out[key] = src[key].double()
        used.add(key)

    dropped = {k for k in src if k.endswith("num_batches_tracked")}
    assert used | dropped == set(src), "源 key 未被完全消费：%s" % sorted(set(src) - used - dropped)
    return out


def export_weights(src_root: str, dst_dir: str, which: str = "pretrained",
                   dtype: torch.dtype = torch.float16) -> None:
    unet_path = os.path.join(src_root, "Moebius", which, "diffusion_pytorch_model.bin")
    vae_path = os.path.join(src_root, "PixelHacker", "vae", "diffusion_pytorch_model.bin")
    raw = torch.load(unet_path, map_location="cpu", weights_only=True)
    vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)

    unet_sd = _export_unet(raw)
    for name, tensor in unet_sd.items():
        peak = tensor.abs().max().item()
        assert peak < 1e4, "fp16 溢出风险：%s 峰值 %.3g" % (name, peak)

    # strict 加载是最终断言：任何 missing / unexpected / shape mismatch 都会在这里炸
    with torch.device("meta"):
        MoebiusUNetLite().load_state_dict(
            {k: v.to(dtype) for k, v in unet_sd.items()}, strict=True, assign=True)
        AutoencoderKLLite().load_state_dict(
            {k: v.to(dtype) for k, v in vae_sd.items()}, strict=True, assign=True)

    out = {"unet": {k: v.to(dtype) for k, v in unet_sd.items()},
           "vae": {k: v.to(dtype) for k, v in vae_sd.items()}}
    dst = os.path.join(dst_dir, WEIGHTS_FILE)
    torch.save(out, dst)
    n_unet = sum(v.numel() for v in out["unet"].values())
    n_vae = sum(v.numel() for v in out["vae"].values())
    print("UNet %.2fM + VAE %.2fM = %.2fM 参数 -> %s (%.0f MB)"
          % (n_unet / 1e6, n_vae / 1e6, (n_unet + n_vae) / 1e6, dst,
             os.path.getsize(dst) / 1e6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--src", default=".", help="Moebius 仓库根（含 Moebius/ 与 PixelHacker/）")
    parser.add_argument("--dst", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--which", default="pretrained",
                        choices=["pretrained", "ft_places2", "ft_celebahq", "ft_ffhq"])
    args = parser.parse_args()
    if args.export:
        export_weights(args.src, args.dst, args.which)
    else:
        parser.error("目前只支持 --export")


if __name__ == "__main__":
    main()
