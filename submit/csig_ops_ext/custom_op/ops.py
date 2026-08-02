"""custom_op 算子的 Python 封装：register_fake meta kernel + 转发到 torch.ops.custom_op::* 的薄包装。

数学语义、形状与 sm70/sm86 实现细节见 csrc 各 .cu 文件顶部；本文件只做注册与转发。
签名与 model_dir/runner.py 里被替换的 triton_* 函数保持一致，便于直接对拍与 monkeypatch。
"""

from __future__ import annotations

from typing import Optional

import torch


# ---- register_fake：meta kernel，供 torch.compile / FakeTensor 推断输出形状 ----
@torch.library.register_fake("custom_op::group_norm_act")
def _fake_group_norm_act(x, weight, bias, num_groups, eps, act):
    return torch.empty_like(x)


@torch.library.register_fake("custom_op::geglu")
def _fake_geglu(y2):
    return y2.new_empty(y2.shape[:-1] + (y2.shape[-1] // 2,))


@torch.library.register_fake("custom_op::sample_latent")
def _fake_sample_latent(moments, noise):
    n, c8, h, w = moments.shape
    return moments.new_empty((n, c8 // 2, h, w))


@torch.library.register_fake("custom_op::attn_d512")
def _fake_attn_d512(q, k, v):
    return torch.empty_like(q)


# ---- 薄包装：签名对齐 runner.py 的 triton_* ----
def group_norm_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    act: Optional[str],
) -> torch.Tensor:
    """act 传字符串以对齐 triton_group_norm_act 的签名："silu" 或 None。"""
    return torch.ops.custom_op.group_norm_act(x, weight, bias, num_groups, eps, 1 if act == "silu" else 0)


def geglu(y2: torch.Tensor) -> torch.Tensor:
    return torch.ops.custom_op.geglu(y2)


def sample_latent(moments: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return torch.ops.custom_op.sample_latent(moments, noise)


def attn_d512(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.ops.custom_op.attn_d512(q, k, v)
