"""group_norm_act：对 eager 参考的正确性、与 Triton 的一致性、确定性、参数校验。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import (
    CUSTOM,
    DTYPES,
    OPS,
    gn_shapes,
    make_nhwc,
    rel_err,
    requires_cuda,
    requires_custom,
    requires_triton,
)

GROUPS = 32
EPS = 1e-5


def _ref(x, w, b, act):
    y = F.group_norm(x.float(), GROUPS, w.float(), b.float(), EPS)
    return F.silu(y) if act == "silu" else y


@requires_cuda
@requires_custom
@pytest.mark.parametrize("hw,c", gn_shapes())
@pytest.mark.parametrize("act", ["silu", None])
def test_matches_fp32_reference(hw, c, act):
    """fp32 下与 F.group_norm(+silu) 比：只有累加顺序差异，相对误差应在 1e-5 以内。"""
    torch.manual_seed(0)
    x = make_nhwc(1, c, hw, torch.float32)
    w = torch.randn(c, device="cuda")
    b = torch.randn(c, device="cuda")
    got = CUSTOM.group_norm_act(x, w, b, GROUPS, EPS, act)
    assert rel_err(got, _ref(x, w, b, act)) < 1e-5


@requires_cuda
@requires_custom
@pytest.mark.parametrize("hw,c", gn_shapes()[::4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_and_layout(hw, c, dtype):
    """输出 dtype / 形状 / channels_last 必须与输入一致。"""
    torch.manual_seed(0)
    x = make_nhwc(1, c, hw, dtype)
    w = torch.randn(c, device="cuda", dtype=dtype)
    b = torch.randn(c, device="cuda", dtype=dtype)
    got = CUSTOM.group_norm_act(x, w, b, GROUPS, EPS, "silu")
    assert got.dtype == dtype and got.shape == x.shape
    assert got.is_contiguous(memory_format=torch.channels_last)
    assert torch.isfinite(got).all()


@requires_cuda
@requires_custom
@requires_triton
@pytest.mark.parametrize("hw,c", gn_shapes())
@pytest.mark.parametrize("act", ["silu", None])
def test_agrees_with_triton(hw, c, act):
    """fp16 下 custom_op 与 triton 的差异，不应超过两者各自对 fp32 参考的误差量级。"""
    torch.manual_seed(0)
    x = make_nhwc(1, c, hw, torch.float16)
    w = torch.randn(c, device="cuda", dtype=torch.float16)
    b = torch.randn(c, device="cuda", dtype=torch.float16)
    ref = _ref(x, w, b, act)
    got = CUSTOM.group_norm_act(x, w, b, GROUPS, EPS, act)
    tri = OPS.triton_group_norm_act(x, w, b, GROUPS, EPS, act)
    # fp16 的 1 ulp 相对量级约 1e-3；两条实现都只在写回时舍入一次，故同量级即算对齐。
    assert rel_err(got, ref) < 2e-3
    assert rel_err(tri, ref) < 2e-3
    assert rel_err(got, tri) < 2e-3


@requires_cuda
@requires_custom
def test_deterministic():
    """归约顺序固定 -> 同输入连跑两次必须逐位相同。"""
    torch.manual_seed(0)
    x = make_nhwc(1, 512, 65536, torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, device="cuda", dtype=torch.float16)
    a1 = CUSTOM.group_norm_act(x, w, b, GROUPS, EPS, "silu")
    a2 = CUSTOM.group_norm_act(x, w, b, GROUPS, EPS, "silu")
    assert torch.equal(a1, a2)


@requires_cuda
@requires_custom
def test_rejects_invalid_input():
    """非法输入必须被 TORCH_CHECK 挡住，而不是算出错误结果。"""
    x = make_nhwc(1, 320, 4096, torch.float16)
    w = torch.randn(320, device="cuda", dtype=torch.float16)
    b = torch.randn(320, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="divisible"):  # C 不被 num_groups 整除
        CUSTOM.group_norm_act(x, w, b, 7, EPS, "silu")
    with pytest.raises(RuntimeError, match="channels_last"):  # 非 channels_last
        CUSTOM.group_norm_act(x.contiguous(), w, b, GROUPS, EPS, "silu")
    with pytest.raises(RuntimeError, match="C elements"):  # weight 长度不对
        CUSTOM.group_norm_act(x, w[:16], b, GROUPS, EPS, "silu")
