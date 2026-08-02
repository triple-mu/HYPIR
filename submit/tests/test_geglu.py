"""geglu：对 eager 参考的正确性、与 Triton 的一致性、确定性、参数校验。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import (
    CUSTOM,
    DTYPES,
    OPS,
    geglu_shapes,
    rel_err,
    requires_cuda,
    requires_custom,
    requires_triton,
)


def _ref(y2):
    h, g = y2.float().chunk(2, dim=-1)
    return h * F.gelu(g)  # F.gelu 默认就是 erf 精确版，与 kernel 一致


@requires_cuda
@requires_custom
@pytest.mark.parametrize("rows,d", geglu_shapes())
def test_matches_fp32_reference(rows, d):
    torch.manual_seed(0)
    y2 = torch.randn(1, rows, 2 * d, device="cuda", dtype=torch.float32)
    assert rel_err(CUSTOM.geglu(y2), _ref(y2)) < 1e-5


@requires_cuda
@requires_custom
@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_and_shape(dtype):
    torch.manual_seed(0)
    y2 = torch.randn(1, 256, 5120 * 2, device="cuda", dtype=dtype)
    got = CUSTOM.geglu(y2)
    assert got.dtype == dtype and got.shape == (1, 256, 5120)
    assert torch.isfinite(got).all()


@requires_cuda
@requires_custom
@requires_triton
@pytest.mark.parametrize("rows,d", geglu_shapes())
def test_agrees_with_triton(rows, d):
    torch.manual_seed(0)
    y2 = torch.randn(1, rows, 2 * d, device="cuda", dtype=torch.float16)
    ref = _ref(y2)
    got = CUSTOM.geglu(y2)
    tri = OPS.triton_geglu(y2)
    assert rel_err(got, ref) < 2e-3
    assert rel_err(tri, ref) < 2e-3
    assert rel_err(got, tri) < 2e-3


@requires_cuda
@requires_custom
def test_non_power_of_two_and_odd_reject():
    """D 不是 8 的倍数时走标量路径；最后一维为奇数必须报错。"""
    torch.manual_seed(0)
    y2 = torch.randn(1, 7, 2 * 13, device="cuda", dtype=torch.float32)  # D=13，非 4/8 倍数
    assert rel_err(CUSTOM.geglu(y2), _ref(y2)) < 1e-5
    with pytest.raises(RuntimeError, match="even"):
        CUSTOM.geglu(torch.randn(1, 4, 7, device="cuda", dtype=torch.float32))


@requires_cuda
@requires_custom
def test_deterministic():
    torch.manual_seed(0)
    y2 = torch.randn(1, 1024, 5120, device="cuda", dtype=torch.float16)
    assert torch.equal(CUSTOM.geglu(y2), CUSTOM.geglu(y2))
