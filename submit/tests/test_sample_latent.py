"""sample_latent：对 eager 参考的正确性、布局约定、与 Triton 的一致性、确定性。

布局是这个算子最容易写错的地方：moments 是 NHWC，noise 是 NCHW（保持 torch.randn 的原始
布局，使随机数流与 eager 一致），输出 NHWC。所以专门测了「换成错误布局必须报错」。
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    CUSTOM,
    DTYPES,
    OPS,
    rel_err,
    requires_cuda,
    requires_custom,
    requires_triton,
)

SHAPES = [(1, 64, 64), (1, 32, 32), (1, 96, 128), (2, 64, 64)]


def _make(n, h, w, dtype):
    mom = torch.randn(n, 8, h, w, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last)
    nz = torch.randn(n, 4, h, w, device="cuda", dtype=dtype)  # NCHW，不转 channels_last
    return mom, nz


def _ref(mom, nz):
    mean, logvar = mom.float().chunk(2, dim=1)
    return mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * nz.float()


@requires_cuda
@requires_custom
@pytest.mark.parametrize("n,h,w", SHAPES)
def test_matches_fp32_reference(n, h, w):
    torch.manual_seed(0)
    mom, nz = _make(n, h, w, torch.float32)
    got = CUSTOM.sample_latent(mom, nz)
    assert got.shape == (n, 4, h, w)
    assert got.is_contiguous(memory_format=torch.channels_last)
    assert rel_err(got, _ref(mom, nz)) < 1e-5


@requires_cuda
@requires_custom
def test_logvar_clamped():
    """logvar 必须被 clamp 到 [-30, 20]，否则 exp 会溢出成 inf。"""
    torch.manual_seed(0)
    mom = torch.zeros(1, 8, 8, 8, device="cuda", dtype=torch.float32)
    mom[:, 4:] = 1e4  # 远超上界
    mom = mom.contiguous(memory_format=torch.channels_last)
    nz = torch.ones(1, 4, 8, 8, device="cuda", dtype=torch.float32)
    got = CUSTOM.sample_latent(mom, nz)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.full_like(got, float(torch.exp(torch.tensor(10.0)))), rtol=1e-5)


@requires_cuda
@requires_custom
@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype(dtype):
    torch.manual_seed(0)
    mom, nz = _make(1, 64, 64, dtype)
    got = CUSTOM.sample_latent(mom, nz)
    assert got.dtype == dtype and torch.isfinite(got).all()


@requires_cuda
@requires_custom
@requires_triton
@pytest.mark.parametrize("n,h,w", SHAPES)
def test_agrees_with_triton(n, h, w):
    torch.manual_seed(0)
    mom, nz = _make(n, h, w, torch.float16)
    ref = _ref(mom, nz)
    got = CUSTOM.sample_latent(mom, nz)
    tri = OPS.triton_sample_latent(mom, nz)
    assert rel_err(got, ref) < 2e-3
    assert rel_err(tri, ref) < 2e-3
    assert rel_err(got, tri) < 2e-3


@requires_cuda
@requires_custom
def test_rejects_wrong_layout():
    torch.manual_seed(0)
    mom, nz = _make(1, 64, 64, torch.float16)
    with pytest.raises(RuntimeError, match="channels_last"):
        CUSTOM.sample_latent(mom.contiguous(), nz)
    with pytest.raises(RuntimeError, match="8 channels"):
        CUSTOM.sample_latent(mom[:, :4].contiguous(memory_format=torch.channels_last), nz)


@requires_cuda
@requires_custom
def test_deterministic():
    torch.manual_seed(0)
    mom, nz = _make(1, 64, 64, torch.float16)
    assert torch.equal(CUSTOM.sample_latent(mom, nz), CUSTOM.sample_latent(mom, nz))
