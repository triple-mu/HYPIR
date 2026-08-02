"""attn_d512：对 fp32 SDPA 参考的正确性、strided(融合 QKV) 布局、与 Triton 的一致性、确定性。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import CUSTOM, OPS, requires_cuda, requires_custom, requires_triton

NS = [64, 256, 1024, 4096]  # 4096 是生产形状（512x512 输入下的 VAE mid-block）


def _ref(q, k, v):
    return F.scaled_dot_product_attention(q[:, None].float(), k[:, None].float(), v[:, None].float())[:, 0]


@requires_cuda
@requires_custom
@pytest.mark.parametrize("n", NS)
def test_matches_fp32_sdpa(n):
    """fp16 输入、fp32 参考：误差应在 fp16 量级（softmax 后值域 [0,1]，故看绝对误差）。"""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, 512, device="cuda", dtype=torch.float16) for _ in range(3))
    got = CUSTOM.attn_d512(q, k, v)
    assert got.shape == q.shape and got.dtype == q.dtype
    assert float((got.float() - _ref(q, k, v)).abs().max()) < 2e-2


@requires_cuda
@requires_custom
def test_strided_fused_qkv_view():
    """生产路径里 q/k/v 是融合 QKV 张量 chunk 出来的视图，行 stride=1536 而非 512。"""
    torch.manual_seed(0)
    qkv = torch.randn(1, 1024, 1536, device="cuda", dtype=torch.float16)
    q, k, v = qkv.chunk(3, dim=-1)
    assert q.stride(1) == 1536 and not q.is_contiguous()
    got = CUSTOM.attn_d512(q, k, v)
    assert float((got.float() - _ref(q, k, v)).abs().max()) < 2e-2


@requires_cuda
@requires_custom
@requires_triton
@pytest.mark.parametrize("n", NS)
def test_agrees_with_triton(n):
    """与 Triton 对拍。

    注意：Triton 的 attn_d512 在 **Volta(sm70)** 上是坏的——它的 tl.dot 在 sm70 上算不对，
    实测输出与 fp32 参考的最大误差达 0.13~0.78（输出本身量级仅 0.03），端到端 PSNR 只有
    27.3 dB。这种情况下 Triton 无法充当对拍基准，故明确跳过并报出实测误差，而不是让
    本测试红着——真正的判据是上面 test_matches_fp32_sdpa 里 custom 对 fp32 参考的一致性。
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, 512, device="cuda", dtype=torch.float16) for _ in range(3))
    ref = _ref(q, k, v)
    got = CUSTOM.attn_d512(q, k, v)
    tri = OPS.triton_attn_d512(q, k, v)
    assert float((got.float() - ref).abs().max()) < 2e-2

    tri_err = float((tri.float() - ref).abs().max())
    if tri_err > 2e-2:
        cap = torch.cuda.get_device_capability()
        pytest.skip(
            "Triton attn_d512 在 sm%d%d 上自身就与 fp32 参考不符（max=%.3e），无法作为对拍基准"
            % (cap[0], cap[1], tri_err)
        )
    # 两条实现的 softmax 归约结构不同（online vs 两遍），差异仍应在 fp16 量级
    assert float((got.float() - tri.float()).abs().max()) < 3e-2


@requires_cuda
@requires_custom
def test_deterministic():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1024, 512, device="cuda", dtype=torch.float16) for _ in range(3))
    assert torch.equal(CUSTOM.attn_d512(q, k, v), CUSTOM.attn_d512(q, k, v))


@requires_cuda
@requires_custom
def test_rejects_wrong_head_dim():
    q, k, v = (torch.randn(1, 64, 256, device="cuda", dtype=torch.float16) for _ in range(3))
    with pytest.raises(RuntimeError, match="head_dim must be 512"):
        CUSTOM.attn_d512(q, k, v)
