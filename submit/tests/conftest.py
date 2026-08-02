"""pytest 公共设施：生产形状表、后端可用性、参考实现。

形状直接从 model_dir/csig_ops.py 的静态调优表读，避免测试与实际推理链路脱节。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


# 先导 custom_op 再把 model_dir 放进 sys.path：提交包里 model_dir/custom_op/ 放的是给
# 评测机编的 .so（Python 版本可能与本机不同），会遮住 custom_op_cpp_ext 的可编辑安装。
# 单测要测的是源码构建的那份。
try:
    import custom_op

    CUSTOM = custom_op
except Exception:
    CUSTOM = None

sys.path.insert(0, str(REPO_ROOT / "model_dir"))

try:
    import csig_ops as OPS
except Exception:  # triton 缺失 / 语法不兼容时，Triton 对拍整体跳过
    OPS = None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
requires_custom = pytest.mark.skipif(CUSTOM is None, reason="custom_op 未编译")
HAS_TRITON_OPS = OPS is not None and hasattr(OPS, "triton_group_norm_act")
requires_triton = pytest.mark.skipif(
    not HAS_TRITON_OPS, reason="当前环境没有 triton，无 triton_* 可对拍"
)

# fp16 与 fp32 都测；sm70 无 bf16 tensor core，推理链路也不用，故不测 bf16。
DTYPES = [torch.float16, torch.float32]


def gn_shapes():
    """GroupNorm 的生产形状 (HW, C)：来自 csig_ops.py 的 _GN_STATS_BEST。"""
    tbl = getattr(OPS, "_GN_STATS_BEST", None)
    if tbl is None:
        return [(64, 1280), (256, 640), (1024, 320), (4096, 512), (16384, 256), (65536, 128), (262144, 128)]
    return sorted(tbl.keys())


def geglu_shapes():
    """GEGLU 的生产形状 (rows, D)：_GEGLU_BEST 的键是 (total, D)，total = rows * D。"""
    tbl = getattr(OPS, "_GEGLU_BEST", None)
    if tbl is None:
        return [(4096, 1280), (1024, 2560), (256, 5120), (64, 5120)]
    return sorted((total // d, d) for total, d in tbl.keys())


def make_nhwc(n, c, hw, dtype, device="cuda"):
    """按 (HW, C) 造一张 channels_last 的 4D 张量；HW 尽量摆成接近正方形。"""
    side = int(hw ** 0.5)
    while side > 1 and hw % side:
        side -= 1
    h, w = side, hw // side
    return torch.randn(n, c, h, w, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = float(ref.float().abs().max())
    return float((got.float() - ref.float()).abs().max()) / max(denom, 1e-12)
