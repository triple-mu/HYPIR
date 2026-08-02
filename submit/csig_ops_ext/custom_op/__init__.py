"""custom_op——HYPIR 推理链路四个融合算子的 CUDA C++ 实现，用于替换 Triton 版并做对拍/基准。

只走 fp16/fp32 推理路径（sm70 无 bf16 tensor core）：group_norm_act、geglu、sample_latent、
attn_d512。算子的 TORCH_LIBRARY 注册由 ``_C`` 触发。
"""

from __future__ import annotations

import glob
import os
import sys

import torch  # 必须先加载 libc10 / libtorch，再 dlopen _C

# 四个算子全部经 TORCH_LIBRARY 注册、无 pybind11 绑定，故 .so 里没有 PyInit__C，
# 不能 `import _C`；用 torch.ops.load_library dlopen 触发注册（纯算子扩展的标准做法）。
# 同一棵源码树被两个 python 编过时会留下多颗 _C*.so，而它们链的 CUDA 运行时可能不同
# （容器系统 python3.10 配 cu12，评测机对齐的 py39 venv 配 cu118）。按解释器 ABI 标签挑，
# 不能直接取排序第一个——字典序下 "310" < "39"，正好每次都挑中错的那颗。
_TAG = "cpython-%d%d" % sys.version_info[:2]
_ALL = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "_C*.so")))
_SO = [p for p in _ALL if _TAG in os.path.basename(p)] or _ALL
if not _SO:
    raise ImportError(
        "custom_op: 找不到 _C*.so，请先编译："
        'CSIG_CUDA_ARCH_LIST="8.6+PTX" pip install -e custom_op_cpp_ext --no-build-isolation'
    )
torch.ops.load_library(_SO[0])

from .ops import (  # noqa: E402
    attn_d512,
    geglu,
    group_norm_act,
    sample_latent,
)

__all__ = [
    "attn_d512",
    "geglu",
    "group_norm_act",
    "sample_latent",
]
