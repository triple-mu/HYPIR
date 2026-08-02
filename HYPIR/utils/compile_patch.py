"""torch.compile 的前置补丁。在 init_models() 最前面调一次。

三条补丁解决三个具体问题（依据是 torch 2.6 上的实测；本文件对 2.13 做了容错，
打不上就跳过并告警，不让训练崩）：

1. **共享 attn processor** —— diffusers 给每个 Attention 建一个独立的 processor 对象，
   dynamo 按对象 id 做 guard，导致每个 Transformer2DModel 各编一份图，撞穿 cache_size_limit。
   统一成一个无状态实例后，编译时间 302s -> 55s，加速比 1.30x -> 1.40x。

2. **常量化 is_torch_version** —— diffusers 在 bf16/fp16 路径上每次前向都会走到它，
   是 graph break 的唯一来源。缓存结果即可。注意 diffusers 用 LazyModule，
   不显式 import 子模块就打不到补丁。

3. **woq pattern guard** —— torch 2.6.0 的 inductor 在编 UNet 推理图时会踩
   `_is_valid_woq_optimization_pattern` 的 AttributeError。2.7+ 已修，
   所以这里只在能打到时才打。

另外设 `optimize_ddp = False`：LoRA 梯度只有 1.04 GB，NVLink 上 allreduce 几毫秒，
不值得为通信重叠把编译图切碎。
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


def patch_for_dynamo(unet=None):
    """返回打上的补丁数，便于日志核对。"""
    n = 0

    # --- (1) 共享 attn processor + dynamo 配置 --------------------------------
    if unet is not None:
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            unet.set_attn_processor(AttnProcessor2_0())
            n += 1
        except Exception as e:
            logger.warning("compile_patch: 共享 attn processor 失败（跳过）: %s", e)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.optimize_ddp = False
    # inductor 的 donated buffer 优化会假设反向只走一次且不 retain_graph。
    # 这条管线里 G 的输出同时喂给 MSE/LPIPS/D 三路损失，反向图结构不满足该假设，
    # 会报 "compiled with non-empty donated buffers which requires create_graph=False"。
    try:
        import torch._functorch.config as _fc
        _fc.donated_buffer = False
    except Exception:
        pass

    # --- (2) 常量化 diffusers 的 is_torch_version ----------------------------
    try:
        import diffusers.utils.import_utils as iu
        # LazyModule：不显式 import 这些子模块，下面的 sys.modules 扫描就打不到它们
        import diffusers.models.upsampling            # noqa: F401
        import diffusers.models.downsampling          # noqa: F401
        import diffusers.models.resnet                # noqa: F401
        import diffusers.models.unets.unet_2d_blocks  # noqa: F401
        import diffusers.models.unets.unet_2d_condition  # noqa: F401
        import diffusers.models.autoencoders.vae      # noqa: F401

        orig, cache = iu.is_torch_version, {}

        def cached(op, ver):
            key = (op, ver)
            if key not in cache:
                cache[key] = orig(op, ver)
            return cache[key]

        cached("<", "2.1")
        cached(">=", "1.11.0")
        hit = 0
        for name, mod in list(sys.modules.items()):
            if name.startswith("diffusers") and getattr(mod, "is_torch_version", None) is not None:
                mod.is_torch_version = cached
                hit += 1
        if hit:
            n += 1
    except Exception as e:
        logger.warning("compile_patch: is_torch_version 常量化失败（跳过）: %s", e)

    # --- (3) woq pattern guard（torch 2.6.0 专属 bug，2.7+ 已修）--------------
    try:
        import torch._inductor.fx_passes.quantization as _q
        _ov = _q._is_valid_woq_optimization_pattern

        def _safe_woq():
            f = _ov()

            def g(match):
                try:
                    return f(match)
                except AttributeError:
                    return False
            return g

        _q._is_valid_woq_optimization_pattern = _safe_woq
        n += 1
    except (ImportError, AttributeError):
        pass    # 新版 torch 没有这个符号，说明 bug 已修，无需打

    logger.info("compile_patch: 打上 %d 组补丁 (torch %s)", n, torch.__version__)
    return n
