"""把 diffusers 的 AutoencoderTiny 包成 AutoencoderKL 的接口。

训练侧（trainer/base.py + trainer/sd2.py）和推理侧（enhancer/base.py + enhancer/sd2.py）
用到 VAE 的地方只有四个：`encode(x).latent_dist.sample()`、`decode(z).sample`、
`config.scaling_factor`、`config.latent_channels`。这个壳把它们全兜住，两边都不用改。

换算 z_tae = A * z_sd + B 由 csig/taesd_probe.py 在真实图上拟合（相关系数 0.9639），
与 csig/taesd_exp.py、submit/model_dir/runner.py 是同一组常量 ——
训练和推理必须用同一份，否则条件失配。
"""
from types import SimpleNamespace

import torch

TAESD_A, TAESD_B = 0.16668, 0.01702


class _TAESDLatent:
    """冒充 AutoencoderKL 的 latent_dist。

    TAESD 是确定性编码器，没有后验分布，sample() 就是恒等。这是本方案与官方唯一
    无法消除的偏离：官方 encode().latent_dist.sample() 带 VAE 后验采样噪声。
    """

    def __init__(self, z):
        self.z = z

    def sample(self):
        return self.z


class TAESDWrapper(torch.nn.Module):
    """对外的 latent 约定是「未缩放 SD latent」，UNet 与官方 LoRA 权重因此原样可用。"""

    def __init__(self, tae):
        super().__init__()
        self.tae = tae
        # AutoencoderTiny 的 config 是 FrozenDict，改不动，另起一个只放用得着的字段。
        # scaling_factor 用 SD2.1 的 0.18215，而不是 AutoencoderTiny 自己的 1.0。
        self.config = SimpleNamespace(scaling_factor=0.18215, latent_channels=4)

    def encode(self, x):
        # 调用方写的是 encode(x).latent_dist.sample()，两层都得有
        z = (self.tae.encode(x).latents - TAESD_B) / TAESD_A
        return SimpleNamespace(latent_dist=_TAESDLatent(z))

    def decode(self, z):
        return self.tae.decode(z * TAESD_A + TAESD_B)


def build_taesd(dtype, device, compile_parts=False):
    """建一个冻结的 TAESD 并包好。compile_parts 只在训练侧用。"""
    from diffusers import AutoencoderTiny

    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
    tae.eval().requires_grad_(False)
    if compile_parts:
        tae.encoder = torch.compile(tae.encoder)
        tae.decoder = torch.compile(tae.decoder)
    return TAESDWrapper(tae)
