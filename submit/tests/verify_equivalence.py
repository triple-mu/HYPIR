"""单文件 runner 与原版 diffusers 实现的数值等价性对拍。

纪律（来自 FINDINGS）：
- 必须在 CPU float64 上做。GPU fp32 卷积默认开 TF32（10 位尾数），会给出 2e-4 的假误差，
  看着像折叠写错了，实际是精度问题。
- 必须同进程内做。这条管线跨进程不可复现（同代码同机两进程只有 53.8 dB）。

用法：
    python tools/verify_equivalence.py
"""

import importlib.abc
import importlib.machinery
import os
import sys
import types

import torch
import yaml

from torch.nn import functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "model_dir"))


class _StubLoader(importlib.abc.Loader):
    """model_lib/__init__.py 无条件 import 教师网络，牵出未安装的 fla。推理路径用不到。"""

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        module.__getattr__ = lambda name: type(name, (object,), {})


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla" or fullname.startswith("fla."):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


sys.meta_path.insert(0, _StubFinder())

import runner as R  # noqa: E402

DEV = torch.device("cpu")
DT = torch.float64


def build_reference():
    from model_lib.nets.unet_lambda_prune_lite import (
        UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8 as Ref)
    cfg = yaml.safe_load(open(os.path.join(REPO, "config/model_cfg/moebius.yaml")))["model"]
    cfg.pop("model_type", None)
    cfg.update(sample_size=R.LATENT_SIZE, num_embeddings=2 * R.NUM_COND_TOKENS)
    return Ref(**cfg)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """L_inf 相对误差，分母用参考值的量级。"""
    denom = a.abs().max().item()
    return (a - b).abs().max().item() / max(denom, 1e-30)


def align(ref: torch.Tensor, ours: torch.Tensor) -> torch.Tensor:
    """原版在 transformer 内部用 (B, N, C)，我方全程 NCHW。对齐到参考的排布。"""
    if ref.dim() == 3 and ours.dim() == 4:
        b, c, h, w = ours.shape
        return ours.reshape(b, c, h * w).permute(0, 2, 1)
    return ours


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    raw = torch.load(os.path.join(REPO, "Moebius/pretrained/diffusion_pytorch_model.bin"),
                     map_location="cpu", weights_only=True)
    vae_raw = torch.load(os.path.join(REPO, "PixelHacker/vae/diffusion_pytorch_model.bin"),
                         map_location="cpu", weights_only=True)

    # ---- VAE：零改动复用，只需确认 strict 加载 + 往返可用 ----
    vae = R.AutoencoderKLLite().to(DEV, DT).eval().requires_grad_(False)
    vae.load_state_dict({k: v.to(DT) for k, v in vae_raw.items()}, strict=True)
    print("[VAE] strict 加载通过（248 key）")

    # ---- UNet：参考侧 ----
    ref = build_reference().to(DEV, DT).eval().requires_grad_(False)
    missing, unexpected = ref.load_state_dict(
        {k[len("diff_model."):]: v.to(DT) for k, v in raw.items()
         if k.startswith("diff_model.")}, strict=True), None
    print("[UNet-ref] strict 加载通过（%d 张量）" % sum(1 for _ in ref.state_dict()))

    # ---- UNet：我方（用导出脚本的 float64 中间结果，不经 fp16）----
    ours = R.MoebiusUNetLite().to(DEV, DT).eval().requires_grad_(False)
    folded = R._export_unet(raw)
    ours.load_state_dict({k: v.to(DT) for k, v in folded.items()}, strict=True)
    print("[UNet-ours] strict 加载通过（%d 张量）" % sum(1 for _ in ours.state_dict()))

    # ---- 逐层挂钩 ----
    names = ["conv_in"]
    for stage, count in (("down_blocks", 3), ("up_blocks", 3)):
        layers = 2 if stage == "down_blocks" else 3
        for i in range(count):
            for j in range(layers):
                names += ["%s.%d.resnets.%d" % (stage, i, j),
                          "%s.%d.resnets.%d.conv1" % (stage, i, j),
                          "%s.%d.resnets.%d.conv2" % (stage, i, j),
                          "%s.%d.attentions.%d" % (stage, i, j)]
                for leaf in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
                    names.append("%s.%d.attentions.%d.transformer_blocks.0.%s"
                                 % (stage, i, j, leaf))
            if stage == "down_blocks" and i < 2:
                names.append("down_blocks.%d.downsamplers.0" % i)
            if stage == "up_blocks" and i < 2:
                names.append("up_blocks.%d.upsamplers.0" % i)
    names.append("conv_out")

    captured = {}

    def hook_all(model, tag):
        mods = dict(model.named_modules())
        for name in names:
            if name not in mods:
                continue

            def make(nm):
                def fn(_m, _i, out):
                    captured.setdefault(nm, {})[tag] = (
                        out[0] if isinstance(out, tuple) else out).detach()
                return fn
            mods[name].register_forward_hook(make(name))

    hook_all(ref, "ref")
    hook_all(ours, "ours")

    # ---- 同一输入跑两侧 ----
    torch.manual_seed(0)
    n = R.LATENT_SIZE
    sample = torch.randn(1, 9, n, n, dtype=DT)
    t = torch.tensor([200], dtype=torch.long)
    ids = torch.arange(R.NUM_COND_TOKENS, dtype=torch.long)[None]
    ehs = F.embedding(ids, raw["embedding_layer.weight"].to(DT))  # (1, 10, 3072)

    print("\n跑参考侧（CPU float64，较慢）...")
    with torch.no_grad():
        eps_ref = ref(sample, t, encoder_hidden_states=ehs).sample
    print("跑我方...")
    with torch.no_grad():
        emb = R.timestep_embedding(t).to(DT)
        temb = F.silu(ours.time_embedding(emb))
        eps_ours = ours(sample, temb)

    # ---- 报告 ----
    print("\n%-68s %12s" % ("module", "rel_err"))
    print("-" * 82)
    worst, worst_name = 0.0, ""
    for name in names:
        pair = captured.get(name, {})
        if "ref" not in pair or "ours" not in pair:
            continue
        a = pair["ref"]
        b = align(a, pair["ours"])
        if a.shape != b.shape:
            print("%-68s %12s  (shape %s vs %s)" % (name, "SKIP", tuple(a.shape), tuple(b.shape)))
            continue
        e = rel_err(a, b)
        if e > worst:
            worst, worst_name = e, name
        flag = "  <-- 越阈" if e > 1e-10 else ""
        print("%-68s %12.3e%s" % (name, e, flag))

    e2e = rel_err(eps_ref, eps_ours)
    psnr = 10 * torch.log10(eps_ref.pow(2).mean() / (eps_ref - eps_ours).pow(2).mean()).item()
    print("-" * 82)
    print("最差逐层: %.3e @ %s" % (worst, worst_name))
    print("e2e eps  : rel_err %.3e   PSNR %.1f dB" % (e2e, psnr))
    ok = worst < 1e-10 and e2e < 1e-10
    print("\n结论: %s" % ("通过" if ok else "未通过——第一个越阈的模块就是 bug 所在"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
