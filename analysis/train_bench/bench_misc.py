"""单卡微基准：(1) EMA 更新耗时（现实现 vs _foreach_lerp_）(2) DDP bucket-view stride 告警的根因。"""
import json, time, warnings
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig

W = "/root/.cache/huggingface/csig/weights"
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]

G = UNet2DConditionModel.from_pretrained(W, subfolder="unet", torch_dtype=torch.bfloat16).to("cuda")
G.eval().requires_grad_(False)
G.enable_gradient_checkpointing()
G.add_adapter(LoraConfig(r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=LORA_MODULES))
for p in G.parameters():
    if p.requires_grad:
        p.data = p.to(torch.float32)

named = [(n, p) for n, p in G.named_parameters() if p.requires_grad]
params = [p for _, p in named]
print(json.dumps(dict(n_tensors=len(params), n_params=sum(p.numel() for p in params))), flush=True)

# ---- (1) EMA ----
ema = {n: p.clone().detach() for n, p in named}
decay = 0.999


def ema_naive():
    for n, p in named:
        ema[n] = decay * ema[n] + (1 - decay) * p.clone().detach()


ema_list = [ema[n] for n, _ in named]


def ema_foreach():
    torch._foreach_lerp_(ema_list, params, 1 - decay)


for name, fn in (("naive", ema_naive), ("foreach", ema_foreach)):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    print("EMA " + json.dumps({name: round((time.perf_counter() - t0) / 20 * 1e3, 3)}), flush=True)
    ema_list = [ema[n] for n, _ in named]

# ---- (2) grad stride vs bucket view ----
z = torch.randn(2, 4, 64, 64, device="cuda")
c = torch.randn(2, 77, 1024, device="cuda", dtype=torch.bfloat16)
t = torch.full((2,), 200, dtype=torch.long, device="cuda")
with torch.autocast("cuda", torch.bfloat16):
    out = G(z, t, encoder_hidden_states=c).sample
out.float().pow(2).mean().backward()

bad = []
for n, p in named:
    if p.grad is None:
        bad.append((n, "NO_GRAD", None, None))
        continue
    mf = torch.channels_last if p.dim() == 4 and p.is_contiguous(memory_format=torch.channels_last) \
        and not p.is_contiguous() else torch.contiguous_format
    if not p.grad.is_contiguous(memory_format=mf):
        bad.append((n, tuple(p.shape), p.grad.stride(), p.stride()))
print("STRIDE_MISMATCH " + json.dumps(dict(total=len(named), bad=len(bad), sample=bad[:8]), default=str), flush=True)
nog = [n for n, p in named if p.grad is None]
print("NO_GRAD " + json.dumps(dict(count=len(nog), sample=nog[:8])), flush=True)
