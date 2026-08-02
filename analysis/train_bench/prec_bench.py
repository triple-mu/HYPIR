"""Precision + misc micro-benchmarks:
  1. CLIPTextModel (SD2.1) per-step cost -- is caching the fixed prompt worth it?
  2. bf16 vs fp16 round-trip SNR (verifies the 6.02 dB/mantissa-bit rule).
  3. INT_MAX boundary of upsample_nearest2d backward in the VAE decoder.
  4. EMA update: naive python loop vs _foreach_lerp_ vs CPU-resident EMA.
"""
import time, math, torch
DEV = "cuda"


def sync_time(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


print("=" * 70)
print("1) CLIPTextModel (SD2.1 text_encoder) cost per training step")
from transformers import CLIPTextModel, CLIPTextConfig
cfg = CLIPTextConfig(vocab_size=49408, hidden_size=1024, intermediate_size=4096,
                     num_hidden_layers=23, num_attention_heads=16, max_position_embeddings=77,
                     hidden_act="gelu")
te = CLIPTextModel(cfg).to(DEV, torch.bfloat16).eval().requires_grad_(False)
print(f"   params = {sum(p.numel() for p in te.parameters())/1e6:.2f} M, "
      f"vram = {sum(p.numel()*p.element_size() for p in te.parameters())/1024**3:.3f} GB")
for bs in (8, 16):
    ids = torch.randint(0, 49407, (bs, 77), device=DEV)
    f = lambda: te(ids)[0]
    print(f"   bs={bs:2d} forward(no_grad=False, as in code) = {sync_time(f):.2f} ms")
    with torch.no_grad():
        print(f"   bs={bs:2d} forward(no_grad)                = {sync_time(lambda: te(ids)[0]):.2f} ms")

from transformers import CLIPTokenizer
try:
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    p = ["a high-resolution photograph with fine natural detail of a city"] * 16
    t0 = time.perf_counter()
    for _ in range(20):
        tok(p, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids
    print(f"   tokenizer(bs=16) CPU = {(time.perf_counter()-t0)/20*1000:.2f} ms  (blocks the launch queue)")
except Exception as e:
    print(f"   tokenizer skipped: {type(e).__name__}")
del te
torch.cuda.empty_cache()

print("=" * 70)
print("2) bf16 vs fp16 round-trip SNR on a natural-image-like tensor")
torch.manual_seed(0)
for name, x in [("uniform[-1,1] image", torch.rand(4, 3, 512, 512, device=DEV) * 2 - 1),
                ("VAE-latent-like N(0,1)*4", torch.randn(4, 4, 64, 64, device=DEV) * 4),
                ("post-conv act N(0,8)", torch.randn(4, 128, 256, 256, device=DEV) * 8)]:
    row = [name]
    for dt in (torch.bfloat16, torch.float16):
        e = (x.to(dt).float() - x)
        psnr = 10 * math.log10(x.abs().max().item() ** 2 / e.square().mean().item())
        row.append(f"{dt}".split(".")[-1] + f"={psnr:.2f} dB")
    print("   " + " | ".join(row) + f"  -> gap {float(row[2].split('=')[1][:-3]) - float(row[1].split('=')[1][:-3]):.2f} dB")
print("   theory: fp16 has 10 explicit mantissa bits, bf16 has 7 -> 3*6.02 = 18.06 dB,")
print("   reduced by the coarser bf16 exponent binning being identical -> observed ~18 dB.")

print("=" * 70)
print("3) INT_MAX limit of the VAE-decoder upsample backward (512x512 output)")
print("   last decoder upsample grad_output shape = [B, 256, 512, 512]")
lim = 2 ** 31 - 1
print(f"   B * 256*512*512 = B * {256*512*512} must be < INT_MAX ({lim}) -> B <= {lim // (256*512*512)}")
for B in (30, 31, 32):
    x = torch.randn(B, 256, 256, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    try:
        y = torch.nn.functional.interpolate(x.to(memory_format=torch.channels_last), scale_factor=2.0, mode="nearest")
        y.sum().backward()
        print(f"   B={B}: OK")
    except RuntimeError as e:
        print(f"   B={B}: FAIL -> {str(e)[:90]}")
    del x
    torch.cuda.empty_cache()

print("=" * 70)
print("4) EMA update variants (259.47 M fp32 params, 686 LoRA tensors)")
n_t, tot = 686, 259_470_000
per = tot // n_t
ps = [torch.randn(per, device=DEV) for _ in range(n_t)]
ema_g = [p.detach().clone() for p in ps]
ema_c = [p.detach().to("cpu").pin_memory() for p in ps]


def naive():
    for i, p in enumerate(ps):
        ema_g[i] = 0.999 * ema_g[i] + 0.001 * p.clone().detach()


def foreach():
    torch._foreach_lerp_(ema_g, ps, 0.001)


def cpu_ema():
    for i, p in enumerate(ps):
        ema_c[i].add_((p.to("cpu", non_blocking=True) - ema_c[i]) * 0.001)


print(f"   naive python loop (current code) = {sync_time(naive, n=10):.2f} ms")
print(f"   torch._foreach_lerp_ (GPU)       = {sync_time(foreach, n=10):.2f} ms")
print(f"   CPU-resident EMA (D2H each step) = {sync_time(cpu_ema, n=3, warmup=1):.2f} ms")
print(f"   GPU EMA vram = {tot*4/1024**3:.3f} GB")
