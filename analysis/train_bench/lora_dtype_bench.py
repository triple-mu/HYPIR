"""Does peft's `x = x.to(lora_A.weight.dtype)` (fp32) cost anything under bf16 autocast?

Three variants, all with fp32 master LoRA weights except (c):
  a) as in HYPIR today: LoRA params fp32  -> peft upcasts every LoRA input to fp32
  b) same, but cast_input_dtype_enabled=False -> input stays bf16, autocast handles the weight
  c) LoRA params bf16 (no fp32 master; convergence-unsafe, measured only as a bound)
"""
import time, gc, torch
import mem_bench as MB

DEV = "cuda"
BS = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 8


def bench(tag, mutate=None, lora_bf16=False):
    gc.collect(); torch.cuda.empty_cache()
    unet = MB.build_unet(grad_ckpt=True)
    if lora_bf16:
        for p in unet.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    n_touched = 0
    if mutate:
        for m in unet.modules():
            if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})):
                mutate(m); n_touched += 1
    z = torch.randn(BS, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    t = torch.full((BS,), 200, device=DEV, dtype=torch.long)
    ctx = torch.randn(BS, 77, 1024, device=DEV, dtype=torch.bfloat16)

    def f():
        with torch.autocast("cuda", torch.bfloat16):
            o = unet(z, t, encoder_hidden_states=ctx).sample
        o.float().square().mean().backward()
        for p in unet.parameters():
            p.grad = None

    for _ in range(3):
        f()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    f()
    peak = (torch.cuda.max_memory_allocated() - base) / 1024 ** 3
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"{tag:52s} peak_act={peak:6.3f} GB  {ms:7.2f} ms   (lora layers touched={n_touched})", flush=True)
    del unet, z, ctx
    gc.collect(); torch.cuda.empty_cache()


print(f"UNet+LoRA(r=256) fwd+bwd, grad_ckpt=ON, bs={BS}, 64x64 latent")
bench("a) LoRA fp32, peft input-cast ON  (HYPIR today)")


def off(m):
    m.cast_input_dtype_enabled = False


bench("b) LoRA fp32, peft input-cast OFF", mutate=off)
bench("c) LoRA bf16 (no fp32 master, bound only)", lora_bf16=True)
