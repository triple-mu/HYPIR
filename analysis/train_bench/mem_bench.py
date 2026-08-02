"""Memory/speed micro-benchmark for the HYPIR LoRA-GAN training step.

Reproduces the exact module stack and dtype policy of HYPIR/trainer/{base,sd2}.py
and measures per-component static memory, activation memory and step time.
Random weights: architecture and dtypes are what matter for memory/speed.
"""
import argparse, gc, json, time
import torch, torch.nn.functional as F
from torch import nn

DEV = "cuda"

SD21_UNET_CFG = dict(
    act_fn="silu", attention_head_dim=[5, 10, 20, 20],
    block_out_channels=[320, 640, 1280, 1280], center_input_sample=False,
    cross_attention_dim=1024,
    down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    downsample_padding=1, dual_cross_attention=False, flip_sin_to_cos=True, freq_shift=0,
    in_channels=4, layers_per_block=2, mid_block_scale_factor=1, norm_eps=1e-5,
    norm_num_groups=32, num_class_embeds=None, only_cross_attention=False, out_channels=4,
    sample_size=64,
    up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    use_linear_projection=True, upcast_attention=False,
)
SD21_VAE_CFG = dict(
    act_fn="silu", block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock2D"] * 4, in_channels=3, latent_channels=4,
    layers_per_block=2, norm_num_groups=32, out_channels=3, sample_size=768,
    up_block_types=["UpDecoderBlock2D"] * 4,
)


def mb(x):
    return x / 1024 ** 3


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(fn, warmup=1, iters=3):
    """Return (peak_alloc_delta_GB, ms_per_iter). Peak measured on a clean slate."""
    for _ in range(warmup):
        fn()
    reset()
    base = torch.cuda.memory_allocated()
    fn()
    peak = torch.cuda.max_memory_allocated()
    reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    return mb(peak - base), dt


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def build_unet(lora_rank=256, grad_ckpt=True, dtype=torch.bfloat16):
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    unet = UNet2DConditionModel(**SD21_UNET_CFG).to(DEV, dtype)
    unet.eval().requires_grad_(False)
    if grad_ckpt:
        unet.enable_gradient_checkpointing()
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                        "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"],
    )
    unet.add_adapter(cfg)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return unet


def build_vae(dtype=torch.bfloat16):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL(**SD21_VAE_CFG).to(DEV, dtype)
    vae.eval().requires_grad_(False)
    return vae


def build_lpips_like():
    """torchvision VGG16 features == the trunk of lpips.LPIPS(net='vgg')."""
    import torchvision
    vgg = torchvision.models.vgg16(weights=None).features[:30].to(DEV).eval().requires_grad_(False)
    return vgg


def build_convnext_d():
    """timm convnext_xxlarge == the open_clip CLIP-convnext_xxlarge visual trunk."""
    import timm
    m = timm.create_model("convnext_xxlarge", pretrained=False, num_classes=0)
    m = m.to(DEV, torch.bfloat16).eval().requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--parts", type=str, default="all")
    args = ap.parse_args()
    R = args.res
    out = {}

    def rec(k, v):
        out[k] = v
        print(f"{k:58s} peak={v[0]:7.3f} GB  time={v[1]:8.2f} ms", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parts = args.parts.split(",") if args.parts != "all" else ["static", "unet", "vae", "lpips", "d", "e2e"]

    # ---------------- static memory ----------------
    if "static" in parts:
        reset()
        b0 = torch.cuda.memory_allocated()
        unet = build_unet()
        b1 = torch.cuda.memory_allocated()
        vae = build_vae()
        b2 = torch.cuda.memory_allocated()
        vgg = build_lpips_like()
        b3 = torch.cuda.memory_allocated()
        d = build_convnext_d()
        b4 = torch.cuda.memory_allocated()
        tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"[static] UNet+LoRA  {mb(b1-b0):.3f} GB  total={nparams(unet)/1e6:.2f}M trainable={tr/1e6:.2f}M")
        print(f"[static] VAE bf16   {mb(b2-b1):.3f} GB  total={nparams(vae)/1e6:.2f}M")
        print(f"[static] VGG fp32   {mb(b3-b2):.3f} GB  total={nparams(vgg)/1e6:.2f}M")
        print(f"[static] ConvNeXtXL {mb(b4-b3):.3f} GB  total={nparams(d)/1e6:.2f}M", flush=True)
        opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-5)
        for p in unet.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        b5 = torch.cuda.memory_allocated()
        opt.step()
        b6 = torch.cuda.memory_allocated()
        print(f"[static] G grads    {mb(b5-b4):.3f} GB")
        print(f"[static] G AdamW    {mb(b6-b5):.3f} GB", flush=True)
        del unet, vae, vgg, d, opt
        reset()

    for BS in args.bs:
        print(f"\n########## batch = {BS} @ {R}x{R} ##########", flush=True)

        # ---------------- UNet fwd+bwd ----------------
        if "unet" in parts:
            for gc_on in [True, False]:
                unet = build_unet(grad_ckpt=gc_on)
                z = torch.randn(BS, 4, R // 8, R // 8, device=DEV, dtype=torch.bfloat16)
                t = torch.full((BS,), 200, device=DEV, dtype=torch.long)
                ctx = torch.randn(BS, 77, 1024, device=DEV, dtype=torch.bfloat16)

                def f():
                    with torch.autocast("cuda", torch.bfloat16):
                        o = unet(z, t, encoder_hidden_states=ctx).sample
                    o.float().square().mean().backward()
                    for p in unet.parameters():
                        p.grad = None
                rec(f"unet_fwdbwd grad_ckpt={gc_on}", measure(f))
                del unet, z, ctx
                reset()

        # ---------------- VAE encode / decode ----------------
        if "vae" in parts:
            vae = build_vae()
            img = torch.randn(BS, 3, R, R, device=DEV, dtype=torch.bfloat16)

            def f_enc():
                with torch.no_grad():
                    vae.encode(img).latent_dist.sample()
            rec("vae_encode (no_grad)", measure(f_enc))

            z = torch.randn(BS, 4, R // 8, R // 8, device=DEV, dtype=torch.bfloat16, requires_grad=True)

            def f_dec():
                x = vae.decode(z).sample.float()
                x.square().mean().backward()
                z.grad = None
            rec("vae_decode fwd+bwd (grad flows to z)", measure(f_dec))

            # with decoder gradient checkpointing
            vae.enable_gradient_checkpointing()
            vae.decoder.gradient_checkpointing = True
            rec("vae_decode fwd+bwd  +grad_ckpt", measure(f_dec))
            del vae, img, z
            reset()

        # ---------------- LPIPS trunk ----------------
        if "lpips" in parts:
            vgg = build_lpips_like()
            x = torch.randn(BS, 3, R, R, device=DEV, requires_grad=True)
            y = torch.randn(BS, 3, R, R, device=DEV)

            def f32():
                a = vgg(x); b = vgg(y)
                (a - b).square().mean().backward()
                x.grad = None
            rec("lpips_vgg fp32 (current: outside autocast)", measure(f32))

            def fbf():
                with torch.autocast("cuda", torch.bfloat16):
                    a = vgg(x); b = vgg(y)
                    l = (a - b).square().mean()
                l.float().backward()
                x.grad = None
            rec("lpips_vgg bf16 autocast", measure(fbf))
            del vgg, x, y
            reset()

        # ---------------- Discriminator backbone ----------------
        if "d" in parts:
            d = build_convnext_d()
            x = torch.randn(BS, 3, R, R, device=DEV, requires_grad=True)

            def fd():
                with torch.autocast("cuda", torch.bfloat16):
                    o = d(x)
                o.float().square().mean().backward()
                x.grad = None
            rec("D convnext_xxlarge fwd+bwd (grad->x, G step)", measure(fd))

            def fd_nograd():
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    d(x)
            rec("D convnext_xxlarge fwd only (D step, frozen)", measure(fd_nograd))

            d.set_grad_checkpointing(True)
            rec("D convnext_xxlarge fwd+bwd +grad_ckpt", measure(fd))
            del d, x
            reset()

    print("\nJSON " + json.dumps({k: [round(v[0], 4), round(v[1], 2)] for k, v in out.items()}))


if __name__ == "__main__":
    main()
