"""单卡微基准：复刻 HYPIR G-step / D-step，扫 gradient_checkpointing x batch_size。

不碰真训练：只用 CUDA_VISIBLE_DEVICES 指定的一张空闲卡，合成数据，不读磁盘。
"""
import argparse, gc, json, os, time, warnings

import torch
import torch.nn.functional as F
import lpips
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel

from HYPIR.model.D import ImageConvNextDiscriminator

W = "/root/.cache/huggingface/csig/weights"
DT = torch.bfloat16
DEV = "cuda"
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]


def build(grad_ckpt):
    sched = DDPMScheduler.from_pretrained(W, subfolder="scheduler")
    te = CLIPTextModel.from_pretrained(W, subfolder="text_encoder", torch_dtype=DT).to(DEV).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(W, subfolder="vae", torch_dtype=DT).to(DEV).eval().requires_grad_(False)
    G = UNet2DConditionModel.from_pretrained(W, subfolder="unet", torch_dtype=DT).to(DEV)
    G.eval().requires_grad_(False)
    if grad_ckpt:
        G.enable_gradient_checkpointing()
    G.add_adapter(LoraConfig(r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=LORA_MODULES))
    for p in G.parameters():
        if p.requires_grad:
            p.data = p.to(torch.float32)
    D = ImageConvNextDiscriminator(precision="bf16").to(DEV)
    D.train().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    return sched, te, vae, G, D, net_lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[8])
    ap.add_argument("--grad-ckpt", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    results = []
    for gckpt in args.grad_ckpt:
        sched, te, vae, G, D, net_lpips = build(bool(gckpt))
        G_params = [p for p in G.parameters() if p.requires_grad]
        D_params = [p for p in D.parameters() if p.requires_grad]
        if not results:
            nG = sum(p.numel() for p in G_params)
            nD = sum(p.numel() for p in D_params)
            meta = dict(
                G_trainable=nG, G_dtype=str(G_params[0].dtype),
                D_trainable=nD, D_dtype=str(D_params[0].dtype),
                G_grad_bytes=sum(p.numel() * p.element_size() for p in G_params),
                D_grad_bytes=sum(p.numel() * p.element_size() for p in D_params),
                G_buffers=sum(b.numel() for b in G.buffers()),
                D_buffers=sum(b.numel() for b in D.buffers()),
                D_buffer_count=len(list(D.buffers())),
            )
            print("META " + json.dumps(meta), flush=True)
        G_opt = torch.optim.AdamW(G_params, lr=1e-5, betas=(0.9, 0.999))
        D_opt = torch.optim.AdamW(D_params, lr=1e-5, betas=(0.9, 0.999))

        for bs in args.bs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gt = (torch.rand(bs, 3, 512, 512, device=DEV) * 2 - 1)
            lq = (torch.rand(bs, 3, 512, 512, device=DEV) * 2 - 1)
            ids = torch.randint(0, 40000, (bs, 77), device=DEV)
            with torch.no_grad():
                c_txt = te(ids)[0]
            t = torch.full((bs,), 200, dtype=torch.long, device=DEV)

            def fwd_G():
                with torch.no_grad():
                    z_lq = vae.encode(lq.to(DT)).latent_dist.sample()
                z_in = z_lq * vae.config.scaling_factor
                with torch.autocast("cuda", DT):
                    eps = G(z_in, t, encoder_hidden_states=c_txt).sample
                z = sched.step(eps, 200, z_in).pred_original_sample
                return vae.decode(z.to(DT) / vae.config.scaling_factor).sample.float()

            def g_step():
                D.eval().requires_grad_(False)
                x = fwd_G()
                loss = F.mse_loss(x, gt) + 5 * net_lpips(x, gt).mean()
                with torch.autocast("cuda", DT):
                    loss = loss + 0.5 * D(x, for_G=True).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(G_params, 1.0)
                G_opt.step(); G_opt.zero_grad(set_to_none=True)

            def d_step():
                with torch.no_grad():
                    x = fwd_G()
                D.train().requires_grad_(True)
                with torch.autocast("cuda", DT):
                    loss = D(gt, for_real=True).mean() + D(x, for_real=False).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(D_params, 1.0)
                D_opt.step(); D_opt.zero_grad(set_to_none=True)

            try:
                timings = {}
                for name, fn in (("G", g_step), ("D", d_step)):
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for _ in range(args.iters):
                        fn()
                    torch.cuda.synchronize()
                    timings[name] = (time.perf_counter() - t0) / args.iters
                peak = torch.cuda.max_memory_allocated() / 2**30
                r = dict(grad_ckpt=gckpt, bs=bs, t_G=round(timings["G"], 4),
                         t_D=round(timings["D"], 4), t_iter=round(timings["G"] + timings["D"], 4),
                         peak_GB=round(peak, 2),
                         img_per_s=round(2 * bs / (timings["G"] + timings["D"]), 2))
            except torch.OutOfMemoryError:
                r = dict(grad_ckpt=gckpt, bs=bs, oom=True)
                torch.cuda.empty_cache()
            print("RES " + json.dumps(r), flush=True)
            results.append(r)
            del gt, lq, ids, c_txt, t
            gc.collect(); torch.cuda.empty_cache()
        del sched, te, vae, G, D, net_lpips, G_opt, D_opt, G_params, D_params
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print("ALL " + json.dumps(results))


if __name__ == "__main__":
    main()
