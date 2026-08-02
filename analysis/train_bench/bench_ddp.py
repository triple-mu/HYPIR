"""2 卡 DDP 端到端微基准：复刻 HYPIR 的 G/D 交替，扫 DDP 各项开关。

torchrun --nproc_per_node=2 bench_ddp.py --cfg base gbv bcast_off bf16hook fup
合成数据、无 dataloader，故测到的差异纯粹来自 DDP 与优化器/EMA。
"""
import argparse, json, os, time, warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
import lpips
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel

from HYPIR.model.D import ImageConvNextDiscriminator

W = "/root/.cache/huggingface/csig/weights"
DT = torch.bfloat16
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]

CFGS = {
    # name: (ddp_kwargs_G, ddp_kwargs_D, comm_hook, fused_ema)
    "base":       (dict(), dict(), None, False),
    "gbv":        (dict(gradient_as_bucket_view=True), dict(gradient_as_bucket_view=True), None, False),
    "bcast_off":  (dict(gradient_as_bucket_view=True, broadcast_buffers=False),
                   dict(gradient_as_bucket_view=True, broadcast_buffers=False), None, False),
    "bf16hook":   (dict(gradient_as_bucket_view=True, broadcast_buffers=False),
                   dict(gradient_as_bucket_view=True, broadcast_buffers=False), "bf16", False),
    "ema":        (dict(gradient_as_bucket_view=True, broadcast_buffers=False),
                   dict(gradient_as_bucket_view=True, broadcast_buffers=False), "bf16", True),
    "fup":        (dict(find_unused_parameters=True), dict(find_unused_parameters=True), None, False),
    "static":     (dict(static_graph=True), dict(static_graph=True), None, False),
    "bucket200":  (dict(gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=200),
                   dict(gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=200), None, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="+", default=["base"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--no-ddp", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = "cuda"

    sched = DDPMScheduler.from_pretrained(W, subfolder="scheduler")
    te = CLIPTextModel.from_pretrained(W, subfolder="text_encoder", torch_dtype=DT).to(dev).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(W, subfolder="vae", torch_dtype=DT).to(dev).eval().requires_grad_(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)

    G0 = UNet2DConditionModel.from_pretrained(W, subfolder="unet", torch_dtype=DT).to(dev)
    G0.eval().requires_grad_(False)
    G0.enable_gradient_checkpointing()
    G0.add_adapter(LoraConfig(r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=LORA_MODULES))
    for p in G0.parameters():
        if p.requires_grad:
            p.data = p.to(torch.float32)
    D0 = ImageConvNextDiscriminator(precision="bf16").to(dev)
    D0.train().requires_grad_(True)

    bs = args.bs
    gt = torch.rand(bs, 3, 512, 512, device=dev) * 2 - 1
    lq = torch.rand(bs, 3, 512, 512, device=dev) * 2 - 1
    with torch.no_grad():
        c_txt = te(torch.randint(0, 40000, (bs, 77), device=dev))[0]
    t = torch.full((bs,), 200, dtype=torch.long, device=dev)

    for cfg_name in args.cfg:
        gk, dk, hook, fused_ema = CFGS[cfg_name]
        if args.no_ddp:
            G, D = G0, D0
        else:
            G = DDP(G0, device_ids=[torch.cuda.current_device()], **gk)
            D = DDP(D0, device_ids=[torch.cuda.current_device()], **dk)
            if hook == "bf16":
                G.register_comm_hook(None, default_hooks.bf16_compress_hook)
                D.register_comm_hook(None, default_hooks.bf16_compress_hook)
        G_params = [p for p in G0.parameters() if p.requires_grad]
        D_params = [p for p in D0.parameters() if p.requires_grad]
        G_opt = torch.optim.AdamW(G_params, lr=1e-5)
        D_opt = torch.optim.AdamW(D_params, lr=1e-5)
        ema = [p.clone().detach() for p in G_params]
        names = [n for n, p in G0.named_parameters() if p.requires_grad]
        ema_d = {n: e for n, e in zip(names, ema)}

        def fwd_G():
            with torch.no_grad():
                z_lq = vae.encode(lq.to(DT)).latent_dist.sample()
            z_in = z_lq * vae.config.scaling_factor
            with torch.autocast("cuda", DT):
                eps = G(z_in, t, encoder_hidden_states=c_txt).sample
            eps = eps.sample if hasattr(eps, "sample") else eps
            z = sched.step(eps, 200, z_in).pred_original_sample
            return vae.decode(z.to(DT) / vae.config.scaling_factor).sample.float()

        def ema_update():
            if fused_ema:
                torch._foreach_lerp_(ema, G_params, 0.001)
            else:
                for n, p in zip(names, G_params):
                    ema_d[n] = 0.999 * ema_d[n] + 0.001 * p.clone().detach()

        def one_iter():
            D0.eval().requires_grad_(False)
            x = fwd_G()
            loss = F.mse_loss(x, gt) + 5 * net_lpips(x, gt).mean()
            with torch.autocast("cuda", DT):
                loss = loss + 0.5 * D(x, for_G=True).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G_params, 1.0)
            G_opt.step(); G_opt.zero_grad(set_to_none=True)
            ema_update()
            with torch.no_grad():
                x = fwd_G()
            D0.train().requires_grad_(True)
            with torch.autocast("cuda", DT):
                loss = D(gt, for_real=True).mean() + D(x, for_real=False).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(D_params, 1.0)
            D_opt.step(); D_opt.zero_grad(set_to_none=True)

        err = None
        try:
            for _ in range(args.warmup):
                one_iter()
            torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
            for _ in range(args.iters):
                one_iter()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.iters
        except Exception as e:
            dt, err = float("nan"), f"{type(e).__name__}: {str(e)[:200]}"
        if rank == 0:
            print("RES " + json.dumps(dict(cfg=cfg_name, ddp=not args.no_ddp, bs=bs,
                                           s_iter=round(dt, 4),
                                           peak_GB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                           err=err)), flush=True)
        del G, D, G_opt, D_opt, ema, ema_d
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
