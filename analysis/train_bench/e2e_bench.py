"""End-to-end G-step / D-step memory+time for HYPIR, and max-batch search.

Mirrors HYPIR/trainer/base.py::optimize_generator and sd2.py::forward_generator,
including accelerate's autocast placement (only prepared models G and D get
autocast + fp32 output cast; vae / net_lpips run outside autocast).
"""
import argparse, gc, json, time
import torch, torch.nn.functional as F

DEV = "cuda"
import mem_bench as MB


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


class Stack:
    def __init__(self, unet_ckpt=True, vae_ckpt=False, lpips_bf16=False, d_ckpt=False,
                 drop_text=True, ema_cpu=False):
        self.cfg = dict(unet_ckpt=unet_ckpt, vae_ckpt=vae_ckpt, lpips_bf16=lpips_bf16,
                        d_ckpt=d_ckpt)
        self.G = MB.build_unet(grad_ckpt=unet_ckpt)
        self.vae = MB.build_vae()
        if vae_ckpt is True:
            self.vae.enable_gradient_checkpointing()
            self.vae.decoder.gradient_checkpointing = True
        elif isinstance(vae_ckpt, (list, tuple)):
            # selective: checkpoint only the listed decoder up_blocks
            from torch.utils.checkpoint import checkpoint
            for i in vae_ckpt:
                blk = self.vae.decoder.up_blocks[i]
                blk._orig_forward = blk.forward
                blk.forward = (lambda b: (lambda *a, **k: checkpoint(
                    b._orig_forward, *a, use_reentrant=False, **k)))(blk)
        self.vgg = MB.build_lpips_like()
        self.D = MB.build_convnext_d()
        self.Dhead = torch.nn.Sequential(
            torch.nn.Linear(3072, 512), torch.nn.LeakyReLU(0.2), torch.nn.Linear(512, 1)
        ).to(DEV).float()          # stand-in for MultiLevelDConv (12.9M trainable)
        if d_ckpt:
            self.D.set_grad_checkpointing(True)
        self.lpips_bf16 = lpips_bf16
        self.Gp = [p for p in self.G.parameters() if p.requires_grad]
        self.Dp = list(self.Dhead.parameters())
        self.Gopt = torch.optim.AdamW(self.Gp, lr=1e-5)
        self.Dopt = torch.optim.AdamW(self.Dp, lr=1e-5)
        self.ema = [p.detach().clone().to("cpu" if ema_cpu else DEV) for p in self.Gp]

    def forward_generator(self, z_lq, t, ctx):
        with torch.autocast("cuda", torch.bfloat16):          # accelerate-prepared G
            eps = self.G(z_lq * 0.18215, t, encoder_hidden_states=ctx).sample
        eps = eps.float()                                     # convert_outputs_to_fp32
        z = (z_lq * 0.18215 - eps) * 0.5 + z_lq * 0.18215     # cheap stand-in for scheduler.step
        x = self.vae.decode(z.to(torch.bfloat16) / 0.18215).sample.float()
        return x

    def g_step(self, gt, z_lq, t, ctx):
        x = self.forward_generator(z_lq, t, ctx)
        loss = F.mse_loss(x, gt)
        if self.lpips_bf16:
            with torch.autocast("cuda", torch.bfloat16):
                lp = (self.vgg(x) - self.vgg(gt)).square().mean()
            loss = loss + 5 * lp.float()
        else:
            loss = loss + 5 * (self.vgg(x) - self.vgg(gt)).square().mean()
        with torch.autocast("cuda", torch.bfloat16):
            d = self.Dhead(self.D(x).float()).mean()
        loss = loss + 0.5 * d.float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Gp, 1.0)
        self.Gopt.step(); self.Gopt.zero_grad(set_to_none=True)

    def d_step(self, gt, z_lq, t, ctx):
        with torch.no_grad():
            x = self.forward_generator(z_lq, t, ctx)
        with torch.autocast("cuda", torch.bfloat16):
            lr_ = self.Dhead(self.D(gt).float()).mean()
            lf_ = self.Dhead(self.D(x).float()).mean()
        loss = (lr_ + lf_).float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dp, 1.0)
        self.Dopt.step(); self.Dopt.zero_grad(set_to_none=True)

    def ema_update_naive(self):
        for i, p in enumerate(self.Gp):
            self.ema[i] = 0.999 * self.ema[i] + 0.001 * p.clone().detach()

    def ema_update_fused(self):
        torch._foreach_lerp_(self.ema, self.Gp, 0.001)


def run(bs, res, cfg, iters=3, ema_cpu=False):
    s = Stack(ema_cpu=ema_cpu, **cfg)
    gt = torch.randn(bs, 3, res, res, device=DEV)
    lq = torch.randn(bs, 3, res, res, device=DEV, dtype=torch.bfloat16)
    t = torch.full((bs,), 200, device=DEV, dtype=torch.long)
    ctx = torch.randn(bs, 77, 1024, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        z_lq = s.vae.encode(lq).latent_dist.sample()

    out = {}
    for name, fn in [("G", s.g_step), ("D", s.d_step)]:
        fn(gt, z_lq, t, ctx)
        reset()
        fn(gt, z_lq, t, ctx)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn(gt, z_lq, t, ctx)
        torch.cuda.synchronize()
        out[name] = (peak, (time.perf_counter() - t0) / iters * 1000)
    # ema variants
    for name, fn in [("ema_naive", s.ema_update_naive), ("ema_foreach_lerp", s.ema_update_fused)]:
        if ema_cpu and name == "ema_foreach_lerp":
            continue
        fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        out[name] = (0.0, (time.perf_counter() - t0) / 5 * 1000)
    del s, gt, lq, ctx, z_lq
    reset()
    return out


CONFIGS = {
    "A_current":            dict(unet_ckpt=True,  vae_ckpt=False, lpips_bf16=False, d_ckpt=False),
    "B_+vae_ckpt":          dict(unet_ckpt=True,  vae_ckpt=True,  lpips_bf16=False, d_ckpt=False),
    "C_+vae+lpips_bf16":    dict(unet_ckpt=True,  vae_ckpt=True,  lpips_bf16=True,  d_ckpt=False),
    "D_+vae+lpips+D_ckpt":  dict(unet_ckpt=True,  vae_ckpt=True,  lpips_bf16=True,  d_ckpt=True),
    "E_no_unet_ckpt":       dict(unet_ckpt=False, vae_ckpt=True,  lpips_bf16=True,  d_ckpt=False),
    "F_no_ckpt_at_all":     dict(unet_ckpt=False, vae_ckpt=False, lpips_bf16=True,  d_ckpt=False),
    # selective VAE-decoder checkpointing: up_blocks[3] is 128ch@512^2, [2] is 256ch@256^2
    "G_vae_ckpt_last1":     dict(unet_ckpt=True,  vae_ckpt=(3,),   lpips_bf16=True, d_ckpt=False),
    "H_vae_ckpt_last2":     dict(unet_ckpt=True,  vae_ckpt=(2, 3), lpips_bf16=True, d_ckpt=False),
    "I_vaelast2_no_unetck": dict(unet_ckpt=False, vae_ckpt=(2, 3), lpips_bf16=True, d_ckpt=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[8])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--configs", type=str, default="all")
    ap.add_argument("--ema-cpu", action="store_true")
    a = ap.parse_args()
    names = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    res = {}
    for bs in a.bs:
        for n in names:
            try:
                r = run(bs, a.res, CONFIGS[n], ema_cpu=a.ema_cpu)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"bs={bs:3d} {n:24s} FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)
                reset(); continue
            res[f"{n}@bs{bs}"] = r
            gp, gt_ = r["G"]; dp, dt = r["D"]
            print(f"bs={bs:3d} {n:24s} G: {gp:7.2f} GB {gt_:7.1f} ms | "
                  f"D: {dp:6.2f} GB {dt:6.1f} ms | ema naive {r['ema_naive'][1]:.2f} ms"
                  + (f" foreach {r['ema_foreach_lerp'][1]:.2f} ms" if 'ema_foreach_lerp' in r else ""),
                  flush=True)
    print("JSON " + json.dumps({k: {kk: [round(v[0], 3), round(v[1], 2)] for kk, v in vv.items()}
                                for k, vv in res.items()}))


if __name__ == "__main__":
    main()
