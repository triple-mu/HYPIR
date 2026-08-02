"""Gradient noise scale (McCandlish et al. 2018) for HYPIR LoRA G and D at a live checkpoint.

B_simple = tr(Sigma) / |G|^2 ; two-point estimator from micro-batch (b) and full-batch (B) grads.
Also reports the Adam-preconditioned variant (the one that actually governs Adam's critical batch).
"""
import os, sys, json, math, argparse
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/root/.cache/huggingface/csig/code/HYPIR")
sys.path.insert(0, "/root/.cache/huggingface/csig/code/tools")
import torch.nn.functional as F
from HYPIR.trainer.sd2 import SD2Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/root/.cache/huggingface/csig/out/p2_main/checkpoint-6000")
ap.add_argument("--b", type=int, default=4)
ap.add_argument("--nmicro", type=int, default=8)
ap.add_argument("--outer", type=int, default=16)
ap.add_argument("--out", default="/tmp/gns_result.json")
a = ap.parse_args()

cfg = OmegaConf.load("/root/.cache/huggingface/csig/code/HYPIR/configs/csig_train.yaml")
cfg.data_config.train.batch_size = a.b
cfg.data_config.train.dataloader_num_workers = 4
cfg.gradient_accumulation_steps = 1
cfg.use_ema = False
cfg.resume_from_checkpoint = None
cfg.output_dir = "/tmp/gns_out"
cfg.report_to = []

tr = SD2Trainer(cfg)
tr.attach_accelerator_hooks()
tr.force_optimizer_ckpt_safe(a.ckpt)
tr.accelerator.load_state(a.ckpt)
print("loaded", a.ckpt, flush=True)

dev = tr.device
G_params = [p for p in tr.G.parameters() if p.requires_grad]
D_params = [p for p in tr.D.parameters() if p.requires_grad]

def flat(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])

def precond(opt, params):
    """1/(sqrt(v_hat)+eps) in Adam's update metric; ones if state empty."""
    out = []
    inner = opt.optimizer if hasattr(opt, "optimizer") else opt
    for p in params:
        st = inner.state.get(p, {})
        if "exp_avg_sq" in st:
            step = st["step"]
            step = step.item() if torch.is_tensor(step) else step
            b2 = inner.param_groups[0]["betas"][1]
            bc2 = 1 - b2 ** max(step, 1)
            out.append((1.0 / (st["exp_avg_sq"].float().div(bc2).sqrt() + 1e-8)).reshape(-1))
        else:
            out.append(torch.ones(p.numel(), device=p.device))
    return torch.cat(out)

pre_G = precond(tr.G_opt, G_params)
pre_D = precond(tr.D_opt, D_params)
print("precond G: median %.3e  D: median %.3e" % (pre_G.median().item(), pre_D.median().item()), flush=True)

it = iter(tr.dataloader)
def nxt():
    global it
    try:
        return next(it)
    except StopIteration:
        it = iter(tr.dataloader); return next(it)

def grad_G():
    tr.unwrap_model(tr.D).eval().requires_grad_(False)
    tr.prepare_batch_inputs(nxt())
    x = tr.forward_generator()
    loss = (F.mse_loss(x, tr.batch_inputs.gt) * cfg.lambda_l2
            + tr.net_lpips(x, tr.batch_inputs.gt).mean() * cfg.lambda_lpips
            + tr.D(x, for_G=True).mean() * cfg.lambda_gan)
    tr.G_opt.zero_grad(set_to_none=True)
    loss.backward()
    return flat(G_params), loss.item()

def grad_D():
    tr.prepare_batch_inputs(nxt())
    with torch.no_grad():
        x = tr.forward_generator()
    tr.unwrap_model(tr.D).train().requires_grad_(True)
    loss = tr.D(tr.batch_inputs.gt, for_real=True).mean() + tr.D(x, for_real=False).mean()
    tr.D_opt.zero_grad(set_to_none=True)
    loss.backward()
    return flat(D_params), loss.item()

def run(name, gfn, pre, params):
    b, B = a.b, a.b * a.nmicro
    rows = []
    for o in range(a.outer):
        acc = torch.zeros(sum(p.numel() for p in params), device=dev)
        accp = torch.zeros_like(acc)
        sq = []; sqp = []; norms = []
        for j in range(a.nmicro):
            g, _ = gfn()
            gp = g * pre
            acc += g; accp += gp
            sq.append(g.pow(2).sum().item()); sqp.append(gp.pow(2).sum().item())
            norms.append(math.sqrt(sq[-1]))
        gB = acc / a.nmicro; gBp = accp / a.nmicro
        s_b = sum(sq) / len(sq); s_B = gB.pow(2).sum().item()
        p_b = sum(sqp) / len(sqp); p_B = gBp.pow(2).sum().item()
        rows.append((s_b, s_B, p_b, p_B, sum(norms) / len(norms), math.sqrt(s_B)))
        print("  [%s] outer %2d  |g_b|^2=%.4e |g_B|^2=%.4e  (pre) %.4e %.4e" % (name, o, s_b, s_B, p_b, p_B), flush=True)
        del acc, accp, gB, gBp
        torch.cuda.empty_cache()

    def est(ib, iB):
        # average numerator/denominator separately, then ratio (paper's recommendation)
        Gn = sum((B * r[iB] - b * r[ib]) / (B - b) for r in rows) / len(rows)
        S = sum((r[ib] - r[iB]) / (1.0 / b - 1.0 / B) for r in rows) / len(rows)
        return Gn, S, (S / Gn if Gn > 0 else float("nan"))
    g1, s1, bs1 = est(0, 1)
    g2, s2, bs2 = est(2, 3)
    mn_b = sum(r[4] for r in rows) / len(rows); mn_B = sum(r[5] for r in rows) / len(rows)
    print("[%s] b=%d B=%d  |G|^2=%.4e trSigma=%.4e  B_simple=%.1f | preconditioned B_simple=%.1f"
          % (name, b, B, g1, s1, bs1, bs2), flush=True)
    print("[%s] mean grad norm: micro(b=%d) %.4f  full(B=%d) %.4f" % (name, b, mn_b, B, mn_B), flush=True)
    return dict(name=name, b=b, B=B, G2=g1, trSigma=s1, B_simple=bs1, B_simple_pre=bs2,
                norm_b=mn_b, norm_B=mn_B, rows=rows)

res = {}
res["G"] = run("G", grad_G, pre_G, G_params)
del pre_G; torch.cuda.empty_cache()
res["D"] = run("D", grad_D, pre_D, D_params)
json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
