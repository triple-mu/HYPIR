"""小模型 DDP 语义验证：复刻 HYPIR 的 G/D 交替 + 在 G-step 里以 requires_grad_(False)
前向 DDP 包裹的 D。检查 (a) 会不会报错 (b) 梯度是否真的被 all-reduce 同步。

torchrun --nproc_per_node=2 bench_ddp_semantics.py
"""
import json
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

CFGS = {
    "default": dict(),
    "gbv": dict(gradient_as_bucket_view=True),
    "bcast_off": dict(broadcast_buffers=False),
    "fup": dict(find_unused_parameters=True),
    "static": dict(static_graph=True),
}


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(32, 32)
        self.frozen.requires_grad_(False)
        self.lora = nn.Linear(32, 32)

    def forward(self, x):
        return self.frozen(x) + self.lora(x)


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 32)
        self.backbone.requires_grad_(False)
        self.register_buffer("bn", torch.zeros(32))
        self.head = nn.Linear(32, 1)

    def train(self, mode=True):
        self.head.train(mode); return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, rg=True):
        self.head.requires_grad_(rg); return self

    def forward(self, x):
        self.bn.add_(1.0)  # 模拟 spectral_norm 的就地 buffer 更新
        return self.head(self.backbone(x))


def grads_in_sync(params, tag):
    """所有 rank 的梯度是否逐元素一致 -> 说明 all-reduce 真的发生了"""
    flat = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
    gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    return bool(torch.allclose(gathered[0], gathered[1], atol=0, rtol=0)), round(flat.abs().sum().item(), 6)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(0)

    for name, kw in CFGS.items():
        torch.manual_seed(0)
        g0, d0 = G().cuda(), D().cuda()
        try:
            g = DDP(g0, device_ids=[torch.cuda.current_device()], **kw)
            d = DDP(d0, device_ids=[torch.cuda.current_device()], **kw)
            gp = [p for p in g0.parameters() if p.requires_grad]
            dp = [p for p in d0.parameters() if p.requires_grad]
            res = {}
            for it in range(3):
                # ---- G step: D 被 requires_grad_(False) 后仍然过 DDP 前向 ----
                d0.eval().requires_grad_(False)
                x = torch.randn(4, 32, device="cuda") * (rank + 1)  # 各 rank 数据不同
                loss = d(g(x)).mean()
                loss.backward()
                res[f"G_sync_it{it}"] = grads_in_sync(gp, "G")
                for p in gp:
                    p.grad = None
                # ---- D step ----
                d0.train().requires_grad_(True)
                with torch.no_grad():
                    y = g(torch.randn(4, 32, device="cuda") * (rank + 1))
                loss = d(y).mean()
                loss.backward()
                res[f"D_sync_it{it}"] = grads_in_sync(dp, "D")
                for p in dp:
                    p.grad = None
            out = dict(cfg=name, ok=True, **res, buf=round(d0.bn[0].item(), 1))
        except Exception as e:
            out = dict(cfg=name, ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
        if rank == 0:
            print("RES " + json.dumps(out), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
