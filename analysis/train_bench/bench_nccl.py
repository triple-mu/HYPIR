"""NCCL allreduce 微基准：在 NCCL_NVLS_ENABLE=0（ring/tree）下测 HYPIR 实际梯度规模的耗时。

torchrun --nproc_per_node=N bench_nccl.py
"""
import json, os, time
import torch
import torch.distributed as dist

SIZES = [
    ("D_grad_fp32", 12_914_692, torch.float32),
    ("G_grad_bf16", 259_474_432, torch.bfloat16),
    ("G_grad_fp32", 259_474_432, torch.float32),
]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    out = []
    for name, numel, dt in SIZES:
        x = torch.randn(numel, dtype=dt, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt_s = (time.perf_counter() - t0) / n
        nbytes = numel * x.element_size()
        # ring allreduce bus bandwidth 定义: 2*(N-1)/N * S / t
        busbw = 2 * (world - 1) / world * nbytes / dt_s / 1e9
        if rank == 0:
            out.append(dict(op=name, world=world, MB=round(nbytes / 2**20, 1),
                            ms=round(dt_s * 1e3, 3), busbw_GBps=round(busbw, 1)))
            print("RES " + json.dumps(out[-1]), flush=True)
        del x
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
