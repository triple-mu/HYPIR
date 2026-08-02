"""nsys / ncu 的采样载荷：预热到稳态后用 cudaProfilerStart/Stop 只圈住有代表性的几次迭代。

不加这个开关的话，profiler 会把加载权重、cudnn.benchmark 算法试跑、Triton JIT 编译、
后端选型的那几十秒全录进去，稳态 kernel 淹没在里面。这里的做法是：
    1. Runner.__init__ 已经做完后端选型与预热
    2. 再空跑 --warmup 次让 cudnn 算法缓存与显存池彻底稳定
    3. torch.cuda.synchronize() 后 cudaProfilerStart()，只跑 --iters 次
    4. 再 synchronize() 后 cudaProfilerStop()，保证窗口里没有未完成的异步工作

NVTX 区间用 module forward hook 打，不改 runner.py，圈的是真实的 infer 路径。

模式：
    infer   真实评测路径 Runner.infer(512x512)——时延分就是量这个
    tile    只跑 _infer_tile（去掉 wavelet 与前后处理）
    graph   把 _infer_tile 捕成 CUDA Graph 后 replay，用来量图捕获到底能省多少

用法见 csig_bench/profile.sh；单独跑（不带 profiler）也能出墙钟数，便于对照。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# 打 NVTX 的模块类型：粗粒度看三段，细粒度再看到每个 block
COARSE = ("VAEEncoder", "UNet2DConditionModelLite", "VAEDecoder")
FINE = COARSE + ("ResnetBlock2D", "VAEAttention", "Transformer2DModel", "VAEMidBlock",
                 "Upsample2D", "Downsample2D")


def add_nvtx(runner, level: str) -> None:
    """给选定类型的 module 挂 forward pre/post hook，在 nsys 时间线上画出区间。

    两个 hook 都必须返回 None：pre_hook 的返回值会被当成替换后的输入，forward_hook 的
    返回值会被当成替换后的输出，而 range_push/pop 返回的是嵌套层数（int）。
    """
    if level == "off":
        return
    wanted = FINE if level == "fine" else COARSE

    def push(tag):
        def hook(_m, _i):
            torch.cuda.nvtx.range_push(tag)
        return hook

    def pop(_m, _i, _o):
        torch.cuda.nvtx.range_pop()

    seen = {}
    for model in (runner.vae, runner.unet):
        for _, m in model.named_modules():
            cls = type(m).__name__
            if cls not in wanted:
                continue
            seen[cls] = seen.get(cls, 0) + 1
            m.register_forward_pre_hook(push(f"{cls}#{seen[cls]}" if level == "fine" else cls))
            m.register_forward_hook(pop)
    print(f"NVTX({level}): {sum(seen.values())} 个区间 {dict(seen)}")


def make_graph_runner(runner, x):
    """把 _infer_tile 捕成 CUDA Graph，返回 (replay 函数, 静态输出)。

    捕获前必须在侧流上跑几次：cudnn.benchmark 的算法试跑、cuBLAS workspace 分配、
    Triton 的 launch metadata 都不能发生在捕获期内。
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    static_in = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                                 SDPBackend.EFFICIENT_ATTENTION]):
        for _ in range(5):
            runner._infer_tile(static_in)
    torch.cuda.current_stream().wait_stream(stream)

    g = torch.cuda.CUDAGraph()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with torch.cuda.graph(g):
            static_out = runner._infer_tile(static_in)

    def replay(src):
        static_in.copy_(src)
        g.replay()
        return static_out

    return replay, static_out


def timeit(fn, n: int) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description="nsys/ncu 采样载荷")
    ap.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "model_dir"))
    ap.add_argument("--backend", type=str, default="", help="覆盖 CSIG_OP_BACKEND")
    ap.add_argument("--mode", type=str, default="infer", choices=["infer", "tile", "graph"])
    ap.add_argument("--nvtx", type=str, default="coarse", choices=["off", "coarse", "fine"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3, help="落在 profiler 窗口内的迭代数")
    args = ap.parse_args()

    if args.backend:
        os.environ["CSIG_OP_BACKEND"] = args.backend
    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import csig_ops
    from runner import Runner

    t0 = time.time()
    runner = Runner(args.model_dir)
    size = runner.patch_size
    print(f"{torch.cuda.get_device_name(0)} | 后端 {csig_ops.BACKEND} "
          f"channels_last={runner.channels_last} | 加载 {time.time() - t0:.1f}s")

    x = torch.randn(1, 3, size, size, device=runner.device).clamp(-1, 1)
    if args.mode == "infer":
        run = lambda: runner.infer(x)
    else:
        xf = x.to(runner.weight_dtype)
        if runner._mf:
            xf = xf.contiguous(memory_format=runner._mf)
        if args.mode == "tile":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def run():
                # sdpa_kernel 的上下文对象是一次性的，每次调用都得新建
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    return runner._infer_tile(xf)
        else:
            replay, _ = make_graph_runner(runner, xf)
            run = lambda: replay(xf)

    # NVTX hook 放在 graph 捕获之后：range_push/pop 是主机侧调用，捕不进图，
    # 挂早了会在捕获期打出无意义的区间
    add_nvtx(runner, args.nvtx if args.mode != "graph" else "off")

    print(f"预热 {args.warmup} 次 …", flush=True)
    print(f"  稳态墙钟 {timeit(run, args.warmup):.2f} ms/iter")

    cudart = torch.cuda.cudart()
    torch.cuda.synchronize()
    cudart.cudaProfilerStart()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"{args.mode}#{i}")
        run()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    print(f"已采样 {args.iters} 次 {args.mode}")


if __name__ == "__main__":
    main()
