# custom_op_cpp_ext

把 `model_dir/runner.py` 里四个 Triton 融合算子改写成 CUDA C++ 扩展，用于逐算子对拍与性能对比。
布局与代码风格对齐 `CTI-2026/custom_op_cpp_ext`（CMake 构建 + `TORCH_LIBRARY` 注册 + `ops.py` 的
`register_fake` 包装）。

**本扩展只用于基准实验，不进提交包**——`model_dir/runner.py` 一行未改，端到端对比走
`ops_backend.patch_runner()` 的 monkeypatch。

## 编译

```bash
# 本地 RTX 3080 Ti (sm86)
CSIG_CUDA_ARCH_LIST="8.6+PTX" CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  pip install -e custom_op_cpp_ext --no-build-isolation

# 服务器 Tesla V100 (sm70)
CSIG_CUDA_ARCH_LIST="7.0+PTX" CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  /root/autodl-tmp/csig/bin/pip install -e custom_op_cpp_ext --no-build-isolation
```

四个算子全部经 `TORCH_LIBRARY` 注册、没有 pybind11 绑定，故 `.so` 里没有 `PyInit__C`，
`__init__.py` 用 `torch.ops.load_library` 而不是 `import _C`。

## 测试与基准

```bash
pytest tests/ -q                       # 对 fp32 eager 参考 + 对 Triton 双向对拍
python csig_bench/op_bench.py          # 逐算子逐形状 eager / triton / custom 计时
python csig_bench/op_bench.py --e2e    # 512x512 端到端注入对比
```

## 实测结果（Tesla V100-PCIE-32GB, sm70, fp16）

逐算子相对 Triton 的加速比：

| 算子 | 加速比 | 备注 |
|---|---|---|
| `geglu` | 1.02 – 5.04x | 纯访存，向量化 |
| `sample_latent` | 3.15 – 3.63x | |
| `attn_d512` | 1.03 – 3.90x | 且 Triton 版在 sm70 上是错的，见下 |
| `group_norm_act` | 0.70 – 3.86x | 仅最大的两个形状（C≥256, HW≥65536）略慢 |

端到端 512×512（`Runner.infer`，30 次均值）：

| 后端 | 时延 | vs eager（基准真值）PSNR |
|---|---:|---:|
| eager | 182.42 ms | — |
| triton | 89.37 ms | **27.32 dB** |
| custom | **86.48 ms** | **42.55 dB** |

## 已知问题：Triton 的 attn_d512 在 Volta 上是坏的

`triton_attn_d512`（Triton 3.1.0，sm70）的输出与 fp32 SDPA 参考的最大误差达 **0.13 ~ 0.78**，
而输出本身量级只有 0.03——即结果基本无效。逐 N 实测：

| N | custom vs fp32 参考 | triton vs fp32 参考 |
|---|---|---|
| 64 | 8.7e-4 | 7.8e-1 |
| 256 | 4.7e-4 | 5.3e-1 |
| 1024 | 3.6e-4 | 2.8e-1 |
| 4096 | 1.1e-4 | 1.3e-1 |

触发条件在推理链路里是满足的（`VAEAttention` 中 `ENABLE_TRITON and c==512 and dtype==fp16`），
所以在 Volta 上跑 `model_dir/runner.py` 会得到被污染的 VAE mid-block attention，端到端 PSNR
从 42.55 掉到 27.32 dB。sm86 上无此问题。

`tests/test_attn_d512.py::test_agrees_with_triton` 会检出这种情况并跳过（附实测误差），
真正的正确性判据是 `test_matches_fp32_sdpa`。

## 实现要点

- **`attn_d512` 走 cuBLAS GEMM ×2 + 融合 softmax，而非 flash**。Volta 没有 `cp.async`、也没有
  `mma.16816`；flash 要把 `BLOCK_M × 512` 的 fp32 累加器留在寄存器里，sm70 上会把 occupancy
  压死。N=4096 时物化分数矩阵只要 33.5 MB，两次 GEMM 交给 cuBLAS 更划算——实测正好印证。
- **`group_norm_act` 的 stats 按 (行, 组内通道) 二维铺开线程**。只按 `cpg` 分线程时，`cpg=4`
  （C=128）会只有 4/256 个线程在干活，实测比 Triton 慢 50 倍。
- `split` 的两个常数是 arch 敏感的，留了 env 覆盖便于在目标机上扫参：
  `CSIG_GN_SPLIT`（直接锁定）、`CSIG_GN_ELEMS_PER_BLOCK`、`CSIG_GN_BLOCKS_PER_SM`。
- 全部 Python 保持 **3.9 兼容**（服务器是 3.9.25，PEP 604 的 `X | Y` 会在函数定义时报错）。
