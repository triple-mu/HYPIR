# CSIG-2026 赛道二 · 工作区说明

本 fork 在上游 HYPIR（`b61d107`）之上加了 CSIG-2026 的全部工作，分支 `csig-2026`。
**这是本赛题唯一的活跃仓库** —— 训练侧与提交侧都在这里，见 §1。
权重与数据不进 git（见 §6），只放代码、配置与结论文档。

---

## 1. 目录

| 目录 | 内容 |
|---|---|
| `HYPIR/` | 上游代码 + 我们的改动（见 §2），所有改动都带 `[csig-speedup]` / `[csig-compile]` 标记 |
| `configs/csig_train.yaml` | 训练配置 |
| `csig/` | **训练与数据管线**：退化配方、评估、数据准备、看门狗、环境脚本 |
| `submit/` | **推理提交版**：单文件 runner + CUDA 扩展源码 + 打包驱动，见 §3 |
| `analysis/` | 一次性的分析脚本（赛题逆向、取证、算子基准、数据源调研），**不参与训练** |
| `docs/` | 结论文档与赛题原件，见 §5 |

工作区里 fork 之外只剩两样东西，都不进 git：

- `../赛题二/` —— 赛题数据（100 张测试集无 GT + 验证集 GT/LQ 对）
- `../csig/` —— **py3.9 + torch 2.5.1+cu118 的 venv，对齐评测机**。
  用来给评测环境编译 `custom_op`，也是做 `torch.jit` 导出唯一能用的环境。
  它的 `bin/` 里写死了容器路径 `/workspace/csig/bin/python3`（宿主机上是断链），
  **只能在 `csig2026` 容器内用**，别移动别改名。

## 2. 对上游 HYPIR 的改动

**数据管线**（`HYPIR/dataset/csig.py`，替换 RealESRGAN）
- 退化按实测重建：真实 LQ **无噪声**（σ 0.014–0.090 灰阶）、恒 q95、重尾各向异性核，
  感知强度约 **7× 重采样**而非此前认为的 2×。原版一半样本带 σ≤30 强噪，
  模型学到的激进降噪在无噪输入上只会吃掉细节。
- 解码搬 GPU：worker 只读字节（0.01 ms/张），nvJPEG 批量解码+裁剪（batch 24 时 3.5 ms/张）。
  原版 CPU 全解 4K 图 100 ms/张，4 worker 仅 27 样本/秒，是硬瓶颈。

**训练加速**（`HYPIR/trainer/{base,sd2}.py`、`HYPIR/utils/{ema,compile_patch}.py`）
- `torch.compile`：G / vae.encoder+decoder / net_lpips / D.model.encode_image
- D 步复用 G 步输出，不再重跑 `forward_generator`（G/D 交替更新，D 无条件不需配对）
- 固定 prompt 的 text embedding 缓存（省每步一次 340M CLIP 前向）
- EMA 换 `torch._foreach_lerp_`（21 ms → 0.85 ms/步）
- 关梯度检查点、fused AdamW、DDP `gradient_as_bucket_view`
- 支持从已发布的 `HYPIR_sd2.pth` / `HYPIR_sd2_D.safetensors` 续训

改动的完整清单与实测依据见 `csig/apply_speedups.py` 的文件头。

**几条踩过的坑，改前务必先读**
- `unwrap_model` 必须 `keep_torch_compile=False` —— 否则 ckpt/EMA 的 key 多出
  `_orig_mod.` 前缀，**resume 静默失效不报错**
- **不要开 `static_graph=True`** —— 首步是 G 步，此时 D 全部 `requires_grad=False`，
  会被永久标记「不产生梯度」，之后所有 D 步都不 allreduce，两卡各练各的，不报错
- **不要开 `channels_last`** —— 实测 VAE encode 慢 25%、UNet 前反慢 19%
- `open_clip==2.31.0` / `timm==1.0.15` 必须 pin：新版改了 ConvNext 特征抽取，
  判别器 decoder 通道变成 384（期望 768），四个 rank 同时炸在第一个 D 步，
  报错离根因很远。两台机器的 venv 各装各的，`csig/env.sh` 里已加启动前硬校验

## 3. `submit/` 推理提交版

赛题要求提交 `model_dir/{模型文件, runner.py}`（原始要求见 `docs/说明.md`，
主办方给的接口基准见 `docs/官方参考-runner.py`）。

```
submit/
├── main.py          批量推理 + 打包驱动（--pack 出 MMDDHHMM_描述.zip，便于把线上分数对回版本）
├── model_dir/       直接提交的目录：runner.py / csig_ops.py / config.yaml / tokenizer/
│                    （hypir_weights.pth 与编译好的 custom_op/ 不入库，见 .gitignore 里的重建命令）
├── csig_ops_ext/    自定义 CUDA 算子源码：attn_d512 / geglu / group_norm_act / sample_latent
├── tests/           算子单测 + Moebius 移植的等价性验证
└── moebius/         Moebius 0.22B 的参考实现（单文件 runner，只依赖 torch）
```

| | 基座 | 说明 |
|---|---|---|
| `model_dir/` | SD2.1 + HYPIR LoRA | **当前主线**。内联了 SD2.1 全部定义（CLIP tokenizer/text encoder/UNet/VAE），依赖 torch + numpy + yaml + `csig_ops` |
| `moebius/` | Moebius 0.22B | 已弃用（画质不及 HYPIR），留作参考。CPU float64 逐层对拍 **296.6 dB** 通过 |

两者接口一致：`Runner(model_dir)` / `infer(image_tensor, prompt="")` / `enhance(image_tensor)`。
`infer` 吃 512×512 分块，`enhance` 吃整图、滑窗后逐块回调 `infer`。

⚠️ **尚未做 `torch.jit` 导出**（说明.md 第 2 条要求）。需在 `../csig/` 那个 py39 venv 里做，
因为用新版 torch 导出的 `.pt` 在评测机上很可能 load 失败。

## 4. `analysis/` 一次性分析

| 目录 | 内容 |
|---|---|
| `probe/` | 探测包构建器 —— **评分公式就是用它逆向出来的**：构造只改一个变量的提交包（恒等/定时/USM/LQ 原样），拿线上分反解 |
| `forensics/` | JPEG 字节级取证：量化表匹配、双压检测、块效应、频谱天花板 |
| `exif/` | EXIF/MakerNote/MPF/ICC/增益图解析，判定 LQ 的相机来源与后处理链 |
| `classic/` | 经典算子路线的完整证伪：去噪/去模糊/滤波器搜索/JPEG 重编码验证 |
| `tone/` | 色调与 gainmap 的 27 组对照实验 |
| `pd12m/` `commons/` | 训练数据源调研与采集 |
| `bench/` `train_bench/` | 推理算子基准 / 训练侧基准（DDP、NCCL、显存、精度） |
| `refs/` | Triton 参考实现 |

## 5. `docs/`

| 文件 | 要点 |
|---|---|
| `FINDINGS-赛题逆向.md` | **评分公式已解出**：`综合分 = 感知分 × 加速比^0.204`（画质线性进分，速度只有 0.2 次方）。任务本质是伪装成 1× 的盲超分。只有 TOPIQ-FR + MANIQA 可信 |
| `FINDINGS-Moebius移植.md` | 单文件移植、等价性验证、画质天花板（GT p_rel 8.34，现基线只吃到 22.5%） |
| `CHANGELOG-提交侧.md` | 原 `triple-mu/CSIG-2026` 仓库的提交记录（该仓库因 2.46 GB 权重进了历史推不上 GitHub，源码已并入本仓库） |
| `说明.md` / `官方参考-runner.py` | 赛题原始要求与接口基准 |
| `csig/README.md` | 数据集复现：三闸门标定、各源实测产出、许可与法律红线 |

## 6. 不在 git 里的东西（在机器上）

```
/root/.cache/huggingface/csig/          （容器内路径，宿主机 /data04/cache/huggingface/csig）
├── weights/     SD2.1-base + HYPIR_sd2.pth + _D.safetensors      6 GB
├── data/        4klsdb_hr 5683 / pd12m_hr 9249 / hdrplus 3640 /
│                shopsign_hr 288 / eval_p0 177对 / lists/         114 GB
├── out/         训练输出与 checkpoint（一份 8.5 GB，500 步一存）
└── logs/
```

两台机器（hyper00 / hyper01）都有同一份，已打通免密直连（同私网，实测 ~800 MB/s）。
**两边是共享机器**，起训练前看门狗会自动挑连续 3 次采样都空闲的卡。

## 7. 常用命令

```bash
# 训练（自动选空闲 GPU；被杀后从最近 checkpoint 续训；连续 3 次秒退会判定为配置错误并停止）
cd /workspace/csig/HYPIR && NGPU=4 bash csig/run_train.sh

# 评估某份权重（177 对 held-out，指标是保留增益）
source csig/env.sh && python csig/eval_p0.py $CSIG/out/p3_main/checkpoint-N/ema_state_dict.pth

# 重建数据集
bash csig/run_all.sh /path/to/data_root

# 推理 + 打包提交（在 csig2026 容器内，用 ../csig 那个 py39 venv）
python submit/main.py --input-dir ../赛题二/测试集
python submit/main.py --pack --desc cudaop_87ms
```

**评估刻度**：未微调 HYPIR **19.0%** → P2（换退化配方 + 关 sharpener）5000 步 **44.0%** / 7000 步 **45.6%**。
