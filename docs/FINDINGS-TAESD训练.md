# TAESD 版 HYPIR：训练与对拍结论

把 HYPIR-SD2 的冻结 SD-VAE 换成冻结的 TAESD（1.2M 参数，SD-VAE 是 83.7M），除此之外训练配方逐项对齐官方 `up/main`，只用 LoRA 微调 SD2.1 的 UNet。

## 一句话结论

**训练把「零样本换 TAESD」造成的画质损失完全补回来了，而推理快 2.04×。** 赛题综合分口径下等价于 +15.6%。

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据 | LSDIR 84,991 张 → 均匀铺满切 512×512，**414,938 个 patch**，prompt 全空（官方 README 的复现配方） |
| 退化 | 官方 `RealESRGANDataset` + `RealESRGANBatchTransform`，参数逐字照抄 `up/main:configs/sd2_train.yaml` |
| 精度 | bf16 混合精度，LoRA 参数 fp32；逐项对齐官方 |
| 初始化 | G 从官方 `HYPIR_sd2.pth`（514 个 LoRA 张量）、D 从 `HYPIR_sd2_D.safetensors`（38 个 decoder 张量） |
| latent 换算 | `z_tae = 0.16668·z_sd + 0.01702`，与提交侧 `runner.py` 同一组常量。在 DIV2K 上重拟合是 0.16857/0.01812（r=0.9643），差 1%，不改 |
| 评估 | StableSR 的 DIV2K_V2_val，3000 对（LQ 128 → GT 512，×4），`upscale=4 / patch_size=512 / stride=256 / seed=231` |
| 硬件 | hyper00 3×H200（lr 1e-5，有效 batch 24）／ hyper01 4×H200（lr 5e-6，有效 batch 32） |

## 2. 主结果（全量 3000 对，逐样本配对检验）

论文口径（7 项）：

| arm | PSNR | SSIM | LPIPS↓ | NIQE↓ | MUSIQ | MANIQA | CLIP-IQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 官方 LoRA + SD-VAE | 20.576 | 0.5423 | **0.3079** | 4.839 | 66.25 | 0.4776 | **0.6467** |
| B 官方 LoRA + TAESD（零样本） | 20.451 | 0.5446 | 0.3185 | **4.324** | 65.71 | 0.4519 | 0.6232 |
| C EMA@1000 | **20.689** | **0.5495** | 0.3125 | 4.603 | 67.39 | 0.4783 | 0.6296 |
| C blend 0.25 | 20.669 | 0.5465 | 0.3146 | 4.647 | 67.77 | 0.4879 | 0.6393 |
| C raw@1500 | 20.659 | 0.5400 | 0.3309 | 4.888 | **68.00** | **0.5027** | 0.6522 |

`EMA@1000` vs `A`：PSNR(t=5.5) / SSIM(t=12.1) / NIQE(t=−14.6) / MUSIQ(t=15.5) 四项显著更好，MANIQA 打平(t=0.56)，LPIPS(+0.0046, t=7.0) 与 CLIP-IQA(−0.0171, t=−10.7) 两项显著更差。

赛题口径（`感知分 = 13.674·TOPIQ-FR + 4.477·MANIQA − 5.731`，见 `FINDINGS-赛题逆向.md`），n=300：

| 权重 | TOPIQ-FR | MANIQA | 感知分 | Δ vs 官方 | t |
|---|---:|---:|---:|---:|---:|
| A 官方 + SD-VAE | 0.4243 | 0.4820 | 2.2293 | — | — |
| B 零样本 | 0.4181 | 0.4567 | 2.0312 | −0.1981 | −6.85 |
| **EMA@1000** | 0.4233 | 0.4828 | 2.2187 | −0.0105 | **−0.39（打平）** |
| **blend 0.25** | 0.4213 | 0.4927 | 2.2348 | +0.0055 | **+0.20（打平）** |
| raw@1500 | 0.4101 | 0.5060 | 2.1426 | −0.0867 | −2.63 |

综合分 = 感知分 × 加速比^0.204，而 TAESD 相对 SD-VAE 是 2.037× 加速（V100 单 tile 85.78 → 42.11 ms，`FINDINGS-P3训练.md`）：

```
综合分比值 = 1.002 × 2.037^0.204 = 1.158
```

**`raw@1500` 是个陷阱**：MANIQA 全场最高（0.5060）但 TOPIQ-FR 掉到 0.4101，而 TOPIQ-FR 权重是 MANIQA 的 3 倍，净结果显著劣于官方。只看无参考指标会选错。

## 3. 训练曲线：峰值在 750~1250 步，之后单调退化

250 步一存轻量快照（只 dump raw + EMA 两份 LoRA，2.0 GB／份，完整 `save_state` 是 8.5 GB）扫出来的。

`EMA@3000` vs `EMA@1000`：LPIPS(t=4.18) / NIQE(t=4.36) / MUSIQ(t=−3.55) / MANIQA(t=−2.49) / CLIP-IQA(t=−2.96) 五项全部显著退化，只有 PSNR 在涨。**跑到 20000 步只会产出更差的权重。**

峰值只消费 **0.12 个 epoch**（1000 步 × 24 样本 × 2 batch/step ÷ 414,938）—— 88% 的 patch 一次都没被看到。

## 4. 六条杠杆全部只在同一条一维曲线上滑动

这是本轮最重要的结构性发现。逐条实测：

| 杠杆 | 结果 |
|---|---|
| 快照步数 | 750~1250 是平台，之后退化 |
| EMA vs raw | raw 沿失真-感知轴振荡，相邻 250 步 CLIP-IQA 摆动 0.033（比要补的缺口还大）；PSNR/SSIM 与 MANIQA/CLIP-IQA **完全反相** |
| 权重插值 λ（EMA↔raw） | 四项指标沿 λ 完全单调，汇率从 10.3 单调降到 0.69。**可行域为空**：LPIPS 要 λ<0.31、CLIP-IQA 要 λ>0.39 |
| patch_size（内部工作分辨率） | 扣掉重采样往返的税后模型净收益为正（MUSIQ +1.24 / MANIQA +0.0134 / LPIPS −0.018），但输出尺寸固定时净效果仍是五项好两项差 |
| **EMA 去偏** | **假设被证伪**。`ema.py` 用 θ₀ 初始化且无 bias correction，EMA@1000 里有 36.8% 是 θ₀=arm B。但去偏后 CLIP-IQA 反而掉 0.0106 —— θ₀ 的 CLIP-IQA(0.6279) 比真实迭代点的凸组合(0.6209) 还高，训练轨迹本身朝保真端走 |
| **学习率减半** | **没有把前沿推出去**。按 lr×batch 对齐后，lr 5e-6 的 LPIPS 好 0.0009，但 MUSIQ 差 0.54 / MANIQA 差 0.009 / CLIP-IQA 差 0.012。曲线更平缓、不塌陷，但峰值更低 |

**结论：约束不在这些旋钮上，而在训练目标本身。**

## 5. 下一个该做的实验

`HYPIR/model/D.py:26-28` 的 `if for_G: for_real = True` 让 **G 的对抗目标也被 label smoothing 成 0.8**。于是 G 的对抗梯度 `dL/dl = σ(l) − 0.8` 在 `l = logit(0.8) = 1.386` 处过零反号 —— **G 一旦被判「够真」，对抗项会主动把它推回去变差**。标准的 one-sided label smoothing（Salimans 2016）只平滑 D 的真样本、G 的目标保持 1.0。

这是三行改动，且是唯一一条声称能同时抬高峰值**并**压小振荡的杠杆（加 λ_gan 会让振荡变大）。上游 `vision_aided_loss` 就是这个行为，所以属于偏离官方。

## 6. 已排除（实测或对抗验证）

- **加数据**：零收益。峰值只跑 0.12 epoch，88% 的 patch 没被看过
- **`use_rot: true`**：同理，收益精确为零
- **`crop_type: random`**：有害。`random_crop_arr` 会先把 GT 缩到短边 ∈[512,732] 再裁，LSDIR 中位 1224×816 意味着被 bicubic 降采样 1.11~1.59 倍
- **ColorJitter / gamma / 亮度对比度**：近似空操作。`enhancer/base.py:153` 的 wavelet 直接用 LQ 低频替换输出低频（等效 σ=13.06 px），输出的色调亮度完全由 LQ 决定
- **通道 shuffle / MixUp / CutMix**：造非自然统计，冲击冻结的 CLIP-ConvNeXt 判别器与 SD2.1 先验
- **降 LoRA rank / SVD 重排**：同 r=256 的 P3 训练能撑 6000~8000 步无回落，说明 1000 步见顶不是容量问题；其机理不过是变相调小 lr
- **精度类改动**（训练 TAESD 提 fp32 / 换 fp16 / 关 TF32 / tiled 缓冲钉 fp32 / D 骨干改回 bf16）：六条全部被驳回，详见 workflow 报告。当前精度 100% 对齐官方
- **VLM caption**：`csig/prompt.py` 的零样本消融显示画质形容词越多，感知指标单调下降

## 7. 一处已知的偏离官方

`base.py` 用 `ImageConvNextDiscriminator(precision="fp32")`，官方是 `bf16`。实测各 level 特征相对误差 **1.4%~3.2%**（ConvNeXt 的 LayerScale `gamma` 不在 autocast 转换列表里，fp32 下会把整条残差流类型提升回 fp32）。**不是数值等价**。维持现状的理由：省下的 3.12 GB 在 60/141 GB 余量下换不到任何 batch 档位，中途改会让已扫快照曲线不可比。

## 8. 工具

| 脚本 | 用途 |
|---|---|
| `csig/prep_lsdir.py` | LSDIR → 均匀铺满的 512 patch + 官方格式 parquet。确定性，两台机器产出逐位一致 |
| `csig/eval_bench.py` | DIV2K_V2_val 上的对拍，8 项指标（含 topiq_fr），支持 `--patch-size/--stride` |
| `csig/sweep_snapshots.sh` | 多卡并行扫快照，`KIND`/`STRIDE`/`MAX_STEP` 三个开关，已评过的跳过 |
| `csig/bench_table.py` | 汇总出表 |
| `csig/blend_lora.py` | 两份 LoRA 权重线性插值 |
| `csig/debias_ema.py` | EMA bias correction（结论是负向，保留供复现） |
