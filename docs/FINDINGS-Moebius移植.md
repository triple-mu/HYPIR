# Moebius 移植：结论与实测

承接上级 `../FINDINGS.md`（评分公式 `Score = 感知分 × 加速比^0.204` 等结论）。本文只记本次
Moebius 移植新得到的东西。测试环境：RTX 3080 Ti Laptop (sm86)，fp16，`cudnn.benchmark=True`。

---

## 一、交付物

```
Moebius/
├── model_dir/
│   ├── runner.py              单文件推理脚本（约 790 行），只依赖 torch
│   ├── moebius_weights.pth    fp16 单文件权重，611 MB（HYPIR 那份是 2.58 GB）
│   └── config.json            推理超参
└── tools/
    ├── verify_equivalence.py  CPU float64 逐层对拍
    ├── probe_zero_shot.py     零训练可行性探针
    ├── iqa.py                 TOPIQ-FR / MANIQA 封装
    └── probe_results.csv      探针全网格结果
```

`runner.py` 里没有 diffusers / transformers / timm / einops / fla / numpy / yaml / cv2。
Moebius 本来就没有 text encoder（条件是一张 20×3072 的 ID 表），且该表在导出期已被吸收，
所以连 tokenizer 都不需要——这是相对 HYPIR 方案最干净的一点。

接口契约与官方样例一致：`Runner.__init__(model_dir)` / `infer(image_tensor, prompt="")` /
`enhance(image_tensor)`。`infer` 吃 512×512 分块，`enhance` 吃整图、滑窗后**逐 tile 回调
`infer`**，两条路径共用同一段模型链（实测 `enhance(512)` 与 `infer(512)` 逐位相同）。

---

## 二、等价性：一次通过，机器精度

`tools/verify_equivalence.py`，CPU float64、`allow_tf32=False`、同进程内，80 个模块挂钩逐层对拍：

| | 值 |
|---|---|
| 最差逐层 rel_err | **1.004e-14** @ `down_blocks.2.attentions.0.transformer_blocks.0.norm3` |
| e2e ε 预测 rel_err | **2.330e-15** |
| e2e PSNR | **296.6 dB** |

这一次验证覆盖了下面全部改动：

1. **124 个 BatchNorm 精确折叠**（62 处 DWConv2d 的 bn1/bn2 + 62 处 λ 层的 norm_q/norm_v），
   都折进各自前面那个原本无 bias 的 conv。实测折出的权重 `max|W'|=80.3`、`max|b'|=18.1`，
   fp16 无溢出/下溢风险。
2. **cross-λ 的 K/V 支路整条化为常量**：条件是常量 ID 表 ⇒ `lambda_c (dk,dv)` 与
   `v_const (dv,10)` 在导出期算好，`to_k` / `to_v` / `norm_v` / `encoder_hid_proj` /
   `embedding_layer` 全部从权重文件消失，CFG 通路一并删除。
3. **`rel_pos_emb[n,m]` 是恒等索引**（`meshgrid(arange(n*n), arange(m))` 配 `indexing='ij'`），
   等于参数本身，gather 与 0.65 MB 的 int64 索引表都删掉。
4. **Yp 重排**：先在 dim_k 上收缩再在 m 上收缩，乘加数从 `n·m·k·v + h·n·k·v` 降到
   `2·h·n·k·m`，且不物化 `(b,4096,40,40)` 的 13 MB 中间量。
5. **`pos_conv` 的 Conv3d → Conv2d 改写**。原 `Conv3d(1,k,(1,15,15))` 的深度维核长为 1，
   即对 V 的每个通道切片独立做 2D 卷积，权重只是 squeeze 掉那个恒为 1 的维度。
   **改写的理由不是速度**（开 `cudnn.benchmark` 后两者打平），而是 rank-5 权重会让
   `model.to(memory_format=channels_last)` 直接抛 `RuntimeError: required rank 4 tensor`。
6. **全程 NCHW，不做 `(b,N,C)` 往返**。原版在 `MixTransformer2DModel` 里 flatten 成
   `(b,N,C)`，但内部每个子层第一件事又都 reshape 回 NCHW，只有 3 个 LayerNorm 真需要
   通道在最后一维。见下节的收益。

**SDXL VAE 与 HYPIR 方案的 `AutoencoderKLLite` 的 248 个 state_dict key 逐条完全一致**
（missing / extra / shape mismatch 全为 0），整段零改动复用，`strict=True` 直接灌权重。
只有 `scaling_factor` 不同：SDXL 是 0.13025，SD2.1 是 0.18215。

---

## 三、时延：换 UNet 没用，VAE 才是全部

### 3.1 Moebius UNet 并不比 SD2.1 UNet 快

| UNet | 参数量 | 64×64 latent 单次前向 |
|---|---:|---:|
| Moebius（diffusers eager，原版） | 226 M | 37.4 ms |
| SD2.1（HYPIR Lite，已折 QKV/temb/ConvT） | 866 M | 36.8 ms |
| **Moebius（本次移植，NCHW + channels_last）** | 222 M | **26.7 ms** |

4× 少的参数换来 0% 的时延优势——原因是 kernel 发射数：λ 层的 einsum/rearrange 链把算力
打得极碎。本次移植的重写把它打到 26.7 ms（相对原版 **1.40×**），但见下。

### 3.2 `infer(512)` 分段（本次移植，fp16，1 步）

| 段 | ms | 占比 |
|---|---:|---:|
| VAE encode | 47.5 | 25.8% |
| **UNet（1 步）** | **26.7** | **14.5%** |
| VAE decode | 108.9 | 59.2% |
| wavelet 颜色校正 | 0.7 | 0.4% |
| **合计** | **183.9** | |
| **其中 VAE** | **156.4** | **85.1%** |

多步：`steps=1 / 2 / 4` 分别是 **184.1 / 211.5 / 265.4 ms**。

**结论：UNet 侧的一切优化（含 Triton）在得分上不值钱。** 把 UNet 从 26.7 打到 0，e2e 也只从
184 降到 157，加速比 1.17× → 按 `T^-0.204` 只值 **+3.2%**。真正的杠杆是 VAE：
- 复用现有 `../model_dir/csig_ops.py` 的 `group_norm_act` Triton kernel（零成本纯复用，
  上级 FINDINGS §7.5 实测把 VAE 从 182 打到 90 ms）
- 内部降到 256 分辨率：实测 encode 12.3 + decode 27.7 = 40 ms（对比 156.4），
  e2e 184→68 ms，加速比 2.7× → 约 **+23%**

### 3.3 整图

`enhance(4096×3072)`：**42.6 s**，165 个 tile（15×11），输出全 finite。
逐 tile 调 `infer` 与旧的三段式 tiled 方案 tile 数完全相同（旧方案 latent 段用
`size=64,stride=32` 在 512×384 latent 上滑窗也是 165），额外代价只有 wavelet 从 1 次整图
变成 165 次 tile，占比可忽略。

`enhance` 的分块/融合正确性单独验过：把 `_infer_tile` 换成恒等函数跑 1536×1280（20 tile），
与输入 PSNR **85.5 dB**（fp32 累加噪声底）。

---

## 四、零训练可行性：判死

`tools/probe_zero_shot.py`，验证集 3 组 LQ/GT 各取 4 个固定坐标的 512 crop（共 12 个），
扫 `mask_value × t_start`（30 配置），按 TOPIQ-FR + MANIQA 打分。LQ 基线
TOPIQ-FR 0.4142 / MANIQA 0.2450 / proxy 1.0303。

| mask | t=100 | t=250 | t=600 | t=900 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.0000 | 1.0019 | 1.0011 | 0.9704 |
| 0.25 | **1.0370** | 1.0119 | 0.9671 | 0.9337 |
| 0.50 | 1.0170 | 0.9458 | 0.7771 | 0.6738 |
| 0.75 | 0.9604 | 0.8958 | 0.6050 | 0.3419 |
| 1.00 | 0.9322 | 0.8528 | 0.5005 | **0.0614** |

（表内为 `p_rel = p(out)/p(lq)`，LQ 恒为 1.0，HYPIR 是 **2.65**）

三条读数：

1. **`mask=0` 严格退化为恒等重建**（p_rel 1.000，TOPIQ-FR 0.4139 vs 基线 0.4142，
   那 0.0003 是 VAE 往返的损耗）。这同时是整条管线接线正确的自检。
2. **mask↑ / t_start↑ 单调变差**，`mask=1, t=900`（即原版 inpainting 的默认配置）掉到
   **0.061**——模型忠实地执行了它被训练的任务：重画内容。有参考项直接崩。
3. **全网格最优只有 1.037**，远低于 1.1 的判据门。那点微弱增益基本是 VAE 往返的轻度平滑，
   不是真的细节复原。

**结论：预训练的 Moebius 做不了画质增强，必须微调。** 这与事前判断一致（成功概率 <15%）：
一个专学「抹掉物体并填背景」的模型，恰好落在「生成新内容但不对齐 GT」这个最危险的象限，
而评分里有参考项（TOPIQ-FR）的权重约为无参考项的 3 倍。

---

## 四点五、画质天花板：latent 扩散没问题，但「VAE 降 256」是白干

在同样 12 个 crop 上，把各种「理想输入」喂进管线量 TOPIQ-FR + MANIQA，得到这条路线的上界。
「保留增益」= `(p − p_LQ) / (p_GT − p_LQ)`。

| 配置 | TOPIQ-FR | p_rel | 保留增益 |
|---|---:|---:|---:|
| LQ 原图（基线） | 0.4142 | 1.000 | 0% |
| **GT（绝对天花板）** | 0.9198 | **8.336** | 100% |
| GT 只降采样 512→256→512 | 0.8369 | 7.060 | 82.6% |
| **GT 只过 VAE @512** | 0.9094 | 8.194 | **98.1%** |
| GT 降采样 + VAE @256 | 0.7350 | 5.668 | **63.6%** |
| （LQ 只过 VAE @512） | 0.4140 | 0.997 | — |

**1. SDXL VAE 不是画质瓶颈。** @512 往返只吃掉 1.9% 的可得增益 —— latent 扩散这条路本身成立，
瓶颈全在 UNet 侧的生成能力。反过来说，想用 TAESD 之类的轻量解码器替换它，必须先确认
那个解码器自己的损失也在同一量级，否则得不偿失。

**2. 「VAE 内部降到 256」是白干，本文档上一版对它的 +23% 估计是错的**——那个估计只算了速度、
没算画质。精算：256 下 encode 12.3 + decode 27.7 + UNet@32² 约 7 = 47 ms，相对 184 ms 提速 3.91×；
但保留增益只有 63.6%，`p` 从 2.65 掉到 `1 + 1.65×0.636 = 2.05`。

```
Score 比 = (2.05 / 2.65) × 3.91^0.204 = 0.773 × 1.312 = 1.014   →  +1.4%
```

对照上级 FINDINGS §3 的盈亏平衡线（4× 提速需保住 60.5%），63.6% 只是勉强过线，
**净收益约 +1%，在噪声里**。这条从「最大杠杆」降级为「边际项」。

注意两种损失不是简单相乘：单独降采样保 82.6%、单独过 VAE 保 98.1%（乘积 81%），
但合起来只有 63.6% —— VAE 的 8× 压缩作用在已经降过采样的图上，相对损失大得多。

**3. 真正的空间在画质本身。** 绝对天花板 p_rel = 8.34，而现有 HYPIR 基线是 2.65，
**只吃到可得增益的 32%**（`(2.65−1)/(8.336−1)`）。感知分线性进总分，速度只有 0.2 次方 ——
把 2.65 推到 4.0 就值 +51%，比任何速度优化都大一个量级。

---

## 五、下一步

按分数收益排序（本次移植**不涨分**，它的价值是决赛的轻量架构叙事 + 干净的代码底座）：

| 方向 | 预估收益 | 工作量 |
|---|---:|---|
| **画质：把 p 从 2.65 往 8.34 的天花板推** | **+10% 每 0.165 的 p** | 见下，这是唯一值得投入的主轴 |
| 微调：按上级 FINDINGS §5 的退化模型（2× area 下采样 + cubic 上采样）对抗训练 | 未知，上界 +215% | 5–8 天 |
| 复用 `csig_ops.py` 的 `group_norm_act` 打 VAE | +8~12% | **0（纯复用）** |
| 微调：蒸馏 HYPIR → Moebius（教师 p=2.65） | 保住画质，速度不变，净约 0 | 3 天 |
| CUDA Graph 捕获整条 `_infer_tile` | +1~2% | 1 天 |
| ~~VAE 内部降到 256 分辨率~~ | ~~+23%~~ → **实测 +1.4%** | **已降级，见 §4.5** |
| self-λ 融合 Triton kernel | **+0.5%** | 3 天，**不建议现在做** |

微调时的模型改造（`onestep` 模式已在 `runner.py` 里实现好）：固定 timestep、`z_t = z_LQ`
不加噪、`ddpm_pred_x0` 单步出图。**mask 通道不必改结构**——它恒为常量，其在 `conv_in` 的
depthwise 输出是一张常量图，可精确折进 bias，权重 100% 复用。

**尚未做**：`torch.jit` 导出（说明.md 第 2 条硬要求）。需要新建 py39 + torch 2.5.1 环境，
既用于导出（当前 torch 2.13 导出的 `.pt` 在评测机 2.5.1 上很可能 load 失败），也用于
py3.9 语法冒烟测试（上级 FINDINGS §7.6 那次提交失败的成因）。`runner.py` 已全文避开
PEP 604 的 `X | Y` 注解。
