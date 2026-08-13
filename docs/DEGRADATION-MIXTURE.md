# CSIG-2026 GT→LQ 多分支退化实现

本实现把三对验证图能够确定的共同事实与测试集中两个文件/成像子域编码为一个可回放的
训练分布；它不是对组织方私有 ISP 的“精确复刻”。所有 down-up 倍率和 PSF 参数都是等效
拟合范围，不能解释为相机数字变焦倍率。

## 入口与数据流

训练配置使用：

```text
CSIGDataset(encoded bytes)
  -> csig_collate(list[bytes])
  -> CSIGBatchTransform(native crop + sample metadata)
  -> TensorDegrader
  -> {GT, LQ, txt [, degradation_meta]}
```

唯一参数真源是 `HYPIR/dataset/csig_degradation.py`：

- `sample_degradation(rng, ...)`：纯 NumPy 采样，返回 strict-JSON-safe dict；
- `DegradationSampler(seed, rank)`：训练态 seed allocator；
- `TensorDegrader.apply(hq, metadata)`：CPU/CUDA RGB BCHW 张量执行；
- `degrade_tensor(...)`：分析脚本的共享便捷入口；
- `csig/degrade.py::degrade(...)`：保留旧 BGR uint8 API，最终走真实 libjpeg/Pillow 字节编码。

训练 batch 默认仍只有 `GT/LQ/txt`，不改变 Trainer 契约。调试或生成固定评估集时设
`return_metadata=true`，会额外得到 `degradation_meta: list[dict]`。

Dataset 按魔数区分 JPEG/PNG/WebP，不信任扩展名；小于 crop 的图会被拒绝，绝不插值放大
后冒充 HQ。JPEG 继续走 nvJPEG，PNG/WebP 用 torchvision 通用解码后搬到训练设备。

file list 支持：

```text
/absolute/day.jpg
/absolute/night.jpg<TAB>night_hdr
relative/building.png<TAB>ordinary
```

profile hint 优先于 transform 的 `profile`；无 hint 且 `profile=auto` 时按 90/10 抽样。

## 分布定义

### ordinary：90%

强度为弱/中/强 `20%/50%/30%`。弱分支的一半为 `near_native`，因此总样本约 10% 只经过
末端 JPEG，作为“输入已经清晰时少动手”的保护。

非 near-native 的算子分布：

| 算子 | 概率 | 主要范围 |
|---|---:|---|
| 重尾空间变化低通 | 45% | sigma：弱 0.2–1.2、中 1.2–3.5、强 3–6 px |
| 各向异性高斯 | 20% | 同上；长短轴比 1–1.5 |
| 仅 down-up | 25% | 弱 1–1.5、中 1.5–4、强 4–7 倍 |
| 低通 + down-up | 10% | 两者串联 |

下采样 area/bilinear/bicubic 为 `45%/30%/25%`，上采样 bilinear/bicubic/Lanczos 为
`55%/40%/5%`。PyTorch 没有原生张量 Lanczos，当前 5% 分支以 non-antialiased bicubic
近似其振铃族，metadata 保留请求的 family；若以后引入可微 Lanczos，应先单独做像素回归。
50% 低通样本使用 ±40% 的平滑空间变化 mask。

非 near-native 样本还可带亚像素位移、0.5–3 px 平滑局部形变、edge-aware 纹理压平、轻 halo、低概率局部融合残影、
色度模糊/位移，以及围绕三对验证图仿射拟合锚点采样的 flat/neutral/contrast tone（约
`0.91x+8/255`、`0.97x+1/255`、`1.15x-20/255`，另保留 identity）。RGB 参数相关采样，
不会凭空制造逐通道独立色偏。普通域末端不加随机噪声，固定为实测 IJG Q95 整数量化表和
4:2:0。10% 样本允许退化前 Q95–100 高质量 JPEG，以覆盖 HQ 源已有轻压缩的情况。

### night_hdr：10%

夜景分支模拟“噪声发生在强降噪之前”的计算摄影链：2–6 倍等效 down-up、sigma 1–5、
0.5–3 px 运动，异方差 shot/read noise，随后 edge-aware 强降噪、低概率局部融合错位、
bloom、高光压缩、暗部压低和轻色彩/饱和度扰动。

其末端采用 9 张真实 MPO 主帧/增益图共享的华为量化表与 4:2:0，而不是把厂商表错误映射成
IJG quality。`csig/mpo_hdr.py` 可解析 MPF、ISO 21496-1、逐帧 ICC 和未索引私有尾块；
gain map 仅用于分析、高光 mask 或教师信息，RGB 恢复模型不依赖文件元数据。

## Metadata 与回放

每个 record 固定包含 schema/version、`sample_seed`、实际 profile、强度/算子、kernel/resize、
几何、tone/chroma、night noise/denoise/fusion/bloom、JPEG backend/qtable，以及 crop 坐标、
原图尺寸、翻转和源路径。不适用字段使用 `None`，禁止 NaN、Tensor、NumPy scalar；验收标准是：

```python
json.dumps(metadata, allow_nan=False)
```

空间 mask 与噪声 realization 由 `sample_seed` 派生；同一 metadata 重放逐位一致。DDP 下 seed
会混入 rank，避免不同进程生成完全相同的退化序列。

训练端的 DiffJPEG 是张量模拟：它确定采用 4:2:0，并使用显式整数表，但输出不是 libjpeg
字节流。单图评估路径则真实编码，可用 Pillow 独立验证量化表和 sampling factors。

## 验证命令

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch
cd /home/ubuntu/workspace/contest/CSIG-2026/HYPIR-fork

PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  tests/test_csig_sampler.py tests/test_csig_dataset.py \
  tests/test_csig_single_image.py tests/test_mpo_hdr.py

CSIG_DATA_ROOT='/home/ubuntu/workspace/contest/CSIG-2026/赛题二' \
  PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_mpo_hdr.py

python -B csig/mpo_hdr.py \
  '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集/case38.jpg' --pretty

python -B csig/degrade_vis.py \
  --val '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集' \
  --out /tmp/csig-degrade-vis --draws 4 --crop 512

python -B csig/mtf_coverage.py \
  --val '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集' \
  --draws 60 --crop 1024
```

2026-08-13 在 `conda activate torch` 下的当前结果：合成/API 测试 `24 passed, 1 skipped`；
启用真实数据后的 MPO 回归 `8 passed`。60 次逐 case MTF 抽样中，case1、case2 分别只有
`1/9`、`0/9` 个探测频点越出 10–90 分位，case3 为 `5/9`。后者的低频传递大于 1、部分
中频为负，并伴随肉眼可见的阴影压黑、局部对比和内容相关边缘变化；这说明现分布覆盖其
主要模糊强度，但没有“精确复刻”该非线性/非平稳链。诊断工具会明确输出“部分覆盖”，
不会再用三张平均百分位掩盖这个残差。

选择 checkpoint 时必须同时看三对真实 LQ/GT、weak/near-native 保真和各内容 held-out，不能再用
一个合成感知分或单一平均 PSNR 作为退化匹配结论。
