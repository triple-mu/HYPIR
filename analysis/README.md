# `analysis/` —— 一次性分析脚本

这里全是**为得出某个结论而写、结论进了 `docs/FINDINGS-*.md` 之后就不再维护**的脚本。
不参与训练，也不参与提交。留着是为了两件事：结论可追溯，方法可复用。

| 目录 | 干了什么 |
|---|---|
| `probe/` | 探测包构建器。**评分公式是用它逆向出来的**：构造只改一个变量的提交包（恒等 runner / 定时 runner / USM 锐化 / LQ 原样），拿线上分反解出 `综合分 = 感知分 × 加速比^0.204`。`probe_results.csv` 是原始数据 |
| `forensics/` | JPEG 字节级取证：量化表匹配、双压检测、块效应、频谱天花板。`*.json` 是实测结果 |
| `exif/` | EXIF / MakerNote / MPF / ICC / 增益图解析，判定 LQ 的相机来源与后处理链 |
| `classic/` | 经典算子路线的完整证伪：去噪、去模糊、最优滤波器搜索、JPEG 重编码验证 |
| `tone/` | 色调与 gainmap 的 27 组对照实验 |
| `pd12m/` `commons/` | 训练数据源调研与采集（`commons/README.md` 里有索引重建步骤） |
| `bench/` | 推理侧算子基准：融合、带宽、分块尺寸、上采样 |
| `train_bench/` | 训练侧基准：DDP、NCCL、单步耗时、显存、精度、LoRA dtype、梯度噪声尺度 |
| `refs/` | Triton 参考实现（fused attention、RMSNorm） |

## 别直接跑

约 24 个脚本里写死了整理前的扁平路径（`CSIG-2026/analysis_tone/`、`CSIG-2026/verdict/`
之类），其中一部分在整理**之前**就已经指向不存在的目录了（`deblur_search/`、`pd12m_recon/`）。
这些脚本是当时一路调出来的产物，没有做成可复跑的工具，要用先改路径。

被引用到的原始数据大多在 `../../赛题二/`，或者已经随脚本一起搬进了各自目录。
