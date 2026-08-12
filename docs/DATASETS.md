# 数据集溯源与归档

每个数据集记三件事：**从哪来、怎么筛的、产出是什么**。目标是任何一份训练/评估数据都能从零复现。

归档原则：**图片本身不备份**（几百 GB，且可按 URL/脚本重建），**备份"选中了哪些"和"用什么参数选的"**——
即 URL 清单、file_list、parquet、以及本文件记录的参数。这几样加起来只有几十 MB，但丢了就无法复现。

---

## 1. 训练数据

### 1.1 LSDIR（当前唯一训练源）

| 项 | 值 |
|---|---|
| 来源 | HF `danjacobellis/LSDIR`（97.8 GB）。官方 `ofsoundof/LSDIR` 是 gated 的 **model** repo（223.6 GB，HR 部分 155.04 GB），非 dataset repo |
| 许可 | LSDIR 原始条款为学术研究用途 |
| 规模 | 84,991 张源图 |
| 产出 | **414,938 个 512×512 patch**，平均 4.88 patch/图，落盘 180 GB PNG |
| 生成脚本 | `csig/prep_lsdir.py` |
| 切块方式 | **均匀铺满**（非等距步进）：`starts()` 计算一维等距起点正好盖住 `[0, size)`，残边不丢弃，相邻 patch 重叠 30–40% |
| 索引 | `data/lsdir_512_nulltxt.parquet`（704 KB，两列 `image_path` / `prompt`） |
| prompt | **全空串**（唯一值 1 个）。依据：`csig/prompt.py` 的零样本消融显示画质形容词越多，感知指标单调下降 |
| 确定性 | 两台机器产出逐位一致 |

**画质核验（实测，8×8 块效应比）**：LSDIR patch **1.0015** / DIV2K_V2_val GT 0.9979 / 赛题真实 GT 1.1031。
→ 训练 GT 无 JPEG 重编码污染。（`danjacobellis/LSDIR` 97.8 GB 比 `LSDIR_raw` 155.2 GB 小 37%，
但块效应检测排除了重编码，体积差来自 PNG 压缩级别。）

**已知局限（实测）**：每图切出的 patch 数分布是 **p10=4 / p50=4 / p90=6 / max=6** ——
`max=6` 说明 LSDIR 的图**全部**在 1024–1536 px 量级，没有一张例外，长边 ≥3000 的只占 94/84,991 = 0.11%。
而赛题测试图 100% 是原生 4K。这是 GT 域与目标域最大的不匹配。

**注意一处容易误解的地方**：三闸门的 `MIN_LONG=3000` **从来没有用在 LSDIR 上**，
它只用于筛 PD12M 这类新增源。所以「现有训练数据 100% 是 1200 px 量级」与
「新增源要求 ≥3400」并不矛盾——两者补的缺口不同：

| | LSDIR | PD12M |
|---|---|---|
| 作用 | 内容多样性（84,991 张） | **尺度与画质档位** |
| 每图 512 视野 | 4–6 个 | **58 个（12 倍）** |

关键差异在**尺度**而非锐度（锐度由 `hf_max` 独立闸门管）：赛题推理时 512 patch 是从
4K 原生图切的，相当于「1/64 张图的视野」，patch 里可能只有一扇窗户；1200 px 图切的 patch
是「1/6 张图」，装得下整栋楼。模型学到的「什么算细节」不同。
**若把新增源放宽到 512，它会退化成第二个 LSDIR，混合就失去意义。**

### 1.2 PD12M（新增，手机 ISP 域，配比 30%）

| 项 | 值 |
|---|---|
| 来源 | HF `Spawning/PD12M`（元数据 parquet + CLIP 向量，图片托管在 AWS S3 桶 `pd12m`） |
| 许可 | **CDLA-Permissive-2.0**，图片为 PD/CC0 —— 本清单里许可最干净，可商用 |
| 元数据 | 125 个 parquet 分片，本地 38 GB |
| 生成脚本 | `csig/prep_pd12m.py`（`index` 出候选 URL → `fetch` 边下边筛） |

**为什么选它**：唯一同时满足「手机 ISP（~90% 真 EXIF）+ 12MP 4:3 + bpp 3.11 + 可商用」的源。
对照 bpp 标尺：赛题原生 4.56–5.81 | HDR+ 4.17 | **PD12M 3.11** | ShopSign 1.84 | 4KLSDB 1.6。

**第一层：元数据筛**（`prep_pd12m.py::do_index`）

```
mime_type == "image/jpeg"
max(w,h) >= 3400  且  min(w,h) >= 512
宽高比 ∈ [1.28, 1.40]                    # 12MP 手机出图的 4:3（含少量 3:2 边缘）
caption 命中 KEEP 且不命中 DROP
  KEEP = building|architecture|facade|skyscraper|tower|city|urban|street|skyline|window|
         balcony|roof|bridge|house|plant|flower|leaf|leaves|tree|foliage|garden|branch|
         petal|blossom|grass|park|forest|sign|shop|store|boat|ship|vehicle|bicycle|
         mountain|landscape
  DROP = portrait|man|woman|person|people|girl|boy|face|painting|drawing|illustration|
         engraving|sketch|manuscript|coin|stamp|map|document|museum|artwork|sculpture of a
```

KEEP/DROP 的设计意图是对齐赛题内容域（实测配比 building 0.38 / vegetation 0.26 / skyline 0.15 /
signage 0.06 / object 0.05 / street 0.04，无人像）。

实测：扫 12,500,000 行 → 候选 **574,908 条**（4.6%）。

**第二层：三闸门锐度筛**（`prep_pd12m.py::one`，与 `csig/screen_dir.py` 同一套常量）

```
闸门0 尺寸  : min(size) >= 512 且 max(size) >= 3000
闸门1 谱截止: hf_max(灰度图) >= 0.005     # 保留 ~92% 真原生，误收 ~15% 假高清，误收 0% 模糊图
闸门2 量化表: IJG quality >= 93           # 仅对 IJG 族生效，厂商自研表跳过
闸门3 bpp   : len(bytes)*8/(w*h) >= 2.0
```

标定依据见 `csig/README.md` 第三节。核心原则：**训练 GT 必须在原生分辨率上真正锐利**，
否则叠上退化管线就是双重模糊，模型学不到该学的东西。

**本轮实测（2026-08-12）**：

```
元数据筛 : 扫 12,400,094 行 -> 候选 604,480 条 (4.87%)
三闸门   : 尝试 20,000 条 -> 通过 12,268 张 (61.3%)，75 GB
下载速度 : 15 张/秒，70 MB/s（96 线程）
源图尺寸 : 长边 p10=4000 / p50=4608 / p90=4608，短边 p50=3456（典型 16MP 手机 4:3）
切块产出 : 59.2 patch/图（对照 LSDIR 的 4.88）
```

注意通过率 **61.3%** 与历史记录的 1.6% 相差 37 倍——历史那个数字疑似统计口径不同
（可能把元数据筛的淘汰也算进了分母）。以本轮实测为准。

---

## 2. 评估数据（四把尺子）

| 尺子 | 来源 | n | 退化类型 | 用途 |
|---|---|---:|---|---|
| **DIV2K_V2_val** | `Iceclear/StableSR-TestSets` | 3000 对 | 合成（Real-ESRGAN） | in-domain 主对拍，论文 Table 1 同协议 |
| **RealSRVal_crop128** | 同上（附带） | **100 对** | **真实**（Canon/Nikon 变焦） | **OOD 泛化主判据** |
| **DrealSRVal_crop128** | 同上（附带） | **93 对** | **真实**（5 台相机） | OOD 交叉验证 |
| 赛题验证集 | `csig_bench/赛题二/验证集` | 3 对 | 真实（手机，4K 原生） | 否决闸门 |
| DPEDiphoneValSet_crop128 | 同上（附带） | 113 张 | 真实（手机） | **仅 LQ 无 GT**，暂未用 |

**RealSRVal / DrealSRVal 的接入方式**（零改代码）：原始目录是 `test_HR` / `test_LR`，
而 `eval_bench.py` 期望 `gt` / `lq`；两者文件名**完全同名**，所以建符号链接即可：

```bash
B=$CSIG/data/bench/StableSR_testsets
for d in RealSRVal_crop128 DrealSRVal_crop128; do
  T=$CSIG/data/bench/${d}_std
  mkdir -p $T && ln -sfn $B/$d/test_HR $T/gt && ln -sfn $B/$d/test_LR $T/lq
done
python csig/eval_bench.py --weight <w> --vae taesd --bench-dir $T --tag <tag>
```

**尺子的分辨力实测**（drt@3000 vs 官方基线）：

| 尺子 | Δ感知分 | t | 胜率 | 可测阈 |
|---|---:|---:|---:|---:|
| DIV2K n=300 | — | +10.27 | — | 0.060 |
| **RealSRVal n=100** | **+0.3425** | **+10.23** | 89% | **0.0663** |
| **DRealSRVal n=93** | **+0.3136** | **+7.44** | 78% | — |

三个独立域给出一致的效应量与方向 → 尺子可信。
另：CLIP-IQA 在 RealSRVal 上 t=+0.84 不显著，与「赛题公式中 CLIP-IQA 权重为零」的逆向结论吻合。

**重要约束**：DRealSR 的 Test_x4 93 张与公开基准 `benchmark_drealsr` 经像素级验证是**同一批图**。
它只能当尺子，**永远不能进训练集**，否则污染 StableSR/SeeSR/OSEDiff/SUPIR 的公共评测基准。

---

## 3. 已评估并否决的数据集

| 数据集 | 实测 | 否决理由 |
|---|---|---|
| `eugenesiow/Div2k` | 3 个文件 15.3 KB，**零张图** | 已失效的 loading script；即便直下 ETH，800 张对 84,991 张仅 +0.94%；且 DIV2K_valid 已是主尺子，有污染风险 |
| `asksnskksnfapjgwanga/DrealSR` | 10.59 GB **全是测试集**（83+84+93=260 对） | Test_x4 与公开基准像素级同批，训练即污染 |
| `IQA-Dataset-team-IVC/IQA-Dataset` | 2 MB 纯标注，图片另需 128.8 GB | 最大的两个子集 KADID-10k / KonIQ-10k **正是 `topiq_fr` 与 `maniqa` 的训练集**，拿来校准评分代理是循环论证 |
| `q-future/Q-Instruct-DB` | 18,969 图 + 200,534 条 QA | LMM 指令微调**对话**数据，无 LQ-HQ 对，`gt_score` 实测出现 0 次 |
| `SingleBicycle/4KLSDB` | 129,484 张原生 4K，CC-BY-4.0 | bpp 仅 1.6（比赛题 GT 的 2.20 还钝），49.6% 人像，中文线索 1.7%，最锐 patch 高频能量比真手机高 **6.57 倍** |
| RealSR / DRealSR / SR-RAW 训练集 | — | 退化是相机光学变焦，与赛题手机 ISP+压缩不同源；且残余亚像素错位会把模型往糊里拉，直接打击 MANIQA（权重 4.477） |
| SA-1B | 1100 万张 | research-only 许可；人脸车牌打码块会被学成伪影源 |
| LAION / COYO / DataComp | URL-only | 链接腐烂 + web JPEG 再编码，当 GT 会教出压缩伪影 |
| FFHQ | 70,000 张 | 赛题测试集无人像，纯负收益 |

---

## 3b. PD12M 混合实验：判负，但定位到了机制

**结论：LSDIR 70% + PD12M 30% 判负。** RealSRVal n=100 逐样本配对（vs drt@3000 基线，同机同批）：

| 预注册点 | Δ感知分 | t | 胜率 |
|---|---:|---:|---:|
| mix@1500 | −0.0855 | −3.51 | 36% |
| mix@2250 | −0.0918 | −3.24 | 37% |
| mix@3000 | −0.0925 | −3.05 | 35% |

全部超过可测阈 0.0663。分项是**教科书式的一维曲线滑动**，方向是往保真端：

```
全参考/保真侧全好： TOPIQ-FR +0.0105 (t=+4.19)  PSNR +0.69 (t=+9.52)  LPIPS −0.017 (t=−6.25)
无参考/感知侧全差： MANIQA  −0.0528 (t=−9.83)  MUSIQ −3.42 (t=−7.76)
```

MANIQA 跌 0.0528 × 4.477 = −0.236，TOPIQ-FR 涨 0.0105 × 13.674 = +0.144，净 −0.09。
参照：drt 那次成功改动的全部 MANIQA 收益才 +0.0241，**这次掉的是它的 2.2 倍**。

### 机制（实测，n=300/组）

| | LSDIR patch | PD12M patch | 比值 |
|---|---:|---:|---:|
| Laplacian 方差中位 | 1639.8 | 115.5 | **0.070×** |
| hf_max 中位 | 0.00559 | 0.00181 | 0.32× |
| **低纹理占比（方差<100）** | **1.0%** | **46.0%** | 46 倍 |

全量分位对照：LSDIR `p05=287 p25=800 p50=1530 p75=2730 p95=5469`；
PD12M `p25=34 p50=158 p75=525 p90=1430` —— **PD12M 的 p90 还低于 LSDIR 的中位**。

### 两条可复用的教训

**教训一：筛选粒度必须与训练粒度一致。**
三闸门在**整图**上测 `hf_max`，PD12M 的 4K 原生图确实锐利、全部过闸；但训练用的是切出来的
512 patch，4K 图「局部放大」后 46% 落在平坦区（墙面、天空、玻璃幕墙）。闸门测整图、训练用
patch —— 粒度脱节。修复见 `csig/filter_patches.py`（patch 级筛）。

**教训二：训练分布匹配推理分布，不等于分数更高。**
赛题输入确实是 4K 图切 patch，PD12M 在这个意义上比 LSDIR 更「真实」，但分数反而降。
原因在评分公式的构成：**MANIQA（权重 4.477）是无参考指标，奖励「输出锐利、细节丰富」，
与 GT 是否忠实无关**。低纹理 GT 教会模型少动手，直接压 MANIQA。
这与两个已知现象同源：prompt 加画质形容词反而掉分；`raw@1500` MANIQA 最高但 TOPIQ-FR 崩。

→ 由此反推出的方向：**不是换更真实的 GT，而是按纹理密度筛 GT**（见 `lsdir_tex70`）。

### 资产（保留，未删）

`data/pd12m_hr` 12,268 张 75 GB、`data/pd12m_512` 296,847 patch 87 GB、
`data/mix_lsdir_pd12m.parquet`、`out/taesd_mix` 13 个快照。
若要试「小配比 + 只取最高纹理的 9%（方差≥1500，27K patch）」，数据是现成的。

---

## 4. 归档清单

**必须备份**（不可再生，总计几十 MB）：

| 文件 | 说明 |
|---|---|
| `data/pd12m_urls.txt` | PD12M 元数据筛后的候选 URL 清单 |
| `data/lsdir_512_nulltxt.parquet` | LSDIR 的 patch 索引 |
| `data/mix_*.parquet` | 各次混合实验的训练索引 |
| `docs/DATASETS.md` | 本文件（筛选参数） |
| `csig/prep_*.py`、`csig/screen_dir.py` | 生成脚本（已在 git） |

**不备份**（可重建）：LSDIR patch 180 GB、PD12M 图片、评估集——
前两者由脚本 + URL 清单重建，评估集从 `Iceclear/StableSR-TestSets` 重下。

打包命令见 `csig/archive_datasets.sh`。
