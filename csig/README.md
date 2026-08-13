# CSIG-2026 赛道二 · 训练数据集复现

一键跑：`bash run_all.sh /path/to/data_root`（分步：`STEPS=pd12m bash run_all.sh <root>`）

产出 `<root>/lists/train_mix.txt`，直接填进 `HYPIR/configs/csig_train.yaml` 的 `file_list`。

> **2026-08-13 更新：**训练真源现为
> `HYPIR/dataset/csig_degradation.py`。早期的“唯一强低通 + 单一参数箱”已被三对验证图和
> 100 张测试图的完整取证推翻；当前采用 ordinary 90% + night/HDR 10% 的可回放混合，
> 含弱/近原生保护、空间变化低通、down-up、tone/chroma、计算摄影近似及逐域真实 JPEG
> 量化表。完整接口、参数与验收命令见 [DEGRADATION-MIXTURE.md](../docs/DEGRADATION-MIXTURE.md)。
>
> file list 可选第二列：`path<TAB>ordinary` 或 `path<TAB>night_hdr`。有标签时优先使用，
> 无标签时才按 90/10 抽样。构建混合时可加 `--profile hdrplus=night_hdr`。

---

## 一、这套管线在解决什么问题

赛题表面是「图像增强」，实质是同尺寸盲恢复。主退化仍是低通/ISP 细节压平，但并非唯一算子：
三对验证图有不同强度、空间变化、重采样等价解和局部 tone/chroma；测试集另有 9 张华为
ISO 21496-1 gain-map MPO 夜景子域，且部分测试输入接近原生清晰度。

由此推出两条约束，整个管线都是围绕它们建的：

**1. 退化必须按实测配方合成，不能用 RealESRGAN 那一套。**
实测真实 LQ 的感知强度相当于 **≈7× 重采样 / σ≈3.6 高斯**（不是此前认为的 2×）。
而 HYPIR 原配置一半样本带 σ≤30 的高斯噪 + 泊松噪，真实 LQ 的噪声实测只有 **σ 0.014–0.090 灰阶**。
模型学到的激进降噪，喂给它完全无噪的输入时只会吃掉本就稀缺的细节。
配方实现见 `degrade.py`（单图真实 JPEG）、`../HYPIR/dataset/csig_degradation.py`
（共享采样/张量执行）与 `../HYPIR/dataset/csig.py`（训练数据接入）。

**2. 训练 GT 必须在原生分辨率上真正锐利。**
退化的 σ≈2–4 px 是在**原生 4K 栅格**上测的。若先把 4K 缩到 512 再加退化，相对模糊强度差 8 倍。
所以 512 crop 必须**从原生高分辨率图上直接裁**，且源图本身不能是插值放大或已被压过的
——否则叠上我们的退化就是双重模糊。这就是下面三闸门存在的理由。

---

## 二、为什么不用那些「常见」数据集

全部实测，不是凭印象：

| 数据集 | 实测 | 裁决 |
|---|---|---|
| LSDIR | 长边 ≥3000 只有 **94 / 84,991**（0.11%），中位 1224 px；HF gated | ❌ |
| DIV2K / Flickr2K | **0%** ≥3000px（硬顶 2040） | ❌ |
| LSVT / C-SVT | **0%**，服务端已统一缩到 1936 | ❌ |
| CTW | 0%（2048 统一）+ CC BY-NC-SA + 腾讯街景车尼康单反 | ❌ |
| RCTW-17 | 量化表反推 **q≈75**，bytes/pixel 仅 0.136，重编码过 | ❌ |
| DIV8K | 72.5% ≥3000px 但**许可空白**（HF 上的 apache-2.0 是上传者随手打的） | ⚠️ |
| 4KRD / 4KID / UHD-Blur / UHD-Haze | **视频抽帧**伪装成图像复原基准，带运动模糊 + 视频压缩 | ❌ 整类 |
| GSMArena（208 张 Pura 80 Ultra 原图，同机型完美匹配） | `license.xml` 逐条 `<prohibits>ai-train</prohibits>`、`train-genai`；robots.txt Disallow ClaudeBot/anthropic-ai/CCBot | ❌ 法律 |
| 500px 中国 / 视觉中国 | 逐字禁「机器学习，生成式或非生成式人工智能算法的训练」，且有成建制维权团队 | ❌ 法律 |
| Pixabay / Unsplash 全量 | 前者明文禁 ML 训练；后者训出的**模型本身**也禁商用 | ❌ 法律 |
| 华为 XMAGE | 大赛条款第 8 条把训练权只授给华为及其转授权方 | ❌ 需授权 |

---

## 三、三闸门：判断一张图能不能当训练 GT

实现在 `csig_data.py`（`hf_max` / `ijg_quality`）与 `screen_dir.py`。

### 闸门 1 · 谱截止 `hf_max >= 0.005`

`hf_max` = 把图切 4×4 个 512 patch（跳过 `std<=8` 的纯色/天空），每个做 Hanning 窗 + `|FFT|²` +
径向平均，报 `P(r>0.5·Nyq) / P(r>0)`，**整图取 max**（取 max 而非中位，是为了避开虚化背景/大片天空的误杀）。

标定（n=16 真手机图，每张同时造三个变体）：

| 变体 | hf_max 中位 | p10 |
|---|---:|---:|
| 原生 | **0.01717** | 0.00567 |
| 2× 下采样+上采样（典型「假高清」） | 0.00266 | 0.00075 |
| 高斯 σ=2.5（≈赛题退化强度） | 0.00006 | 0.00001 |

阈值 0.005 → 保留 ~92% 原生、误收 ~15% 假高清、**误收 0% 模糊图**。数据不缺，宁可多杀。

> 曾用「2× 自往返 PSNR」当判据，**已废弃** —— 内容依赖太强，实测把原生的 `case1.jpg`(47.82 dB) 误拒、
> 把已退化的 `case99.jpg`(41.56 dB) 误收。

### 闸门 2 · 量化表 `IJG quality >= 93`（厂商自研表跳过）

把量化表拟合到 IJG 标准表的缩放族反推 quality；**拟合不上就是厂商自研表，此闸门不适用**。
这一步必须做：华为原生 JPEG 的亮度表和是 **915**，按 IJG 的「表和↔质量」映射会被误判成低质量图，
而那正是最宝贵的域内 GT。

参考值：`369=Q95  441=Q94  518=Q93  814=Q89  1477=Q80`。

### 闸门 3 · 普通单帧图 `bpp >= 2.0`

标尺（实测）：HDR+ 4.17 | PD12M 3.11 | 赛题 GT 2.20 | ShopSign 1.84 | 4KLSDB 1.6 | Pexels 0.83。

9 张华为 MPO 必须例外处理：旧统计把未索引的 5.8–6.1 MB 私有尾块也算进了主图，才得到
4.56–5.81 的假高 bpp。按 MPF 边界计算，主帧实际为 **0.65–1.96 bpp（中位 1.096）**；
它们由 ISO 21496 gain map、厂商量化表和 ICC 共同识别，不能拿普通单帧 IJG 阈值误杀。

---

## 四、各源：来源、许可、实测产出

| 层 | 源 | 许可 | 实测产出 | 备注 |
|---|---|---|---|---|
| **L1 通用大盘 55%** | [4KLSDB](https://huggingface.co/datasets/SingleBicycle/4KLSDB) | CC BY 4.0 | 40 分片 / 180 GB → 扫 23,864 → **5,683 张**（23.8%） | 100% ≥3840px，但 49.6% 人像、中文线索仅 1.7% |
| **L2 手机 ISP 30%** | [PD12M](https://huggingface.co/datasets/Spawning/PD12M) | **CDLA-Permissive-2.0** | 1240 万行 → 60.4 万候选 → **9,225 张**（通过率 61%） | ~90% 真手机 EXIF；抽样见 iPhone 8/SE2、HONOR NTH-AN00、尼康 COOLPIX |
| **L3a 计算摄影 8%** | [Google HDR+](https://hdrplusdata.org/) | CC BY-**SA** ⚠️ | **3,640 张** / 16 GB | Pixel/Nexus 真多帧融合管线成品。SA 是否传染模型权重是未决法律问题，占比宜低 |
| **L3b 中文店招 5%** | ShopSign（公开 1,265 子集） | 未声明 ⚠️ | 1,265 → **288 张**（22.8%） | EXIF 全是国产机：Xiaomi MIX 2 / OPPO A53m·A57 / vivo X9·X6SPlus / HUAWEI MLA-AL10·CHM-TL00H。但发布前被重编码（IJG q95，bpp 1.84） |
| **L3c 中文场景文字 2%** | [CASIA-10K](https://nlpr.ia.ac.cn/pal/CASIA10K.html) | 未声明 ⚠️ | 待测 | 3968×2976 / 4032×3024，与目标 4096×3072 几乎同规格 |

### 需要人工下载的两个

**ShopSign** —— 公开只有 1,265 张；全量 25,362 张需寄硬盘给 `cszhang@henu.edu.cn`。
把 `ShopSign_1265.tar.gz` 解到 `<root>/ShopSign_1265/` 即可，`run_all.sh` 会自动筛。

**CASIA-10K** —— `https://nlpr.ia.ac.cn/pal/Dataset/CASIA-10k.zip`（5.2 GB）。
⚠️ **注意别拿错**：nlpr.ia.ac.cn 底下同名数据集很多。要的是
[CASIA10K.html](https://nlpr.ia.ac.cn/pal/CASIA10K.html)（场景文字，4K 街景照片），
**不是** [handwriting/Download.html](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)
（手写字符库，`80-99.img.tar.gz`，图像长边中位仅 **72 px**，完全用不上）。
页面无许可声明，商用前需去信 `liucl@nlpr.ia.ac.cn`。解到 `<root>/CASIA-10k/`。

### 一个必须知道的偏差

真手机 ISP 的高频能量上限比图库/单反**低 6.57 倍**（`hf_max` 最锐 patch：手机 0.028 封顶，图库到 0.316）。
物理原因是手机 ISP 的降噪给精细纹理设了硬天花板。拿图库数据训出来的模型会**幻想出手机根本不会产生的
精细纹理**，而评分里全参考项（TOPIQ-FR）权重是无参考项（MANIQA）的 3 倍 —— 这些多出来的纹理是净扣分。

L2/L3a 两层存在的意义就是纠正这个偏差。若要更进一步，可用赛题自带的 12 张真机图
（3 张验证集 GT + 9 张未退化的测试集原生图）拟合目标径向谱，对 L1 的 GT 做谱匹配
（只压 `r>0.35·Nyq` 的超额高频，`clip(g,0,1)` 只降不升，只动 Y 通道）。

---

## 五、评估集（`build_eval.py`）

**为什么必须单独造**：赛题验证集只有 3 张，全是建筑/远景城市；而测试集里
**植被 26%、招牌 6%、器物 5%、街景 4% —— 覆盖为零**。拿 3 张建筑图调出来的参数，
在那 41% 的图上没有任何保证。

做法：4KLSDB 的 **00118 / 00119 两个分片专供评估**（`prep_4klsdb.py` 内部显式排除，
保证与训练集零重叠），按测试集内容配比取图、中心裁 512、用固定 seed 的 `degrade()` 造 LQ。

产出 **177 对**（building 69 / vegetation 62 / skyline 24 / street 10 / signage 6 / object 6）。

打分用 `eval_p0.py`，指标是**保留增益** `(p_out − p_lq)/(p_gt − p_lq)`，
其中 `p ≈ 13.674·TOPIQ-FR + 4.477·MANIQA − 5.731`。

> **不要用 `p_out/p_lq` 比值** —— 合成评估集上 `p_lq≈0.14`（甚至为负），比值会爆到 13.9 或翻成 −6.1。
> 只有 TOPIQ-FR + MANIQA 可信；LPIPS/DISTS/MUSIQ/NIQE/PSNR/SSIM 实测全都把过锐化的图排在原图之上。

### 已测出来的数

| | 保留增益 | TOPIQ-FR |
|---|---:|---:|
| 未微调的 HYPIR（基线） | **19.0%** | 0.4009 |
| 换本配方后训练 5000 步 | **44.0%** | 0.6050 |

（同一份权重在赛题验证集上是 22.5%，说明这个合成评估集是有效代理。）

---

## 六、依赖

`pip install torch torchvision pillow numpy opencv-python-headless pyarrow huggingface_hub pyiqa`

`hf` CLI 来自 `huggingface_hub`。`iqa.py` 首次运行会联网下 TOPIQ-FR / MANIQA 权重。

## 七、文件一览

```
run_all.sh              一键驱动，STEPS=<层名> 可分步
csig/（本目录）
  degrade.py            退化配方（单图 numpy 版），含感知标定说明
  csig_data.py          Dataset + GPU 批量退化 + 三闸门（hf_max / ijg_quality / screen）
  prep_4klsdb.py        从 parquet 抽 HR + 内容过滤（丢人像）+ 三闸门
  prep_pd12m.py         元数据筛选 -> 边下边过三闸门
  dl_hdrplus.py         GCS 匿名分页拉取 final.jpg
  screen_dir.py         通用：对散图目录跑三闸门（ShopSign / CASIA / 其它）
  build_mix.py          按配比合并成单一 file_list（按权重重复条目实现过采样）
  build_eval.py         造 held-out 评估集
  eval_p0.py            在评估集上给一份 LoRA 权重打分
  iqa.py                TOPIQ-FR + MANIQA 封装
env.sh / run_train.sh   训练环境与看门狗（见上级 FINDINGS.md）
../configs/csig_train.yaml     训练配置
```
