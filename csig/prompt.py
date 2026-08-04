"""训练 / 评估 / 提交三处共用的 prompt。三边必须逐字一致，否则 train/infer 条件失配。

**结论：空串**，也就是官方 up/main:README.md「Train」一节给的复现配方
（parquet 里 prompt 全填 ""，配 p_empty_prompt: 0.0）。

先按「不绑内容、只表达画质意图」的思路拟了一句 19 token 的通用 prompt，实测被否了。
用官方 HYPIR_sd2.pth 在 DIV2K_V2_val 的 60 对上零样本对照（GT 自身 MUSIQ 63.04 /
MANIQA 0.4068 / CLIP-IQA 0.5901 作参照）：

| prompt                                        | LPIPS↓ | MUSIQ | MANIQA | CLIP-IQA | PSNR  |
|-----------------------------------------------|-------:|------:|-------:|---------:|------:|
| ""（官方）                                     | 0.3189 | 65.55 | 0.4765 |   0.6246 | 20.61 |
| "a photo"                                      | 0.3170 | 64.26 | 0.4620 |   0.6110 | 20.77 |
| "a detailed high quality photo"                | 0.3176 | 62.62 | 0.4419 |   0.5929 | 20.92 |
| "a high quality photograph, sharp and in ..."  | 0.3215 | 59.79 | 0.4106 |   0.5703 | 21.16 |

单调趋势：画质形容词加得越多，感知分越低、PSNR 越高 —— 这些词让模型更保守，
而不是更敢生成细节。LPIPS 四者基本打平。空串既是官方配方，又是感知指标上最好的起点，
而且它本来就是「不绑内容」的极端情形，符合最初的意图。

（注意这是零样本口径：官方权重是拿 LLaVA caption 训的。我们从官方权重续训，
所以起点好不好确实要紧；但训完之后模型会适配所训的那个固定 prompt。）

空 prompt 也让 trainer/base.py 的 text-embed 缓存生效，省掉每步一次 CLIP 前向。
"""

PROMPT = ""
