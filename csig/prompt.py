"""训练 / 评估 / 提交三处共用的 prompt。三边必须逐字一致，否则 train/infer 条件失配。

**不绑内容**：官方训练数据 LSDIR 的 84,991 张是通用大杂烩（人像、动物、室内都有），
而赛题测试集实测是 building 0.38 / vegetation 0.26 / skyline 0.15 / signage 0.06 /
object 0.05 / street 0.04，且完全没有人像。两边内容分布对不上，写内容的 prompt 在训练集
大多数样本上都是错的描述，会把 text 条件变成噪声。只表达画质意图的句子在两边都成立。

**不用否定词**："no blur / no noise" 这类在 CLIP 文本编码里不会被当成否定，
反而把 blur / noise 这两个概念引进了条件里。

全库单一 prompt 还让 trainer/base.py 的 text-embed 缓存生效，省掉每步一次 CLIP 前向。
"""

PROMPT = "a high quality photograph, sharp and in focus, with fine natural detail and realistic texture"
