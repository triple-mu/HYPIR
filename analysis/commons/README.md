# 数据源采集与调研脚本

## 采集（有实际用途，不是一次性）

| 脚本 | 用途 |
|---|---|
| `commons_harvest.py` | **主力**。Wikimedia Commons 三段式：`index`（CirrusSearch 分页出文件名）→ `filter`（按许可/bpp/品牌筛）→ `download`。预设 `raster`（4096x3072 原生栅格 + 中国品牌）/ `huawei` / `china` |
| `fetch_commons.py` | 按文件名列表直下 Commons 原图。注意它走 `wsrv.nl` 代理而非 `upload.wikimedia.org` 直连 —— 本机网络策略下直连会超时 |
| `rzip.py` | **用 HTTP Range 从远程 zip 里抽少量文件，不下整包**。66 行。调研 SIDD（几十 GB）时抽样用的，对付任何大 zip 都好使 |
| `sidd_more.sh` | `rzip.py` 的并发调用示例 |

## 一次性度量（结论已进 docs/，脚本留作复现）

`measure*.py` / `count_models.py` / `allcats.py` / `wq.py` / `probe.py`
—— 量各候选源的分辨率分布、bpp、EXIF 机型构成、许可占比。

## ⚠️ 索引文件已丢失

`idx_*.jsonl` / `cand_*.jsonl`（Commons 检索结果）在一次工作区清理中被误删，
只剩 `*_test.jsonl` 两个小样本。**重建方式**：

```bash
python commons_harvest.py index  --out idx_raster.jsonl --preset raster --repeat 2
python commons_harvest.py filter --idx idx_raster.jsonl --out cand_raster.jsonl
```

代价是重跑一遍 API 分页。已实测的产出规模（供核对）：
- `raster` 预设：index 5,760 唯一文件 → filter 4,978 候选 → 三闸门 80% → 预计 ~3,980 张，27 GB / 100 min
- `huawei`（含荣耀）：≈12,254 命中 → ~8,800 候选 → ≈7,000 张，46 GB / 3 h
- `china`（33 省）：2,962 → 2,210 候选 → 三闸门 82.8% → ≈1,830 张，11 GB / 45 min

⚠️ `filew:>2999` 那条实测 9,942，逼近 CirrusSearch `sroffset` 10000 硬上限；
脚本会告警，届时按品牌拆成两条分别跑。UA 必须写成 `Bot/0.0 (URL; email)` 格式，否则 403。
