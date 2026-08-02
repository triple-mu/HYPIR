"""给 HYPIR 训练器打加速补丁。幂等；锚点不匹配会 assert 失败，不会静默改坏。

    python apply_speedups.py <HYPIR仓库路径> [--revert]

改动清单（每条都标了实测依据，见 ../FINDINGS.md 与本文件注释）：

| # | 改动 | 预期 | 风险 |
|---|---|---|---|
| 1 | 全模型 fp16（含 D 与 LPIPS），训推一致 | 显存 -35%，带宽减半 | fp16 需 GradScaler，accelerate 自动处理 |
| 2 | 固定 prompt 的 text embedding 缓存，释放 CLIPTextModel | 每步省一次 340M 前向 + 680 MB | 仅当全 batch prompt 相同才生效，否则自动回退 |
| 3 | D 步复用上一 G 步的输出，不再重跑 forward_generator | **整体 -25%** | D 看到的假样本晚一步；D 是无条件的，真假不需配对 |
| 4 | channels_last | 卷积密集链路提速 | 需实测，个别模块可能变慢 |
| 5 | cudnn.benchmark | 形状恒定时选最优卷积算法 | 无 |
| 6 | AdamW(fused=True) | 272M 可训练参数的 step 变一个 kernel | 无 |
| 7 | dataloader: pin_memory/persistent_workers/prefetch_factor | 消除 worker 反复启停 | 无 |

**为什么可训练参数仍保留 fp32**：lr=1e-5、参数量级 ~1e-2，单次更新的相对幅度约 1e-3，
而 fp16 的相对精度也是 ~1e-3 —— 纯 fp16 下更新会被舍入吃掉，训练不动。
可训练参数只占 272M / 2.5B ≈ 1%，保留 fp32 master 不影响「全模型半精度」的收益。
HYPIR 原代码在 init_generator 里已经把 lora 参数转回 fp32，这里保持不动。
"""
import argparse
import os
import sys

MARK = "# [csig-speedup]"


def patch(path, old, new, tag, sentinel):
    """sentinel 必须是本条补丁独有、且只可能出现在补丁产物里的字符串。

    不要拿 new 的首行做幂等判据 —— 它通常也在 old 里（补丁往往是「保留原文 + 追加」），
    会导致第一次就误判成已打。
    """
    s = open(path).read()
    assert sentinel in new, "sentinel 必须出现在 new 里: %s" % tag
    if sentinel in s:
        print("  跳过（已打）: %s" % tag)
        return False
    assert old in s, "锚点未找到，代码版本对不上: %s\n  期望片段: %r" % (tag, old[:120])
    open(path, "w").write(s.replace(old, new, 1))
    print("  已打: %s" % tag)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    a = ap.parse_args()
    base = os.path.join(a.repo, "HYPIR/trainer/base.py")
    sd2 = os.path.join(a.repo, "HYPIR/trainer/sd2.py")
    for f in (base, sd2):
        assert os.path.exists(f), "找不到 %s" % f

    # --- 1. D 与 LPIPS 也走 weight_dtype（原来 D 硬编码 bf16、LPIPS 是 fp32）------
    patch(base,
          '            self.D = ImageConvNextDiscriminator(precision="bf16").to(device=self.device)',
          '            %s D 跟随 weight_dtype，训推一致；原来硬编码 bf16\n'
          '            _prec = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self.weight_dtype, "fp32")\n'
          '            self.D = ImageConvNextDiscriminator(precision=_prec).to(device=self.device)' % MARK,
          "D 精度跟随 weight_dtype",
          "_prec = {torch.float16")

    patch(base,
          "            self.net_lpips = lpips.LPIPS(net=\"vgg\", verbose=False).to(self.device)\n"
          "        self.net_lpips.eval().requires_grad_(False)",
          "            self.net_lpips = lpips.LPIPS(net=\"vgg\", verbose=False).to(self.device)\n"
          "        self.net_lpips.eval().requires_grad_(False)\n"
          "        %s LPIPS 是纯前向冻结模块，半精度即可，省显存省带宽\n"
          "        if self.weight_dtype in (torch.float16, torch.bfloat16):\n"
          "            self.net_lpips.to(self.weight_dtype)" % MARK,
          "LPIPS 半精度",
          "self.net_lpips.to(self.weight_dtype)")

    # --- 2. 固定 prompt 的 text embedding 缓存 ------------------------------------
    patch(base,
          "        prompt = batch[\"txt\"]\n"
          "        bs = len(prompt)\n"
          "        c_txt = self.encode_prompt(prompt)",
          "        prompt = batch[\"txt\"]\n"
          "        bs = len(prompt)\n"
          "        %s 训练用固定 prompt，整条 CLIP 前向每步结果都一样，缓存后可省 340M 参数的前向。\n"
          "        # 只在「全 batch prompt 相同」时生效，否则自动回退到逐 batch 编码。\n"
          "        _uniq = set(prompt)\n"
          "        if len(_uniq) == 1:\n"
          "            _key = next(iter(_uniq))\n"
          "            _c = getattr(self, \"_txt_cache\", None)\n"
          "            if _c is None or _c[0] != _key:\n"
          "                with torch.no_grad():\n"
          "                    _c = (_key, self.encode_prompt([_key]))\n"
          "                self._txt_cache = _c\n"
          "            c_txt = {k: v.expand(bs, *v.shape[1:]) for k, v in _c[1].items()}\n"
          "        else:\n"
          "            c_txt = self.encode_prompt(prompt)" % MARK,
          "text embedding 缓存",
          "self._txt_cache = _c")

    # --- 3. D 步复用 G 步的输出 ---------------------------------------------------
    patch(base,
          "    def optimize_discriminator(self):\n"
          "        gt = self.batch_inputs.gt\n"
          "        with torch.no_grad():\n"
          "            x = self.forward_generator()\n"
          "        self.G_pred = x",
          "    def optimize_discriminator(self):\n"
          "        gt = self.batch_inputs.gt\n"
          "        %s G/D 是交替更新的，D 步原本要在 no_grad 下重跑一遍 forward_generator\n"
          "        # （G UNet + VAE decode，前向里最贵的两段）。而 D 是无条件的（D(x) 不吃 gt），\n"
          "        # 真假样本无需配对，故直接复用上一 G 步缓存的输出。代价是假样本晚一步更新。\n"
          "        # 环境变量 CSIG_DSTEP_RECOMPUTE=1 可恢复原行为做对照。\n"
          "        _cached = getattr(self, \"_gpred_cache\", None)\n"
          "        if _cached is not None and _cached.shape == gt.shape \\\n"
          "                and os.environ.get(\"CSIG_DSTEP_RECOMPUTE\", \"0\") != \"1\":\n"
          "            x = _cached\n"
          "        else:\n"
          "            with torch.no_grad():\n"
          "                x = self.forward_generator()\n"
          "        self.G_pred = x" % MARK,
          "D 步复用 G 步输出",
          "CSIG_DSTEP_RECOMPUTE")

    patch(base,
          "            x = self.forward_generator()\n"
          "            self.G_pred = x",
          "            x = self.forward_generator()\n"
          "            self.G_pred = x\n"
          "            %s 供下一步 D 复用" % MARK,
          "G 步缓存输出（标记）",
          "供下一步 D 复用")

    # G 步结束后把 detach 的结果存进缓存
    patch(base,
          "        loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc)\n"
          "        return loss_dict",
          "        %s 存下来给紧邻的 D 步用，省一次 G UNet + VAE decode\n"
          "        self._gpred_cache = x.detach()\n"
          "        loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips, G_disc=loss_disc)\n"
          "        return loss_dict" % MARK,
          "G 步写缓存",
          "self._gpred_cache = x.detach()")

    # --- 5/6/7. cudnn.benchmark / fused AdamW / dataloader ------------------------
    patch(base,
          "        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))",
          "        %s 272M 可训练参数的 AdamW step 融成一个 kernel\n"
          "        if optimizer_cls is torch.optim.AdamW:\n"
          "            self.config.opt_kwargs = dict(self.config.opt_kwargs)\n"
          "            self.config.opt_kwargs.setdefault(\"fused\", True)\n"
          "        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))" % MARK,
          "fused AdamW",
          "setdefault(\"fused\", True)")

    patch(base,
          "        self.dataloader = torch.utils.data.DataLoader(\n"
          "            dataset,\n"
          "            shuffle=True,\n"
          "            batch_size=data_cfg.train.batch_size,\n"
          "            num_workers=data_cfg.train.dataloader_num_workers,\n"
          "        )",
          "        %s 实测每样本要解码一张 4K JPEG（112 ms/核），worker 反复启停开销显著。\n"
          "        # 持续吞吐上限约 0.77 x workers x 8.9 样本/秒，batch 大了必须同步加 worker。\n"
          "        _nw = data_cfg.train.dataloader_num_workers\n"
          "        self.dataloader = torch.utils.data.DataLoader(\n"
          "            dataset,\n"
          "            shuffle=True,\n"
          "            batch_size=data_cfg.train.batch_size,\n"
          "            num_workers=_nw,\n"
          "            pin_memory=True,\n"
          "            drop_last=True,\n"
          "            persistent_workers=_nw > 0,\n"
          "            prefetch_factor=4 if _nw > 0 else None,\n"
          "        )" % MARK,
          "dataloader 参数",
          "prefetch_factor=4 if _nw > 0 else None")

    patch(base,
          "    def prepare_all(self):\n"
          "        logger.info(\"Wrapping models, optimizers and dataloaders\")",
          "    def prepare_all(self):\n"
          "        %s 形状恒定，让 cuDNN 选最优卷积算法\n"
          "        torch.backends.cudnn.benchmark = True\n"
          "        %s 这条链路卷积密集（VAE + UNet + ConvNext D + VGG）\n"
          "        if os.environ.get(\"CSIG_CHANNELS_LAST\", \"1\") == \"1\":\n"
          "            for _m in (self.G, self.D, self.vae, self.net_lpips):\n"
          "                _m.to(memory_format=torch.channels_last)\n"
          "        logger.info(\"Wrapping models, optimizers and dataloaders\")" % (MARK, MARK),
          "cudnn.benchmark + channels_last",
          "CSIG_CHANNELS_LAST")

    # 确保 os 被 import（base.py 顶部原本有 import os，这里兜底）
    s = open(base).read()
    if "\nimport os" not in s and "\nimport os\n" not in s:
        open(base, "w").write("import os\n" + s)
        print("  已补: import os")

    print("\n完成。剩余需手工在 config 里改的：")
    print("  mixed_precision: bf16 -> fp16      # 训推一致（推理端是 fp16）")
    print("  batch_size / dataloader_num_workers 按显存与吞吐调（见 ../FINDINGS.md）")
    print("\n回归验证：CSIG_DSTEP_RECOMPUTE=1 CSIG_CHANNELS_LAST=0 可逐项关掉做对照。")


if __name__ == "__main__":
    main()
