"""恒等 Runner：接口与 model_dir/runner.py 完全一致，但 infer 直接返回输入。

用途：评分逻辑探测实验 3 —— 只改 runner、不改图片，把推理时延压到接近 0，
从而定位综合分里「加速比」一项的上限（是否封顶、封在哪）。

保持与正式 runner 相同的构造签名与字段，避免评测 harness 因缺字段而走异常分支。
"""

from __future__ import annotations

import os

import torch
import yaml


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.config = _load_config(model_dir)
        # WARNING: All models must be using fp16 precision for fair comparisons
        self.weight_dtype = torch.float16

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """[B,3,H,W] in [-1,1] -> 原样返回（恒等映射）。"""
        return image_tensor.to(self.device)
