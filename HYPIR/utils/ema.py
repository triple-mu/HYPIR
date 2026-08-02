import os
from collections import OrderedDict

import torch


class EMAModel:

    def __init__(self, model, decay=0.999, use_ema=False, ema_resume_pth=None, verbose=False):
        self.use_ema = use_ema
        if self.use_ema:
            self.model = model
            self.decay = decay
            self.ema_state_dict = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state_dict[name] = param.clone().detach()
            self.original_weights = None
            if verbose:
                print(f"Keep EMA Parameters: {len(self.ema_state_dict)}")
            if ema_resume_pth:
                if verbose:
                    print(f"Loading EMA Parameters from {ema_resume_pth}")
                ema_ckpt = torch.load(ema_resume_pth, map_location="cpu")
                for name, param in self.ema_state_dict.items():
                    if name in ema_ckpt:
                        _param = ema_ckpt[name].to(param.dtype).to(param.device)
                        self.ema_state_dict[name].copy_(_param.data)

    def update(self):
        if not self.use_ema:
            return
        # [csig-speedup] 逐参数的 lerp 在 259M/514 张量上要 21 ms/步；
        # torch._foreach_lerp_ 批量化后 0.85 ms，数学等价（ema += (1-d)*(p-ema)）
        #
        # 两个列表必须只建一次。dict(named_parameters()) 是一次完整模块树遍历
        # （SD2.1 UNet ~686 模块 / ~1000 参数），写进列表推导式里就是每个 key 遍历一遍：
        # 514 x ~1700 次 Python 操作 ≈ 每步 0.9 s，实测占了 2.65 s/步里的一大块，
        # py-spy 上表现为 70% 采样落在 named_modules / _named_members。
        # model 是 unwrap 后的裸 nn.Module，nn.Parameter 对象在训练中不会被替换，缓存安全。
        if not hasattr(self, "_ema_ps"):
            named = dict(self.model.named_parameters())
            keys = [n for n in self.ema_state_dict if n in named]
            self._ema_ps = [named[n] for n in keys]
            self._ema_es = [self.ema_state_dict[n] for n in keys]
        with torch.no_grad():
            # detach 每步重做：只是 514 次共享 storage 的浅包装，可忽略，
            # 但能挡住优化器万一重新赋值 .data 导致缓存视图失效
            torch._foreach_lerp_(self._ema_es, [p.detach() for p in self._ema_ps], 1 - self.decay)

    def activate_ema_weights(self):
        if not self.use_ema:
            return
        self.original_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.clone().detach()
                param.data.copy_(self.ema_state_dict[name].data)

    def deactivate_ema_weights(self):
        if not self.use_ema:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_weights[name].data)
        self.original_weights = None

    def save_ema_weights(self, save_dir):
        if not self.use_ema:
            return
        save_path = os.path.join(save_dir, "ema_state_dict.pth")
        torch.save(self.ema_state_dict, save_path)
