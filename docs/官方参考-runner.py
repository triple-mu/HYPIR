import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch
import yaml
from torchvision import transforms
import time

logger = logging.getLogger(__name__)

# RAM image preprocessing
_ram_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _load_ram_tag_params(model_dir: str, device: torch.device):
    pth_path = os.path.join(model_dir, "ram_tag_params.pth")
    if os.path.exists(pth_path):
        return torch.load(pth_path, map_location=device, weights_only=False)
    return {}


class Runner:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.config = _load_config(model_dir)

        # WARNING: All models must be using fp16 precision for fair comparisons
        self.weight_dtype = torch.float16

        models_cfg = self.config.get("models", {})

        # Load JIT models
        self.core_model = self._load_jit(
            models_cfg.get("osediff_core", "osediff_core.pt")
        )
        self.clip_text_encoder = self._load_jit(
            models_cfg.get("clip_text_encoder", "clip_text_encoder.pt")
        )
        self.ram_encoder = self._load_jit(
            models_cfg.get("ram_dape_encoder", "ram_dape_encoder.pt")
        )
        self.ram_tag_decoder = self._load_jit(
            models_cfg.get("ram_tag_decoder", "ram_tag_decoder.pt")
        )

        # Load tag params
        tag_params = _load_ram_tag_params(model_dir, self.device)
        self.class_threshold = tag_params.get("class_threshold", None)
        self.tag_list = tag_params.get("tag_list", [])
        self.delete_tag_index = tag_params.get("delete_tag_index", [])
        self.num_class = tag_params.get("num_class", 4584)

        self._tokenizer = None
        self._init_tokenizer()

    def _load_jit(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            return model
        raise Exception(f"JIT model not found: {path}")

    def _init_tokenizer(self):
        pretrained_path = self.config.get("pretrained_model_name_or_path", "")
        if pretrained_path:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.model_dir, pretrained_path), subfolder="tokenizer"
            )
        else:
            raise RuntimeError(
                "Cannot find tokenizer. Set pretrained_model_name_or_path in config.yaml."
            )

    def _generate_tags(self, image_tensor: torch.Tensor) -> str:
        """Run RAM/DAPE tagging pipeline on [0,1] image tensor.

        Keeps RAM internal preprocessing (Resize + Normalize) inside.
        """
        if self.ram_encoder is None or self.ram_tag_decoder is None:
            raise RuntimeError("ram_encoder or ram_tag_decoder is None")

        # RAM internal preprocessing: Resize(384) + ImageNet normalize
        lq_ram = _ram_transforms(image_tensor).to(dtype=self.weight_dtype)
        
        image_embeds = self.ram_encoder(lq_ram)
        logits = self.ram_tag_decoder(image_embeds)

        # Decode logits to tag string
        if self.class_threshold is not None:
            ct = self.class_threshold.to(device=logits.device, dtype=logits.dtype)
            targets = torch.where(
                torch.sigmoid(logits) > ct,
                torch.tensor(1.0, device=logits.device),
                torch.tensor(0.0, device=logits.device),
            )
        else:
            targets = torch.where(
                torch.sigmoid(logits) > 0.68,
                torch.tensor(1.0, device=logits.device),
                torch.tensor(0.0, device=logits.device),
            )

        tag_np = targets.cpu().numpy()
        if self.delete_tag_index:
            tag_np[:, self.delete_tag_index] = 0

        tag_output = []
        for b in range(tag_np.shape[0]):
            index = np.argwhere(tag_np[b] == 1)
            if len(index) > 0 and len(self.tag_list) > 0:
                tokens = [
                    self.tag_list[i[0]] for i in index if i[0] < len(self.tag_list)
                ]
                tag_output.append(", ".join(tokens))
            else:
                tag_output.append("")

        return tag_output[0] if tag_output else ""

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        """Encode text to CLIP prompt embeddings [1, 77, 1024]."""
        if self.clip_text_encoder is None:
            raise RuntimeError("clip_text_encoder is None")

        with torch.no_grad():
            text_input_ids = self._tokenizer(
                prompt_text,
                max_length=self._tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            prompt_embeds = self.clip_text_encoder(text_input_ids)

        return prompt_embeds.to(dtype=self.weight_dtype)

    @torch.no_grad()
    def infer(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """Run full OSEDiff inference pipeline.

        Args:
            image_tensor: [1, 3, H, W] tensor in [-1, 1] range.
            prompt: Optional user prompt.

        Returns:
            output_tensor: [1, 3, H, W] tensor in [-1, 1] range.
        """
        # Step 1: Generate tags from input image
        image_for_tag = image_tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        tag_text = self._generate_tags(image_for_tag)

        # Step 2: Build full prompt
        full_prompt = f"{tag_text}, {prompt}," if tag_text else prompt

        # Step 3: Encode prompt
        prompt_embeds = self._encode_prompt(full_prompt)

        # Step 4: Run core model (VAE + UNet + DDPM)
        output_tensor = self.core_model(image_tensor, prompt_embeds)

        return output_tensor