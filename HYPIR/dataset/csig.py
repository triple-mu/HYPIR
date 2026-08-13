"""CSIG-2026 training input pipeline.

The Dataset performs cheap byte/header reads in workers.  The batch transform
decodes on the training device, crops native-resolution pixels, samples a
replayable degradation record, and invokes :mod:`csig_degradation`.

File-list lines are backwards compatible and may optionally carry a profile:

    /path/to/image.jpg
    /path/to/night.jpg<TAB>night_hdr
    relative/day.png<TAB>ordinary

Content is dispatched by magic bytes, never by filename extension.  This matters
for the supplied validation set, which contains both JPEG-as-.png and PNG-as-.jpg.
"""

from __future__ import annotations

import os
import struct
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image, decode_jpeg

from HYPIR.dataset import csig_degradation as _degradation
from HYPIR.dataset.csig_degradation import (
    DEFAULT_CONFIG,
    DegradationConfig,
    DegradationSampler,
    TensorDegrader,
    degrade_tensor,
    sample_degradation,
)

# Re-export historical constants without a second source of truth.
for _compat_name in _degradation.__all__:
    globals().setdefault(_compat_name, getattr(_degradation, _compat_name))


_VALID_PROFILES = {"ordinary", "night_hdr"}


def _magic_kind(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def _image_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Return (width, height) from JPEG/PNG/WebP headers without pixel decode."""

    kind = _magic_kind(data)
    if kind == "png" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if kind == "jpeg":
        i, n = 2, len(data)
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i + 4 <= n:
            while i < n and data[i] != 0xFF:
                i += 1
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                break
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > n:
                break
            length = struct.unpack(">H", data[i:i + 2])[0]
            if length < 2 or i + length > n:
                break
            if marker in sof and length >= 7:
                height, width = struct.unpack(">HH", data[i + 3:i + 7])
                return int(width), int(height)
            i += length
    # WebP dimensions have several chunk-specific layouts; defer to decode_image.
    return None


@dataclass(frozen=True)
class _Record:
    path: str
    profile_hint: Optional[str]


def _parse_record(line: str) -> _Record:
    fields = line.rstrip("\n").split("\t")
    if not fields[0].strip():
        raise ValueError("empty image path")
    if len(fields) > 2:
        raise ValueError(f"file-list line has more than two tab fields: {line!r}")
    hint = fields[1].strip() if len(fields) == 2 and fields[1].strip() else None
    if hint not in _VALID_PROFILES | {None}:
        raise ValueError(f"invalid profile hint {hint!r} in line: {line!r}")
    return _Record(fields[0].strip(), hint)


class CSIGDataset(Dataset):
    """Read encoded bytes; reject invalid/small HQ sources before GPU decode."""

    def __init__(
        self,
        file_list: str,
        out_size: int = 512,
        prompt: str = "",
        image_path_prefix: str = "",
    ) -> None:
        with open(file_list, "r", encoding="utf-8") as handle:
            self.records = [_parse_record(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"empty file_list: {file_list}")
        self.prefix = image_path_prefix
        self.out_size = int(out_size)
        if self.out_size <= 0:
            raise ValueError("out_size must be positive")
        self.prompt = prompt

    @property
    def paths(self) -> List[str]:
        """Compatibility view used by older corpus scripts."""

        return [r.path for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.prefix, path)

    def __getitem__(self, idx: int) -> Dict[str, Union[bytes, str, None]]:
        errors = []
        for _ in range(10):
            record = self.records[idx]
            path = self._resolve(record.path)
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
                kind = _magic_kind(data)
                if kind == "unknown":
                    raise ValueError("unsupported or corrupt image magic")
                size = _image_size(data)
                if size is not None and min(size) < self.out_size:
                    raise ValueError(f"native image {size[0]}x{size[1]} is smaller than crop {self.out_size}")
                if len(data) <= 32:
                    raise ValueError("encoded image is too short")
                return {
                    "image_bytes": data,
                    "jpeg": data,  # legacy key; both point at the same immutable bytes
                    "txt": self.prompt,
                    "profile_hint": record.profile_hint,
                    "source_path": path,
                }
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                # Deterministic fallback keeps corrupt-file handling from
                # perturbing Python's global RNG or making a run unreplayable.
                idx = (idx + 1) % len(self.records)
        raise RuntimeError("10 consecutive image reads failed; last errors: " + " | ".join(errors[-3:]))


def csig_collate(batch: List[Dict]) -> Dict:
    """Keep variable-length bytes and strings as lists for Accelerate."""

    key = "image_bytes" if "image_bytes" in batch[0] else "jpeg"
    result = {
        "image_bytes": [item[key] for item in batch],
        "txt": [item["txt"] for item in batch],
        "profile_hint": [item.get("profile_hint") for item in batch],
        "source_path": [item.get("source_path") for item in batch],
    }
    # Preserve the old hq_key="jpeg" configuration without duplicating storage.
    result["jpeg"] = result["image_bytes"]
    return result


class CSIGBatchTransform:
    """Decode, native-crop and apply the measured multi-profile degradation."""

    def __init__(
        self,
        hq_key: str = "image_bytes",
        extra_keys: Optional[List[str]] = None,
        out_size: int = 512,
        jpeg_quality: Optional[int] = None,
        use_hflip: bool = True,
        profile: str = "auto",
        profile_probs: Optional[Mapping[str, float]] = None,
        return_metadata: bool = False,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if profile not in ("auto", "ordinary", "night_hdr"):
            raise ValueError(f"unknown degradation profile: {profile!r}")
        self.hq_key = hq_key
        self.extra_keys = extra_keys or ["txt"]
        self.out_size = int(out_size)
        self.jpeg_quality = jpeg_quality
        self.use_hflip = bool(use_hflip)
        self.profile = profile
        self.return_metadata = bool(return_metadata)
        self.device_override = device
        self.config = DEFAULT_CONFIG.with_profile_probs(profile_probs)
        self.sampler = DegradationSampler(seed=seed, config=self.config)
        self.degrader = TensorDegrader()

    @property
    def jpeger(self):
        """Deprecated compatibility handle for historical analysis scripts."""

        return getattr(self, "_legacy_jpeger", None)

    @jpeger.setter
    def jpeger(self, value):
        self._legacy_jpeger = value

    def _device(self) -> torch.device:
        if self.device_override is not None:
            return torch.device(self.device_override)
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda", torch.cuda.current_device())

    def _decode(self, bufs: Sequence[bytes], paths: Sequence[Optional[str]], device: torch.device) -> List[torch.Tensor]:
        cpu_bufs = [torch.frombuffer(bytearray(data), dtype=torch.uint8) for data in bufs]
        images: List[Optional[torch.Tensor]] = [None] * len(bufs)
        jpeg_indices = [i for i, data in enumerate(bufs) if _magic_kind(data) == "jpeg"]
        if jpeg_indices:
            selected = [cpu_bufs[i] for i in jpeg_indices]
            try:
                decoded = decode_jpeg(selected, device=device, mode=ImageReadMode.RGB)
            except Exception as exc:
                names = [paths[i] for i in jpeg_indices]
                raise ValueError(f"JPEG decode failed for {names}: {exc}") from exc
            for i, image in zip(jpeg_indices, decoded):
                images[i] = image
        for i, data in enumerate(bufs):
            if images[i] is not None:
                continue
            kind = _magic_kind(data)
            if kind not in ("png", "webp"):
                raise ValueError(f"unsupported/corrupt image {paths[i]} (magic={kind})")
            try:
                images[i] = decode_image(cpu_bufs[i], mode=ImageReadMode.RGB).to(device, non_blocking=True)
            except Exception as exc:
                raise ValueError(f"{kind.upper()} decode failed for {paths[i]}: {exc}") from exc
        return [image for image in images if image is not None]

    def _decode_crop(
        self,
        bufs: Sequence[bytes],
        paths: Sequence[Optional[str]],
        metadata: Sequence[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        images = self._decode(bufs, paths, device)
        if len(images) != len(bufs):
            raise RuntimeError("decoder returned an incomplete batch")
        crops = []
        size = self.out_size
        for image, meta, path in zip(images, metadata, paths):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"expected RGB CHW image for {path}, got {tuple(image.shape)}")
            _, height, width = image.shape
            if height < size or width < size:
                raise ValueError(
                    f"native image {path} is {width}x{height}, smaller than {size}; "
                    "HQ sources are never silently upscaled")
            crop_rng = np.random.default_rng(int(meta["sample_seed"]) ^ 0xC20F2026)
            y = int(crop_rng.integers(0, height - size + 1))
            x = int(crop_rng.integers(0, width - size + 1))
            hflip = bool(self.use_hflip and crop_rng.random() < 0.5)
            crop = image[:, y:y + size, x:x + size]
            if hflip:
                crop = torch.flip(crop, dims=[2])
            crops.append(crop)
            meta["crop_xy"] = [x, y]
            meta["original_size"] = [width, height]
            meta["hflip"] = hflip
            meta["source_path"] = path
        return torch.stack(crops).float().div_(255.0)

    # Legacy helpers retained for old analysis scripts.  They are not used by the
    # training path and delegate to the shared mathematical implementation.
    def _lowpass(self, x, sigma, aniso, theta, pole):
        outs = []
        for i in range(x.shape[0]):
            a = float(aniso[i])
            meta = {
                "sigma_x": float(sigma[i]) * math.sqrt(a),
                "sigma_y": float(sigma[i]) / math.sqrt(a),
                "theta": float(theta[i]),
                "pole": float(pole[i]),
                "kernel_family": "heavy_tail",
            }
            outs.append(self.degrader._lowpass_response(x[i:i + 1], meta))
        return torch.cat(outs)

    def _resample_chain(self, x):
        """Deprecated compatibility helper using the current measured scale union."""

        outputs = []
        sampler = DegradationSampler(seed=int(torch.initial_seed()), config=self.config)
        for i in range(x.shape[0]):
            meta = sampler.sample(profile="ordinary")
            # Force a resampling operator while preserving the sampled severity ranges.
            if meta["resize_scale"] is None:
                from HYPIR.dataset.csig_degradation import _ORDINARY_RANGES
                rng = np.random.default_rng(int(meta["sample_seed"]))
                bounds = _ORDINARY_RANGES[meta["severity"]]["scale"]
                meta["resize_scale"] = float(rng.uniform(*bounds))
                meta["down_mode"] = str(rng.choice(("area", "bilinear", "bicubic")))
                meta["down_antialias"] = None if meta["down_mode"] == "area" else bool(rng.random() < 0.5)
                meta["up_mode"] = str(rng.choice(("bilinear", "bicubic")))
            outputs.append(self.degrader._resample(x[i:i + 1], meta))
        return torch.cat(outputs)

    @torch.no_grad()
    def __call__(self, batch: Dict) -> Dict:
        bufs = batch.get(self.hq_key)
        if bufs is None and self.hq_key == "image_bytes":
            bufs = batch.get("jpeg")
        if not isinstance(bufs, (list, tuple)) or not bufs:
            raise ValueError(f"batch[{self.hq_key!r}] must be a non-empty list of encoded bytes")
        hints = batch.get("profile_hint", [None] * len(bufs))
        paths = batch.get("source_path", [None] * len(bufs))
        metadata = self.sampler.sample_batch(
            len(bufs),
            profile=self.profile,
            profile_hints=hints,
            jpeg_quality=self.jpeg_quality,
        )
        device = self._device()
        hq = self._decode_crop(bufs, paths, metadata, device)
        lq = self.degrader.apply(hq, metadata)

        result = {"GT": hq, "LQ": lq}
        for key in self.extra_keys:
            if key not in batch:
                raise KeyError(f"missing requested extra key: {key}")
            result[key] = batch[key]
        if self.return_metadata:
            result["degradation_meta"] = metadata
        return result


__all__ = [
    "CSIGBatchTransform", "CSIGDataset", "csig_collate", "_image_size", "_magic_kind",
] + [name for name in globals() if name.isupper()] + [
    "DEFAULT_CONFIG", "DegradationConfig", "DegradationSampler", "TensorDegrader",
    "degrade_tensor", "sample_degradation",
]
