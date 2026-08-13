"""Measured, replayable degradation mixture for CSIG-2026 track 2.

The three public building blocks deliberately separate sampling from execution:

``sample_degradation``
    Pure NumPy parameter sampling.  The returned metadata is strict-JSON-safe and
    is the complete replay token for stochastic masks/noise via ``sample_seed``.

``TensorDegrader``
    Apply sampled metadata to RGB BCHW tensors in [0, 1].  It supports both CPU
    and CUDA and never mutates the input tensor.

``degrade_tensor``
    Convenience wrapper used by analysis tools and the training batch transform.

This is a *distribution inferred from the supplied data*, not a claim about the
organiser's exact ISP.  In particular, down/up scales are equivalent fits rather
than recovered camera zoom factors.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "csig-mixture-v1"

PROFILE_PROBS = {"ordinary": 0.90, "night_hdr": 0.10}
SEVERITY_PROBS = {"weak": 0.20, "medium": 0.50, "strong": 0.30}
OPERATOR_PROBS = {
    "heavy_tail": 0.45,
    "aniso_gaussian": 0.20,
    "resample": 0.25,
    "combined": 0.10,
}
NIGHT_OPERATOR_PROBS = {
    "heavy_tail": 0.25,
    "aniso_gaussian": 0.15,
    "resample": 0.25,
    "combined": 0.35,
}

# Backwards-compatible names used by a few analysis scripts.  New code should
# read the sampled metadata rather than treating these union bounds as one box.
SIGMA_RANGE = (0.2, 6.0)
ANISO_RANGE = (1.0, 1.5)
POLE_RANGE = (0.9, 2.1)
SPATIAL_VAR = 0.4
SCALE_RANGE = (1.0, 7.0)
JPEG_QUALITY = 95
JPEG_RANGE = (95, 95)
NOISE_SIGMA = (0.0, 0.0)
P_AFFINE = 0.40
P_RESAMPLE = OPERATOR_PROBS["resample"] + OPERATOR_PROBS["combined"]
P_CLEAN = SEVERITY_PROBS["weak"] * 0.5
AFFINE_A = (0.88, 1.22)
AFFINE_B = (-26.0 / 255.0, 12.0 / 255.0)
AFFINE_CHROMA = 0.035
AFFINE_PIVOT = 0.55
AFFINE_B_JITTER = 4.0 / 255.0


# Exact natural-order tables observed in the supplied ordinary validation/test
# JPEGs (IJG Q95) and in all nine Huawei MPO main/gain-map frames.
IJG_Q95_Y = (
    2, 1, 1, 2, 2, 4, 5, 6, 1, 1, 1, 2, 3, 6, 6, 6,
    1, 1, 2, 2, 4, 6, 7, 6, 1, 2, 2, 3, 5, 9, 8, 6,
    2, 2, 4, 6, 7, 11, 10, 8, 2, 4, 6, 6, 8, 10, 11, 9,
    5, 6, 8, 9, 10, 12, 12, 10, 7, 9, 10, 10, 11, 10, 10, 10,
)
IJG_Q95_C = (
    2, 2, 2, 5, 10, 10, 10, 10, 2, 2, 3, 7, 10, 10, 10, 10,
    2, 3, 6, 10, 10, 10, 10, 10, 5, 7, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
)
IJG_BASE_Y = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
IJG_BASE_C = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
)
HUAWEI_MPO_Y = (
    3, 3, 4, 4, 3, 9, 19, 9, 2, 4, 4, 6, 6, 9, 9, 15,
    1, 4, 7, 6, 14, 13, 18, 24, 4, 4, 6, 12, 8, 9, 27, 13,
    6, 4, 4, 13, 13, 14, 27, 20, 17, 8, 11, 21, 17, 15, 27, 31,
    15, 14, 21, 12, 22, 30, 36, 28, 12, 25, 22, 30, 26, 37, 22, 36,
)
HUAWEI_MPO_C = (
    3, 3, 9, 8, 11, 8, 24, 28, 4, 3, 15, 11, 14, 16, 23, 24,
    3, 21, 12, 23, 15, 25, 24, 22, 20, 9, 14, 27, 18, 27, 19, 24,
    11, 8, 11, 15, 25, 23, 36, 38, 12, 19, 15, 20, 15, 27, 30, 35,
    28, 27, 18, 23, 27, 24, 35, 1, 17, 20, 15, 26, 34, 33, 31, 36,
)


def _normalise_probs(name: str, probs: Mapping[str, float], allowed: Iterable[str]) -> Dict[str, float]:
    allowed_keys = tuple(allowed)
    allowed_set = set(allowed_keys)
    unknown = set(probs) - allowed_set
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
    # Preserve declared order: sampling must not depend on PYTHONHASHSEED.
    out = {k: float(probs.get(k, 0.0)) for k in allowed_keys}
    if any(not math.isfinite(v) or v < 0 for v in out.values()):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    total = sum(out.values())
    if total <= 0:
        raise ValueError(f"{name} probabilities sum to zero")
    return {k: v / total for k, v in out.items()}


def _choice(rng: np.random.Generator, probs: Mapping[str, float]) -> str:
    keys = tuple(probs)
    values = np.asarray([probs[k] for k in keys], dtype=np.float64)
    values /= values.sum()
    return keys[int(rng.choice(len(keys), p=values))]


def _u(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


@dataclass(frozen=True)
class DegradationConfig:
    """Distribution configuration; tuples are inclusive sampling bounds."""

    profile_probs: Mapping[str, float] = None
    severity_probs: Mapping[str, float] = None
    operator_probs: Mapping[str, float] = None
    night_operator_probs: Mapping[str, float] = None
    near_native_given_weak: float = 0.50
    spatial_variation_prob: float = 0.50
    spatial_variation: float = 0.40
    anisotropy: Tuple[float, float] = (1.0, 1.5)
    pole: Tuple[float, float] = (0.9, 2.1)
    pre_jpeg_prob: float = 0.10
    subpixel_prob: float = 0.20
    edge_aware_prob: float = 0.35
    unsharp_prob: float = 0.10
    fusion_prob: float = 0.05
    local_warp_prob: float = 0.10
    chroma_prob: float = 0.10
    night_fusion_prob: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_probs", _normalise_probs(
            "profile_probs", self.profile_probs or PROFILE_PROBS, PROFILE_PROBS))
        object.__setattr__(self, "severity_probs", _normalise_probs(
            "severity_probs", self.severity_probs or SEVERITY_PROBS, SEVERITY_PROBS))
        object.__setattr__(self, "operator_probs", _normalise_probs(
            "operator_probs", self.operator_probs or OPERATOR_PROBS, OPERATOR_PROBS))
        object.__setattr__(self, "night_operator_probs", _normalise_probs(
            "night_operator_probs", self.night_operator_probs or NIGHT_OPERATOR_PROBS,
            NIGHT_OPERATOR_PROBS))
        for name in (
            "near_native_given_weak", "spatial_variation_prob", "pre_jpeg_prob",
            "subpixel_prob", "edge_aware_prob", "unsharp_prob", "fusion_prob",
            "local_warp_prob", "chroma_prob", "night_fusion_prob",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.spatial_variation < 1.0:
            raise ValueError("spatial_variation must be in [0, 1)")

    def with_profile_probs(self, probs: Optional[Mapping[str, float]]) -> "DegradationConfig":
        return self if probs is None else replace(self, profile_probs=probs)


DEFAULT_CONFIG = DegradationConfig()


_ORDINARY_RANGES = {
    "weak": {"sigma": (0.2, 1.2), "scale": (1.0, 1.5)},
    "medium": {"sigma": (1.2, 3.5), "scale": (1.5, 4.0)},
    "strong": {"sigma": (3.0, 6.0), "scale": (4.0, 7.0)},
}
_NIGHT_RANGES = {
    "weak": {"sigma": (1.0, 2.0), "scale": (2.0, 3.0)},
    "medium": {"sigma": (1.8, 3.8), "scale": (2.5, 4.5)},
    "strong": {"sigma": (3.5, 5.0), "scale": (4.0, 6.0)},
}


def _empty_metadata(sample_seed: int, profile: str, profile_source: str) -> Dict:
    # Fixed schema makes JSONL analysis straightforward.  Inapplicable values are
    # None rather than NaN so strict JSON encoders and non-Python readers agree.
    return {
        "degradation_schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "sample_seed": int(sample_seed),
        "profile": profile,
        "profile_source": profile_source,
        "severity": None,
        "operator": None,
        "kernel_family": None,
        "sigma_x": None,
        "sigma_y": None,
        "theta": None,
        "pole": None,
        "spatial_variation": 0.0,
        "resize_scale": None,
        "down_mode": None,
        "down_antialias": None,
        "up_mode": None,
        "pre_jpeg_quality": None,
        "subpixel_shift_xy": None,
        "edge_aware_strength": 0.0,
        "unsharp_amount": 0.0,
        "fusion_shift_xy": None,
        "fusion_opacity": 0.0,
        "local_warp_amplitude": 0.0,
        "tone_mode": "identity",
        "tone_gain": [1.0, 1.0, 1.0],
        "tone_bias": [0.0, 0.0, 0.0],
        "tone_gamma": 1.0,
        "highlight_compression": 0.0,
        "chroma_blur": 0.0,
        "chroma_shift_xy": None,
        "saturation": 1.0,
        "jpeg_quality": None,
        "jpeg_subsampling": "4:2:0",
        "jpeg_qtable_profile": None,
        "jpeg_backend": None,
        "noise_sigma": 0.0,
        "shot_noise": 0.0,
        "read_noise": 0.0,
        "denoise_strength": 0.0,
        "motion_length": 0.0,
        "motion_theta": None,
        "bloom_strength": 0.0,
        "crop_xy": None,
        "original_size": None,
        "hflip": False,
        "source_path": None,
    }


def sample_degradation(
    rng: np.random.Generator,
    *,
    profile: str = "auto",
    profile_hint: Optional[str] = None,
    config: DegradationConfig = DEFAULT_CONFIG,
    sample_seed: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
) -> Dict:
    """Sample one strict-JSON metadata record.

    ``profile_hint`` is authoritative for tagged file-list entries.  ``profile``
    may force a branch, while ``auto`` samples the configured 90/10 mixture.
    """

    if profile not in ("auto", "ordinary", "night_hdr"):
        raise ValueError(f"unknown degradation profile: {profile!r}")
    if profile_hint not in (None, "ordinary", "night_hdr"):
        raise ValueError(f"unknown profile hint: {profile_hint!r}")
    if sample_seed is None:
        sample_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
    child = np.random.default_rng(int(sample_seed))

    if profile_hint is not None:
        actual, source = profile_hint, "hint"
    elif profile != "auto":
        actual, source = profile, "forced"
    else:
        actual, source = _choice(child, config.profile_probs), "sampled"
    meta = _empty_metadata(int(sample_seed), actual, source)

    severity = _choice(child, config.severity_probs)
    meta["severity"] = severity
    if actual == "ordinary":
        _sample_ordinary(child, meta, severity, config)
    else:
        _sample_night(child, meta, severity, config)

    if jpeg_quality is not None:
        quality = int(jpeg_quality)
        if not 1 <= quality <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {quality}")
        meta.update(
            jpeg_quality=quality,
            jpeg_qtable_profile="ijg_dynamic",
            jpeg_backend="diffjpeg_exact_qtable",
        )

    # Catch NumPy scalars, NaN, accidental tensors, and schema regressions now,
    # close to their source rather than after a multi-hour training run.
    json.dumps(meta, allow_nan=False)
    return meta


def _sample_spatial(
    rng: np.random.Generator,
    meta: Dict,
    severity: str,
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    operator_probs: Mapping[str, float],
    config: DegradationConfig,
) -> None:
    operator = _choice(rng, operator_probs)
    meta["operator"] = operator
    bounds = ranges[severity]

    if operator in ("heavy_tail", "aniso_gaussian", "combined"):
        sigma = _u(rng, bounds["sigma"])
        aniso = _u(rng, config.anisotropy)
        meta["kernel_family"] = "heavy_tail" if operator != "aniso_gaussian" else "gaussian"
        meta["sigma_x"] = float(sigma * math.sqrt(aniso))
        meta["sigma_y"] = float(sigma / math.sqrt(aniso))
        meta["theta"] = _u(rng, (0.0, math.pi))
        meta["pole"] = _u(rng, config.pole) if meta["kernel_family"] == "heavy_tail" else None
        if rng.random() < config.spatial_variation_prob:
            meta["spatial_variation"] = float(config.spatial_variation)
    if operator in ("resample", "combined"):
        meta["resize_scale"] = _u(rng, bounds["scale"])
        meta["down_mode"] = str(rng.choice(
            ("area", "bilinear", "bicubic"), p=(0.45, 0.30, 0.25)))
        meta["down_antialias"] = None if meta["down_mode"] == "area" else bool(rng.random() < 0.5)
        meta["up_mode"] = str(rng.choice(
            ("bilinear", "bicubic", "lanczos"), p=(0.55, 0.40, 0.05)))


def _sample_ordinary(
    rng: np.random.Generator,
    meta: Dict,
    severity: str,
    config: DegradationConfig,
) -> None:
    near_native = severity == "weak" and rng.random() < config.near_native_given_weak
    if near_native:
        meta["operator"] = "near_native"
    else:
        _sample_spatial(rng, meta, severity, _ORDINARY_RANGES, config.operator_probs, config)

        if rng.random() < config.pre_jpeg_prob:
            meta["pre_jpeg_quality"] = int(rng.integers(95, 101))
        if rng.random() < config.subpixel_prob:
            meta["subpixel_shift_xy"] = [_u(rng, (-0.4, 0.4)), _u(rng, (-0.4, 0.4))]
        if rng.random() < config.edge_aware_prob:
            meta["edge_aware_strength"] = _u(rng, (0.15, 0.40))
        if rng.random() < config.unsharp_prob:
            meta["unsharp_amount"] = _u(rng, (0.02, 0.25))
        if rng.random() < config.fusion_prob:
            length, theta = _u(rng, (0.5, 2.0)), _u(rng, (0.0, 2 * math.pi))
            meta["fusion_shift_xy"] = [length * math.cos(theta), length * math.sin(theta)]
            meta["fusion_opacity"] = _u(rng, (0.04, 0.15))
        if rng.random() < config.local_warp_prob:
            meta["local_warp_amplitude"] = _u(rng, (0.5, 3.0))
        if rng.random() < config.chroma_prob:
            if rng.random() < 0.5:
                meta["chroma_blur"] = _u(rng, (0.5, 1.5))
            else:
                meta["chroma_shift_xy"] = [_u(rng, (-0.35, 0.35)), _u(rng, (-0.35, 0.35))]

        # Three fitted val anchors (after equivalent blur) are approximately
        # 0.91*x+8/255, 0.97*x+1/255 and 1.15*x-20/255.  Sample correlated
        # RGB parameters around those archetypes instead of independent colour
        # gains, which would invent unsupported random casts.
        tone = str(rng.choice(
            ("identity", "flat", "neutral", "contrast"),
            p=(0.35, 0.20, 0.20, 0.25),
        ))
        meta["tone_mode"] = tone
        if tone != "identity":
            ranges = {
                "flat": ((0.88, 0.96), (4.0, 12.0)),
                "neutral": ((0.95, 1.05), (-4.0, 4.0)),
                "contrast": ((1.10, 1.22), (-26.0, -10.0)),
            }
            gain_bounds, bias_bounds = ranges[tone]
            gain = _u(rng, gain_bounds)
            bias = _u(rng, bias_bounds) / 255.0
            meta["tone_gain"] = [float(gain + rng.uniform(-0.012, 0.012)) for _ in range(3)]
            meta["tone_bias"] = [float(bias + rng.uniform(-2.0, 2.0) / 255.0) for _ in range(3)]

    meta.update(
        jpeg_quality=95,
        jpeg_qtable_profile="ijg_q95_exact",
        jpeg_backend="diffjpeg_exact_qtable",
    )


def _sample_night(
    rng: np.random.Generator,
    meta: Dict,
    severity: str,
    config: DegradationConfig,
) -> None:
    _sample_spatial(rng, meta, severity, _NIGHT_RANGES, config.night_operator_probs, config)
    if rng.random() < 0.20:
        meta["subpixel_shift_xy"] = [_u(rng, (-0.4, 0.4)), _u(rng, (-0.4, 0.4))]
    if rng.random() < 0.75:
        meta["motion_length"] = _u(rng, (0.5, 3.0))
        meta["motion_theta"] = _u(rng, (0.0, 2 * math.pi))

    # Noise is deliberately inserted before strong denoising.  ``shot_noise`` is
    # the variance coefficient in var(x)=read^2+shot*x, all in [0,1] units.
    shot_at_white = _u(rng, (2.0, 8.0)) / 255.0
    meta["shot_noise"] = float(shot_at_white * shot_at_white)
    meta["read_noise"] = _u(rng, (0.5, 3.0)) / 255.0
    meta["denoise_strength"] = _u(rng, (0.45, 0.85))
    if rng.random() < config.night_fusion_prob:
        length, theta = _u(rng, (0.5, 2.0)), _u(rng, (0.0, 2 * math.pi))
        meta["fusion_shift_xy"] = [length * math.cos(theta), length * math.sin(theta)]
        meta["fusion_opacity"] = _u(rng, (0.05, 0.20))
    if rng.random() < 0.20:
        meta["local_warp_amplitude"] = _u(rng, (0.5, 3.0))
    if rng.random() < 0.65:
        meta["bloom_strength"] = _u(rng, (0.01, 0.12))

    gain = _u(rng, (0.98, 1.06))
    meta["tone_mode"] = "night_local"
    meta["tone_gain"] = [float(gain + rng.uniform(-0.025, 0.025)) for _ in range(3)]
    meta["tone_bias"] = [float(rng.uniform(-12.0, 0.0) / 255.0) for _ in range(3)]
    meta["tone_gamma"] = _u(rng, (0.90, 1.12))
    meta["highlight_compression"] = _u(rng, (0.02, 0.20))
    meta["saturation"] = _u(rng, (0.90, 1.10))
    if rng.random() < 0.30:
        meta["chroma_blur"] = _u(rng, (0.5, 1.5))
    meta.update(
        jpeg_quality=None,
        jpeg_qtable_profile="huawei_mpo_2026",
        jpeg_backend="diffjpeg_exact_qtable",
    )


class DegradationSampler:
    """Stateful seed allocator with per-sample replayable child streams."""

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        config: DegradationConfig = DEFAULT_CONFIG,
        rank: Optional[int] = None,
    ) -> None:
        if rank is None:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if seed is None:
            seed = int(torch.initial_seed())
        seq = np.random.SeedSequence([int(seed), int(rank)])
        self.rng = np.random.default_rng(seq)
        self.config = config

    def sample(
        self,
        *,
        profile: str = "auto",
        profile_hint: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
    ) -> Dict:
        return sample_degradation(
            self.rng,
            profile=profile,
            profile_hint=profile_hint,
            config=self.config,
            jpeg_quality=jpeg_quality,
        )

    def sample_batch(
        self,
        count: int,
        *,
        profile: str = "auto",
        profile_hints: Optional[Sequence[Optional[str]]] = None,
        jpeg_quality: Optional[int] = None,
    ) -> List[Dict]:
        hints = list(profile_hints) if profile_hints is not None else [None] * count
        if len(hints) != count:
            raise ValueError(f"profile_hints length {len(hints)} != batch size {count}")
        return [self.sample(profile=profile, profile_hint=h, jpeg_quality=jpeg_quality) for h in hints]


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed) & ((1 << 63) - 1))
    return gen


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    radius = max(1, min(12, int(math.ceil(3.0 * sigma))))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    c = x.shape[1]
    xp = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    xp = F.conv2d(xp, kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    xp = F.pad(xp, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(xp, kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _translate(x: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    _, _, h, w = x.shape
    theta = x.new_tensor([[[1.0, 0.0, -2.0 * dx / max(w, 1)],
                           [0.0, 1.0, -2.0 * dy / max(h, 1)]]])
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _lanczos_resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # PyTorch has no native Lanczos tensor op.  Non-antialiased bicubic preserves
    # the ringing/overshoot family needed by this rare branch better than the
    # antialiased approximation; metadata keeps the requested family explicit.
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=False)


class TensorDegrader:
    """Apply degradation metadata to RGB BCHW tensors in [0, 1]."""

    def __init__(self) -> None:
        self._jpegers: Dict[Tuple[str, str], torch.nn.Module] = {}

    @staticmethod
    def _lowpass_response(x: torch.Tensor, meta: Mapping, sigma_scale: float = 1.0) -> torch.Tensor:
        _, _, h, w = x.shape
        sx = float(meta["sigma_x"]) * sigma_scale
        sy = float(meta["sigma_y"]) * sigma_scale
        theta = float(meta["theta"])
        fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype).view(h, 1)
        fx = torch.fft.fftfreq(w, device=x.device, dtype=x.dtype).view(1, w)
        c, s = math.cos(theta), math.sin(theta)
        u = fx * c + fy * s
        v = -fx * s + fy * c
        if meta["kernel_family"] == "gaussian":
            resp = torch.exp(-2.0 * math.pi * math.pi * ((sx * u) ** 2 + (sy * v) ** 2))
        else:
            f0x = 0.1325 / max(sx, 1e-6)
            f0y = 0.1325 / max(sy, 1e-6)
            rr = (u / f0x) ** 2 + (v / f0y) ** 2
            resp = (1.0 + rr).pow(-float(meta["pole"]))
        return torch.fft.ifft2(torch.fft.fft2(x) * resp.view(1, 1, h, w)).real

    def _spatial_lowpass(self, x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        variation = float(meta["spatial_variation"])
        if variation <= 0:
            return self._lowpass_response(x, meta)
        lo = self._lowpass_response(x, meta, 1.0 - variation)
        hi = self._lowpass_response(x, meta, 1.0 + variation)
        gen = _torch_generator(x.device, int(meta["sample_seed"]) ^ 0x51A7)
        mask = torch.rand((1, 1, 8, 8), generator=gen, device=x.device, dtype=x.dtype)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        return lo * (1.0 - mask) + hi * mask

    @staticmethod
    def _resample(x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        h, w = x.shape[-2:]
        scale = float(meta["resize_scale"])
        small_size = (max(8, int(round(h / scale))), max(8, int(round(w / scale))))
        down = str(meta["down_mode"])
        up = str(meta["up_mode"])
        kwargs = {} if down == "area" else {
            "align_corners": False,
            "antialias": bool(meta.get("down_antialias", True)),
        }
        small = F.interpolate(x, size=small_size, mode=down, **kwargs)
        if up == "lanczos":
            return _lanczos_resize(small, (h, w))
        return F.interpolate(small, size=(h, w), mode=up, align_corners=False, antialias=False)

    @staticmethod
    def _edge_aware(x: torch.Tensor, strength: float) -> torch.Tensor:
        if strength <= 0:
            return x
        low = _gaussian_blur(x, 1.0 + 1.5 * strength)
        detail = (x - low).abs().mean(dim=1, keepdim=True)
        structure = (detail / 0.04).clamp(0, 1)
        blend = strength * (1.0 - structure)
        return x * (1.0 - blend) + low * blend

    @staticmethod
    def _motion(x: torch.Tensor, length: float, theta: float) -> torch.Tensor:
        if length <= 0:
            return x
        samples = max(3, int(math.ceil(length * 2)) | 1)
        offsets = torch.linspace(-0.5, 0.5, samples).tolist()
        return torch.stack([
            _translate(x, float(t * length * math.cos(theta)), float(t * length * math.sin(theta)))
            for t in offsets
        ]).mean(0)

    @staticmethod
    def _local_fusion(x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        shift = meta["fusion_shift_xy"]
        opacity = float(meta["fusion_opacity"])
        if shift is None or opacity <= 0:
            return x
        shifted = _translate(x, float(shift[0]), float(shift[1]))
        gen = _torch_generator(x.device, int(meta["sample_seed"]) ^ 0xF0510)
        mask = torch.rand((1, 1, 6, 6), generator=gen, device=x.device, dtype=x.dtype)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        alpha = opacity * mask
        return x * (1.0 - alpha) + shifted * alpha

    @staticmethod
    def _local_warp(x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        amplitude = float(meta.get("local_warp_amplitude", 0.0))
        if amplitude <= 0:
            return x
        _, _, height, width = x.shape
        gen = _torch_generator(x.device, int(meta["sample_seed"]) ^ 0x10CA1)
        flow = torch.rand((1, 2, 5, 5), generator=gen, device=x.device, dtype=x.dtype) * 2.0 - 1.0
        flow -= flow.mean(dim=(-2, -1), keepdim=True)
        peak = flow.abs().amax().clamp_min(1e-6)
        flow = F.interpolate(flow / peak, size=(height, width), mode="bicubic", align_corners=False)
        # Bicubic interpolation can overshoot the control-point range.  Renormalise
        # after interpolation so metadata's amplitude remains a true hard bound.
        flow = flow / flow.abs().amax().clamp_min(1.0)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        grid[..., 0] += flow[:, 0] * (2.0 * amplitude / max(width - 1, 1))
        grid[..., 1] += flow[:, 1] * (2.0 * amplitude / max(height - 1, 1))
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @staticmethod
    def _chroma(x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        blur = float(meta["chroma_blur"])
        shift = meta["chroma_shift_xy"]
        saturation = float(meta["saturation"])
        if blur <= 0 and shift is None and abs(saturation - 1.0) < 1e-8:
            return x
        y = (x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114)
        chroma = x - y
        if blur > 0:
            chroma = _gaussian_blur(chroma, blur)
        if shift is not None:
            chroma = _translate(chroma, float(shift[0]), float(shift[1]))
        return (y + chroma * saturation).clamp(0, 1)

    @staticmethod
    def _tone(x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        gain = x.new_tensor(meta["tone_gain"]).view(1, 3, 1, 1)
        bias = x.new_tensor(meta["tone_bias"]).view(1, 3, 1, 1)
        out = (x * gain + bias).clamp(0, 1)
        gamma = float(meta["tone_gamma"])
        if abs(gamma - 1.0) > 1e-8:
            out = out.clamp_min(1e-6).pow(gamma)
        compression = float(meta["highlight_compression"])
        if compression > 0:
            high = (out - 0.65).clamp_min(0.0)
            out = out - compression * high * high / 0.35
        return out.clamp(0, 1)

    def _apply_one(self, x: torch.Tensor, meta: Mapping) -> torch.Tensor:
        # near_native is intentionally strict: only the mandatory final JPEG is
        # applied.  It is the "already good, do little" training guardrail.
        if meta["operator"] == "near_native":
            return x

        out = x
        shift = meta["subpixel_shift_xy"]
        if shift is not None:
            out = _translate(out, float(shift[0]), float(shift[1]))
        operator = meta["operator"]
        if operator in ("heavy_tail", "aniso_gaussian", "combined"):
            out = self._spatial_lowpass(out, meta)
        if operator in ("resample", "combined"):
            out = self._resample(out, meta)
        if float(meta["motion_length"]) > 0:
            out = self._motion(out, float(meta["motion_length"]), float(meta["motion_theta"]))

        if meta["profile"] == "night_hdr":
            gen = _torch_generator(out.device, int(meta["sample_seed"]) ^ 0xA11CE)
            variance = float(meta["read_noise"]) ** 2 + float(meta["shot_noise"]) * out.clamp(0, 1)
            noise = torch.randn(out.shape, generator=gen, device=out.device, dtype=out.dtype)
            out = (out + noise * variance.sqrt()).clamp(0, 1)
            out = self._edge_aware(out, float(meta["denoise_strength"]))
        else:
            out = self._edge_aware(out, float(meta["edge_aware_strength"]))

        out = self._local_warp(out, meta)
        out = self._local_fusion(out, meta)
        bloom = float(meta["bloom_strength"])
        if bloom > 0:
            highlights = (out - 0.72).clamp_min(0.0)
            out = (out + bloom * _gaussian_blur(highlights, 2.0)).clamp(0, 1)
        amount = float(meta["unsharp_amount"])
        if amount > 0:
            out = (out + amount * (out - _gaussian_blur(out, 1.0))).clamp(0, 1)
        out = self._chroma(out, meta)
        return self._tone(out, meta)

    def _make_jpeger(self, device: torch.device, profile: str):
        from HYPIR.dataset.diffjpeg import DiffJPEG

        key = (str(device), profile)
        if key in self._jpegers:
            return self._jpegers[key]
        module = DiffJPEG(differentiable=False).to(device)
        if profile == "ijg_q95_exact":
            y, c = IJG_Q95_Y, IJG_Q95_C
        elif profile.startswith("ijg_q") and profile.endswith("_exact"):
            quality = int(profile[len("ijg_q"):-len("_exact")])
            if not 1 <= quality <= 100:
                raise ValueError(f"invalid IJG quality profile: {profile}")
            scale = 5000 // quality if quality < 50 else 200 - 2 * quality
            y = tuple(min(255, max(1, (value * scale + 50) // 100)) for value in IJG_BASE_Y)
            c = tuple(min(255, max(1, (value * scale + 50) // 100)) for value in IJG_BASE_C)
        elif profile == "huawei_mpo_2026":
            y, c = HUAWEI_MPO_Y, HUAWEI_MPO_C
        else:
            self._jpegers[key] = module
            return module
        # diffjpeg stores its base tables transposed relative to Pillow/libjpeg's
        # natural row-major order.  Calling with quality=50 gives factor=1.
        yt = torch.tensor(y, dtype=torch.float32, device=device).view(8, 8).T
        ct = torch.tensor(c, dtype=torch.float32, device=device).view(8, 8).T
        # DiffJPEG's upstream implementation points every instance at module-level
        # Parameter objects.  Mutating ``.data`` would silently make the IJG and
        # Huawei encoders overwrite one another.  Give every compressor and
        # decompressor its own immutable table instead.
        module.compress.y_quantize.y_table = torch.nn.Parameter(yt.clone(), requires_grad=False)
        module.decompress.y_dequantize.y_table = torch.nn.Parameter(yt.clone(), requires_grad=False)
        module.compress.c_quantize.c_table = torch.nn.Parameter(ct.clone(), requires_grad=False)
        module.decompress.c_dequantize.c_table = torch.nn.Parameter(ct.clone(), requires_grad=False)
        self._jpegers[key] = module
        return module

    def _jpeg(self, batch: torch.Tensor, metas: Sequence[Mapping]) -> torch.Tensor:
        result = torch.empty_like(batch)
        groups: Dict[Tuple[str, Optional[int]], List[int]] = {}
        for i, meta in enumerate(metas):
            profile = str(meta["jpeg_qtable_profile"])
            groups.setdefault((profile, meta["jpeg_quality"]), []).append(i)
        for (profile, quality), indices in groups.items():
            idx = torch.tensor(indices, device=batch.device, dtype=torch.long)
            subset = batch.index_select(0, idx).clamp(0, 1)
            module_profile = f"ijg_q{int(quality)}_exact" if profile == "ijg_dynamic" else profile
            module = self._make_jpeger(batch.device, module_profile)
            if module_profile.endswith("_exact") or module_profile == "huawei_mpo_2026":
                q = subset.new_full((len(indices),), 50.0)
            else:
                q = subset.new_full((len(indices),), float(quality))
            encoded = module(subset, quality=q.clone())
            result.index_copy_(0, idx, encoded)
        return result

    @torch.no_grad()
    def apply(
        self,
        hq: torch.Tensor,
        metadata: Sequence[Mapping],
        *,
        apply_jpeg: bool = True,
        quantize: bool = True,
    ) -> torch.Tensor:
        if hq.ndim != 4 or hq.shape[1] != 3:
            raise ValueError(f"expected RGB BCHW tensor, got shape {tuple(hq.shape)}")
        if not hq.is_floating_point():
            raise ValueError(f"expected floating tensor in [0,1], got {hq.dtype}")
        if len(metadata) != hq.shape[0]:
            raise ValueError(f"metadata length {len(metadata)} != batch size {hq.shape[0]}")
        if not torch.isfinite(hq).all() or hq.min() < 0 or hq.max() > 1:
            raise ValueError("hq must be finite and contained in [0,1]")

        outs = []
        for i, meta in enumerate(metadata):
            x = hq[i:i + 1].clone()
            pre_q = meta.get("pre_jpeg_quality")
            if pre_q is not None and meta["operator"] != "near_native":
                pre_meta = dict(meta)
                pre_meta.update(
                    jpeg_quality=int(pre_q),
                    jpeg_qtable_profile="ijg_dynamic",
                    jpeg_backend="diffjpeg_exact_qtable",
                )
                x = self._jpeg(x, [pre_meta])
            outs.append(self._apply_one(x, meta))
        out = torch.cat(outs, dim=0).clamp(0, 1)
        if apply_jpeg:
            out = self._jpeg(out, metadata).clamp(0, 1)
            if quantize:
                out = (out * 255.0).round().clamp(0, 255) / 255.0
        return out


@torch.no_grad()
def degrade_tensor(
    hq: torch.Tensor,
    *,
    rng: Optional[np.random.Generator] = None,
    profile: str = "auto",
    profile_hints: Optional[Sequence[Optional[str]]] = None,
    config: DegradationConfig = DEFAULT_CONFIG,
    metadata: Optional[Sequence[Mapping]] = None,
    sample_seeds: Optional[Sequence[int]] = None,
    jpeg_quality: Optional[int] = None,
    return_metadata: bool = False,
    degrader: Optional[TensorDegrader] = None,
):
    """Degrade an already-cropped RGB BCHW tensor.

    Passing ``metadata`` replays an existing batch.  Otherwise metadata is
    sampled from ``rng`` (or a fresh generator seeded from ``torch.initial_seed``).
    """

    if metadata is None:
        rng = rng or np.random.default_rng()
        hints = list(profile_hints) if profile_hints is not None else [None] * hq.shape[0]
        seeds = list(sample_seeds) if sample_seeds is not None else [None] * hq.shape[0]
        if len(hints) != hq.shape[0] or len(seeds) != hq.shape[0]:
            raise ValueError("profile_hints/sample_seeds must match batch size")
        metadata = [sample_degradation(
            rng,
            profile=profile,
            profile_hint=hints[i],
            config=config,
            sample_seed=seeds[i],
            jpeg_quality=jpeg_quality,
        ) for i in range(hq.shape[0])]
    else:
        metadata = [dict(m) for m in metadata]
    out = (degrader or TensorDegrader()).apply(hq, metadata)
    return (out, metadata) if return_metadata else out


__all__ = [
    "AFFINE_A", "AFFINE_B", "AFFINE_B_JITTER", "AFFINE_CHROMA", "AFFINE_PIVOT",
    "ANISO_RANGE", "DEFAULT_CONFIG", "DegradationConfig", "DegradationSampler",
    "HUAWEI_MPO_C", "HUAWEI_MPO_Y", "IJG_BASE_C", "IJG_BASE_Y", "IJG_Q95_C", "IJG_Q95_Y",
    "IMPLEMENTATION_VERSION", "JPEG_QUALITY", "JPEG_RANGE", "NOISE_SIGMA",
    "OPERATOR_PROBS", "P_AFFINE", "P_CLEAN", "P_RESAMPLE", "POLE_RANGE",
    "PROFILE_PROBS", "SCALE_RANGE", "SCHEMA_VERSION", "SEVERITY_PROBS",
    "SIGMA_RANGE", "SPATIAL_VAR", "TensorDegrader", "degrade_tensor",
    "sample_degradation",
]
