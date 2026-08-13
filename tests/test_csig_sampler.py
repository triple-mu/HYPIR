import json

import numpy as np
import pytest
import torch

from HYPIR.dataset.csig_degradation import (
    DegradationConfig,
    DegradationSampler,
    TensorDegrader,
    sample_degradation,
)


def test_sampler_is_replayable_and_strict_json():
    a = sample_degradation(np.random.default_rng(123), profile="auto")
    b = sample_degradation(np.random.default_rng(123), profile="auto")
    assert a == b
    assert json.loads(json.dumps(a, allow_nan=False)) == a
    assert isinstance(a["sample_seed"], int)
    assert a["profile"] in {"ordinary", "night_hdr"}


def test_hint_and_forced_profile_precedence():
    rng = np.random.default_rng(1)
    hinted = sample_degradation(rng, profile="ordinary", profile_hint="night_hdr")
    assert hinted["profile"] == "night_hdr"
    assert hinted["profile_source"] == "hint"
    forced = sample_degradation(rng, profile="ordinary")
    assert forced["profile"] == "ordinary"
    assert forced["profile_source"] == "forced"
    with pytest.raises(ValueError, match="unknown degradation profile"):
        sample_degradation(rng, profile="wrong")


def test_probability_and_range_coverage():
    rng = np.random.default_rng(20260813)
    records = [sample_degradation(rng) for _ in range(20000)]
    p_night = sum(m["profile"] == "night_hdr" for m in records) / len(records)
    assert abs(p_night - 0.10) < 0.015

    ordinary = [m for m in records if m["profile"] == "ordinary"]
    severity = {s: sum(m["severity"] == s for m in ordinary) / len(ordinary)
                for s in ("weak", "medium", "strong")}
    assert abs(severity["weak"] - 0.20) < 0.02
    assert abs(severity["medium"] - 0.50) < 0.02
    assert abs(severity["strong"] - 0.30) < 0.02

    non_native = [m for m in ordinary if m["operator"] != "near_native"]
    expected = {"heavy_tail": 0.45, "aniso_gaussian": 0.20, "resample": 0.25, "combined": 0.10}
    for operator, probability in expected.items():
        observed = sum(m["operator"] == operator for m in non_native) / len(non_native)
        assert abs(observed - probability) < 0.025
    for meta in records:
        if meta["sigma_x"] is not None:
            assert meta["sigma_x"] > 0 and meta["sigma_y"] > 0
            assert 1.0 <= meta["sigma_x"] / meta["sigma_y"] <= 1.5 + 1e-9
        if meta["resize_scale"] is not None:
            assert 1.0 <= meta["resize_scale"] <= 7.0
        assert 0.0 <= meta["local_warp_amplitude"] <= 3.0


def test_stateful_sampler_does_not_touch_numpy_global_rng():
    np.random.seed(99)
    before = np.random.get_state()
    a = DegradationSampler(seed=17).sample_batch(4)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert [m["sample_seed"] for m in a] == [m["sample_seed"] for m in DegradationSampler(seed=17).sample_batch(4)]
    assert len({m["sample_seed"] for m in a}) == 4


def test_config_validation_and_explicit_jpeg_quality():
    with pytest.raises(ValueError):
        DegradationConfig(profile_probs={"ordinary": 0, "night_hdr": 0})
    with pytest.raises(ValueError):
        DegradationConfig(profile_probs={"ordinary": 1, "bad": 1})
    meta = sample_degradation(np.random.default_rng(2), profile="ordinary", jpeg_quality=91)
    assert meta["jpeg_quality"] == 91
    assert meta["jpeg_backend"] == "diffjpeg_exact_qtable"
    config = DegradationConfig()
    assert list(config.profile_probs) == ["ordinary", "night_hdr"]
    assert list(config.severity_probs) == ["weak", "medium", "strong"]


@pytest.mark.parametrize("size", [64, 65])
def test_tensor_degrader_shape_range_and_replay(size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(4, 3, size, size, device=device)
    sampler = DegradationSampler(seed=1234)
    metadata = sampler.sample_batch(4)
    degrader = TensorDegrader()
    a = degrader.apply(x, metadata)
    b = degrader.apply(x, metadata)
    assert torch.equal(a, b)
    assert a.shape == x.shape and a.dtype == x.dtype and a.device == x.device
    assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert torch.allclose((a * 255).round(), a * 255, atol=2e-5, rtol=0)


def test_near_native_is_jpeg_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 64, 64, device=device)
    rng = np.random.default_rng(0)
    meta = None
    while meta is None or meta["operator"] != "near_native":
        candidate = sample_degradation(rng, profile="ordinary")
        if candidate["operator"] == "near_native":
            meta = candidate
    degrader = TensorDegrader()
    full = degrader.apply(x, [meta])
    jpeg_reference = degrader._jpeg(x, [meta]).clamp(0, 1)
    jpeg_reference = (jpeg_reference * 255).round().clamp(0, 255) / 255
    assert torch.equal(full, jpeg_reference)


def test_local_warp_is_bounded_finite_and_replayable():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 65, 67, device=device)
    meta = sample_degradation(np.random.default_rng(77), profile="ordinary")
    meta["local_warp_amplitude"] = 3.0
    degrader = TensorDegrader()
    a = degrader.apply(x, [meta])
    b = degrader.apply(x, [meta])
    assert torch.equal(a, b)
    assert torch.isfinite(a).all() and 0 <= a.min() and a.max() <= 1


def test_exact_jpeg_profiles_do_not_share_quantisation_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degrader = TensorDegrader()
    ordinary = degrader._make_jpeger(device, "ijg_q95_exact")
    ordinary_before = ordinary.compress.y_quantize.y_table.detach().clone()
    huawei = degrader._make_jpeger(device, "huawei_mpo_2026")
    assert torch.equal(ordinary.compress.y_quantize.y_table, ordinary_before)
    assert not torch.equal(ordinary.compress.y_quantize.y_table, huawei.compress.y_quantize.y_table)
    meta = sample_degradation(np.random.default_rng(5), profile="ordinary", jpeg_quality=100)
    out = degrader.apply(torch.rand(1, 3, 64, 64, device=device), [meta])
    assert torch.isfinite(out).all()
