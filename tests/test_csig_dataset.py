import io
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from HYPIR.dataset.csig import CSIGBatchTransform, CSIGDataset, _magic_kind, csig_collate
from HYPIR.utils.common import instantiate_from_config


def _encoded(fmt: str, color, size=(80, 72)) -> bytes:
    array = np.empty((size[1], size[0], 3), dtype=np.uint8)
    array[:] = color
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=fmt, quality=95, subsampling=2)
    return stream.getvalue()


def test_magic_dispatch_ignores_extension(tmp_path: Path):
    jpeg_as_png = tmp_path / "jpeg.png"
    png_as_jpg = tmp_path / "png.jpg"
    jpeg_as_png.write_bytes(_encoded("JPEG", (255, 0, 0)))
    png_as_jpg.write_bytes(_encoded("PNG", (0, 255, 0)))
    assert _magic_kind(jpeg_as_png.read_bytes()) == "jpeg"
    assert _magic_kind(png_as_jpg.read_bytes()) == "png"

    listing = tmp_path / "list.txt"
    listing.write_text(f"{jpeg_as_png}\tordinary\n{png_as_jpg}\tnight_hdr\n")
    dataset = CSIGDataset(str(listing), out_size=64)
    batch = csig_collate([dataset[0], dataset[1]])
    assert batch["profile_hint"] == ["ordinary", "night_hdr"]
    transform = CSIGBatchTransform(
        out_size=64, use_hflip=False, return_metadata=True, seed=7,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    result = transform(batch)
    assert set(result) == {"GT", "LQ", "txt", "degradation_meta"}
    assert result["GT"].shape == result["LQ"].shape == (2, 3, 64, 64)
    assert [m["profile"] for m in result["degradation_meta"]] == ["ordinary", "night_hdr"]
    assert result["GT"][0, 0].mean() > 0.95
    assert result["GT"][0, 1].mean() < 0.05
    assert result["GT"][1, 1].mean() > 0.95


def test_default_batch_contract_has_only_trainer_keys(tmp_path: Path):
    path = tmp_path / "x.jpg"
    path.write_bytes(_encoded("JPEG", (10, 20, 30)))
    listing = tmp_path / "list.txt"
    listing.write_text(str(path) + "\n")
    item = CSIGDataset(str(listing), out_size=64)[0]
    transform = CSIGBatchTransform(out_size=64, seed=1,
                                   device="cuda" if torch.cuda.is_available() else "cpu")
    assert set(transform(csig_collate([item]))) == {"GT", "LQ", "txt"}


def test_small_native_image_is_rejected(tmp_path: Path):
    path = tmp_path / "tiny.jpg"
    path.write_bytes(_encoded("JPEG", (0, 0, 0), size=(32, 32)))
    listing = tmp_path / "list.txt"
    listing.write_text(str(path) + "\n")
    with pytest.raises(RuntimeError, match="smaller than crop"):
        CSIGDataset(str(listing), out_size=64)[0]


def test_checked_in_training_config_instantiates(tmp_path: Path):
    path = tmp_path / "x.jpg"
    path.write_bytes(_encoded("JPEG", (40, 80, 120)))
    listing = tmp_path / "list.txt"
    listing.write_text(str(path) + "\tordinary\n")
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "csig_train.yaml")
    train = config.data_config.train
    train.dataset.params.file_list = str(listing)
    train.dataset.params.out_size = 64
    train.batch_transform.params.out_size = 64
    train.batch_transform.params.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = instantiate_from_config(train.dataset)
    transform = instantiate_from_config(train.batch_transform)
    batch = train.collate_fn
    assert batch == "HYPIR.dataset.csig.csig_collate"
    result = transform(csig_collate([dataset[0]]))
    assert set(result) == {"GT", "LQ", "txt"}
    assert result["GT"].shape == (1, 3, 64, 64)
