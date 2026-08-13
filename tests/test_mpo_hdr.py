from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import struct

import pytest

from csig.mpo_hdr import (
    ICC_IDENTIFIER,
    ISO_21496_IDENTIFIER,
    MpoMetadataError,
    extract_frame,
    inspect_bytes,
    inspect_file,
)


def _segment(marker: int, payload: bytes) -> bytes:
    return b"\xFF" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def _sof(width: int, height: int) -> bytes:
    payload = struct.pack(">BHHB", 8, height, width, 3)
    payload += bytes([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1])
    return _segment(0xC0, payload)


def _sos() -> bytes:
    payload = bytes([3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0])
    return _segment(0xDA, payload) + b"\x11\x22\xFF\x00\x33"


def _jpeg(width: int, height: int, app_segments: list[bytes]) -> bytes:
    return b"\xFF\xD8" + b"".join(app_segments) + _sof(width, height) + _sos() + b"\xFF\xD9"


def _align4(blob: bytes) -> bytes:
    return blob + b"\x00" * (-len(blob) % 4)


def _icc_profile(description: str = "sRGB EOTF with DCI-P3 Color Gamut") -> bytes:
    desc_text = description.encode("ascii") + b"\x00"
    desc = _align4(b"desc" + b"\x00" * 4 + struct.pack(">I", len(desc_text)) + desc_text)
    copyright_text = b"Synthetic test profile\x00"
    copyright_tag = _align4(b"text" + b"\x00" * 4 + copyright_text)
    table_end = 132 + 2 * 12
    desc_offset = table_end
    copyright_offset = desc_offset + len(desc)
    size = copyright_offset + len(copyright_tag)

    header = bytearray(128)
    struct.pack_into(">I", header, 0, size)
    struct.pack_into(">I", header, 8, 0x04400000)
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    header[36:40] = b"acsp"
    table = struct.pack(">I", 2)
    table += struct.pack(">4sII", b"desc", desc_offset, len(desc))
    table += struct.pack(">4sII", b"cprt", copyright_offset, len(copyright_tag))
    return bytes(header) + table + desc + copyright_tag


def _icc_segments(profile: bytes, *, chunks: int = 1, reverse: bool = False) -> list[bytes]:
    cuts = [len(profile) * index // chunks for index in range(chunks + 1)]
    payloads = [
        ICC_IDENTIFIER + bytes([index + 1, chunks]) + profile[cuts[index] : cuts[index + 1]]
        for index in range(chunks)
    ]
    if reverse:
        payloads.reverse()
    return [_segment(0xE2, payload) for payload in payloads]


def _rational(numerator: int, denominator: int, *, signed: bool = False) -> bytes:
    return struct.pack(">iI" if signed else ">II", numerator, denominator)


def _iso_body(*, denominator: int = 1_000_000, multichannel: bool = False) -> bytes:
    flags = 0x40 | (0x80 if multichannel else 0)
    body = struct.pack(">HHB", 0, 0, flags)
    body += _rational(0, denominator)
    body += _rational(2_148_445, denominator)
    channel = _rational(0, denominator, signed=True)
    channel += _rational(2_169_925, denominator, signed=True)
    channel += _rational(denominator, denominator)
    channel += _rational(0, denominator, signed=True)
    channel += _rational(0, denominator, signed=True)
    return body + channel * (3 if multichannel else 1)


def _mpf_payload(
    primary_size: int,
    secondary_size: int,
    secondary_offset: int,
    *,
    endian: str,
) -> bytes:
    byte_order = b"MM" if endian == ">" else b"II"
    ifd_offset = 8
    entry_data_offset = 8 + 2 + 3 * 12 + 4
    tiff = bytearray(byte_order + struct.pack(endian + "HI", 42, ifd_offset))
    tiff += struct.pack(endian + "H", 3)
    tiff += struct.pack(endian + "HHI4s", 0xB000, 7, 4, b"0100")
    tiff += struct.pack(endian + "HHII", 0xB001, 4, 1, 2)
    tiff += struct.pack(endian + "HHII", 0xB002, 7, 32, entry_data_offset)
    tiff += struct.pack(endian + "I", 0)
    tiff += struct.pack(endian + "IIIHH", 0x00030000, primary_size, 0, 0, 0)
    tiff += struct.pack(
        endian + "IIIHH", 0x00050000, secondary_size, secondary_offset, 0, 0
    )
    return b"MPF\x00" + bytes(tiff)


def _synthetic_mpo(
    *,
    endian: str = ">",
    tail: bytes = b"private\xFF\xD8\xFF\xE2not-a-frame",
    icc_chunks: int = 2,
    reverse_icc: bool = True,
    iso_denominator: int = 1_000_000,
) -> tuple[bytes, bytes]:
    profile = _icc_profile()
    secondary_apps = [
        _segment(0xE2, ISO_21496_IDENTIFIER + _iso_body(denominator=iso_denominator)),
        *_icc_segments(profile),
    ]
    secondary = _jpeg(2048, 1536, secondary_apps)
    descriptor = _segment(0xE2, ISO_21496_IDENTIFIER + b"\x00" * 4)
    primary_icc = _icc_segments(profile, chunks=icc_chunks, reverse=reverse_icc)

    placeholder_mpf = _segment(0xE2, _mpf_payload(0, len(secondary), 0, endian=endian))
    primary = _jpeg(4096, 3072, [descriptor, placeholder_mpf, *primary_icc])
    mpf_marker_offset = primary.find(b"MPF\x00") - 4
    tiff_base = mpf_marker_offset + 8
    relative_secondary_offset = len(primary) - tiff_base
    real_mpf = _segment(
        0xE2,
        _mpf_payload(
            len(primary),
            len(secondary),
            relative_secondary_offset,
            endian=endian,
        ),
    )
    rebuilt = _jpeg(4096, 3072, [descriptor, real_mpf, *primary_icc])
    assert len(rebuilt) == len(primary)
    return rebuilt + secondary + tail, profile


def test_synthetic_mpo_metadata_and_opaque_tail() -> None:
    data, profile = _synthetic_mpo()
    info = inspect_bytes(data, strict=True)

    assert info.is_jpeg and info.is_mpo
    assert info.mpf_version == "0100"
    assert info.number_of_images == 2
    assert len(info.frames) == 2
    assert [(frame.width, frame.height) for frame in info.frames] == [
        (4096, 3072),
        (2048, 1536),
    ]
    assert [entry.attributes for entry in info.mp_entries] == [0x00030000, 0x00050000]
    assert info.frames[1].offset == info.frames[0].size
    assert info.unindexed_tail_size == len(b"private\xFF\xD8\xFF\xE2not-a-frame")
    assert len(info.frames) == 2  # marker-shaped private bytes were not scanned

    assert len(info.frames[0].iso21496) == 1
    assert info.frames[0].iso21496[0].descriptor_only
    assert info.gain_map_frame_index == 1
    assert info.gain_map is not None
    assert info.gain_map.flags == 0x40
    assert info.gain_map.alternate_hdr_headroom.numerator == 2_148_445
    assert info.gain_map.channels[0].gain_map_max.value == pytest.approx(2.169925)

    assert info.frames[0].icc is not None
    assert info.frames[0].icc.chunk_count == 2
    assert info.frames[0].icc.sha256 == sha256(profile).hexdigest()
    assert info.frames[0].icc.description == "sRGB EOTF with DCI-P3 Color Gamut"
    assert info.frames[0].icc.copyright == "Synthetic test profile"
    assert info.frames[1].icc is not None
    assert info.frames[1].icc.sha256 == sha256(profile).hexdigest()

    assert bytes(extract_frame(data, info, 1)).startswith(b"\xFF\xD8")
    assert bytes(extract_frame(data, info, 1)).endswith(b"\xFF\xD9")
    assert info.as_dict()["gain_map"]["channels"][0]["gamma"]["value"] == 1.0


def test_little_endian_mpf() -> None:
    data, _ = _synthetic_mpo(endian="<")
    info = inspect_bytes(data, strict=True)
    assert info.is_mpo
    assert info.frames[1].offset == info.frames[0].size


def test_non_mpo_and_non_jpeg_are_graceful() -> None:
    jpeg = _jpeg(32, 24, [_segment(0xE2, ISO_21496_IDENTIFIER + b"\x00" * 4)])
    info = inspect_bytes(jpeg, strict=True)
    assert info.is_jpeg and not info.is_mpo
    assert len(info.frames) == 1
    assert info.frames[0].iso21496[0].descriptor_only
    assert info.gain_map is None

    png = inspect_bytes(b"\x89PNG\r\n\x1a\n", strict=True)
    assert not png.is_jpeg and not png.is_mpo and not png.frames


def test_bad_segment_length_strict_and_tolerant() -> None:
    malformed = b"\xFF\xD8\xFF\xE2\x00\x01\xFF\xD9"
    with pytest.raises(MpoMetadataError, match="invalid segment length"):
        inspect_bytes(malformed, strict=True)
    tolerant = inspect_bytes(malformed)
    assert tolerant.is_jpeg and not tolerant.is_mpo
    assert tolerant.warnings


def test_bad_mpf_byte_order_is_rejected() -> None:
    data, _ = _synthetic_mpo()
    marker = data.find(b"MPF\x00")
    damaged = data[: marker + 4] + b"ZZ" + data[marker + 6 :]
    with pytest.raises(MpoMetadataError, match="invalid TIFF byte order"):
        inspect_bytes(damaged, strict=True)
    tolerant = inspect_bytes(damaged)
    assert not tolerant.is_mpo
    assert any("byte order" in warning for warning in tolerant.warnings)


def test_zero_iso_rational_denominator_is_rejected() -> None:
    data, _ = _synthetic_mpo(iso_denominator=0)
    with pytest.raises(MpoMetadataError, match="zero rational denominator"):
        inspect_bytes(data, strict=True)


def test_missing_icc_chunk_is_rejected() -> None:
    data, _ = _synthetic_mpo(icc_chunks=2, reverse_icc=False)
    first = data.find(ICC_IDENTIFIER)
    second = data.find(ICC_IDENTIFIER, first + 1)
    assert second > first
    # Keep MPF sizes and JPEG framing unchanged, but make chunk 2 unrecognisable.
    damaged = data[:second] + b"X" + data[second + 1 :]
    with pytest.raises(MpoMetadataError, match="missing ICC chunks"):
        inspect_bytes(damaged, strict=True)


def _real_data_root() -> Path | None:
    raw = os.environ.get("CSIG_DATA_ROOT")
    if not raw:
        return None
    root = Path(raw)
    return root / "测试集" if (root / "测试集").is_dir() else root


@pytest.mark.skipif(_real_data_root() is None, reason="CSIG_DATA_ROOT is not configured")
def test_real_csig_mpo_regression() -> None:
    test_root = _real_data_root()
    assert test_root is not None
    mpo_cases = {1, 2, 3, 4, 5, 38, 64, 65, 66}
    discovered = {
        case
        for case in range(1, 101)
        if inspect_file(test_root / f"case{case}.jpg", strict=True).is_mpo
    }
    assert discovered == mpo_cases

    primary_sizes = [
        1_023_147,
        1_393_001,
        1_036_411,
        1_742_400,
        1_335_950,
        3_086_401,
        1_844_491,
        2_271_582,
        1_723_358,
    ]
    gain_sizes = [
        126_036,
        135_927,
        103_658,
        169_886,
        125_000,
        222_096,
        219_107,
        150_046,
        179_760,
    ]
    tail_sizes = [
        6_022_866,
        6_058_086,
        6_011_862,
        6_011_862,
        6_031_082,
        5_827_210,
        6_003_916,
        6_003_916,
        6_003_916,
    ]

    for index, case in enumerate(sorted(mpo_cases)):
        info = inspect_file(test_root / f"case{case}.jpg", strict=True)
        assert info.mpf_version == "0100"
        assert info.number_of_images == 2
        assert len(info.frames) == 2
        assert [entry.attributes for entry in info.mp_entries] == [0x00030000, 0x00050000]
        assert info.frames[0].offset == 0
        assert info.frames[0].size == primary_sizes[index]
        assert info.frames[1].offset == primary_sizes[index]
        assert info.frames[1].size == gain_sizes[index]
        assert info.unindexed_tail_size == tail_sizes[index]
        assert info.frames[0].segments[-1].marker == 0xD9
        assert info.frames[1].segments[-1].marker == 0xD9

        primary_dimensions = (3072, 4096) if case == 64 else (4096, 3072)
        gain_dimensions = (1536, 2048) if case == 64 else (2048, 1536)
        assert (info.frames[0].width, info.frames[0].height) == primary_dimensions
        assert (info.frames[1].width, info.frames[1].height) == gain_dimensions

        assert len(info.frames[0].iso21496) == 1
        assert info.frames[0].iso21496[0].descriptor_only
        assert info.frames[0].iso21496[0].body_size == 4
        assert info.gain_map_frame_index == 1
        assert info.gain_map is not None
        assert info.gain_map.minimum_version == 0
        assert info.gain_map.writer_version == 0
        assert info.gain_map.flags == 0x40
        assert not info.gain_map.is_multichannel
        assert info.gain_map.use_base_colour_space
        assert info.gain_map.base_hdr_headroom.numerator == 0
        assert info.gain_map.base_hdr_headroom.denominator == 1_000_000
        assert info.gain_map.alternate_hdr_headroom.numerator == 2_148_445
        assert info.gain_map.alternate_hdr_headroom.denominator == 1_000_000
        channel = info.gain_map.channels[0]
        assert channel.gain_map_min.value == 0.0
        assert channel.gain_map_max.value == pytest.approx(2.169925)
        assert channel.gamma.value == 1.0
        assert channel.base_offset.value == 0.0
        assert channel.alternate_offset.value == 0.0

        assert info.frames[0].icc is not None
        assert info.frames[1].icc is not None
        assert info.frames[0].icc.description is not None
        assert info.frames[0].icc.description.endswith("sRGB EOTF with DCI-P3 Color Gamut")
        if case in {1, 2, 3, 4, 5, 38}:
            assert info.frames[0].icc.declared_size == 684
            assert info.frames[0].icc.sha256.startswith("9f76bdecbcf5")
            assert info.frames[1].icc.sha256.startswith("9f76bdecbcf5")
        else:
            assert info.frames[0].icc.declared_size == 660
            assert info.frames[0].icc.sha256.startswith("f429bcb7f1d7")
            assert info.frames[1].icc.declared_size == 684
            assert info.frames[1].icc.sha256.startswith("08afdeef67fe")

    # This file contains marker-shaped bytes in its private tail; MPF still says two frames.
    assert len(inspect_file(test_root / "case3.jpg", strict=True).frames) == 2
