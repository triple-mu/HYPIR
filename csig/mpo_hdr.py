"""Small, dependency-free JPEG MPO/HDR metadata parser.

The parser deliberately stops at the EOI of each MPF-indexed JPEG frame.  Bytes
outside the MPF entries are reported as an opaque tail and are never searched
for marker-shaped byte sequences.  This matters for phone-camera MPO files,
which may contain several megabytes of private data after the gain-map JPEG.

Only metadata needed by the CSIG data pipeline is decoded: MPF frame entries,
ISO/TS 21496-1 gain-map parameters, SOF dimensions, and embedded ICC profiles.
Pixel decoding and colour conversion remain optional concerns for Pillow/lcms.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
import argparse
import json
import struct
from typing import Any, Iterable, Optional, Union


ISO_21496_IDENTIFIER = b"urn:iso:std:iso:ts:21496:-1\x00"
ICC_IDENTIFIER = b"ICC_PROFILE\x00"
MPF_IDENTIFIER = b"MPF\x00"

_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}
_STANDALONE_MARKERS = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}
_TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
}


class MpoMetadataError(ValueError):
    """Raised for malformed or internally inconsistent metadata in strict mode."""


@dataclass(frozen=True)
class Rational:
    numerator: int
    denominator: int

    @property
    def value(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True)
class GainMapChannel:
    gain_map_min: Rational
    gain_map_max: Rational
    gamma: Rational
    base_offset: Rational
    alternate_offset: Rational


@dataclass(frozen=True)
class GainMapMetadata:
    minimum_version: int
    writer_version: int
    flags: int
    is_multichannel: bool
    use_base_colour_space: bool
    base_hdr_headroom: Rational
    alternate_hdr_headroom: Rational
    channels: tuple[GainMapChannel, ...]


@dataclass(frozen=True)
class Iso21496Record:
    offset: int
    body_size: int
    descriptor_only: bool
    metadata: Optional[GainMapMetadata]


@dataclass(frozen=True)
class IccSummary:
    declared_size: int
    version: int
    device_class: str
    colour_space: str
    pcs: str
    description: Optional[str]
    copyright: Optional[str]
    tag_signatures: tuple[str, ...]
    chunk_count: int
    sha256: str


@dataclass(frozen=True)
class SegmentInfo:
    marker: int
    offset: int
    end: int
    payload_offset: int
    payload_size: int

    @property
    def name(self) -> str:
        if 0xE0 <= self.marker <= 0xEF:
            return f"APP{self.marker - 0xE0}"
        return {
            0xC0: "SOF0",
            0xC2: "SOF2",
            0xD8: "SOI",
            0xD9: "EOI",
            0xDA: "SOS",
            0xDB: "DQT",
            0xC4: "DHT",
        }.get(self.marker, f"0x{self.marker:02X}")


@dataclass(frozen=True)
class MpEntryInfo:
    index: int
    attributes: int
    size: int
    data_offset: int
    absolute_offset: int
    dependent_image_1: int
    dependent_image_2: int

    @property
    def mp_type(self) -> int:
        return self.attributes & 0xFFFFFF

    @property
    def representative(self) -> bool:
        return bool(self.attributes & 0x20000000)


@dataclass(frozen=True)
class FrameInfo:
    index: int
    offset: int
    size: int
    width: Optional[int]
    height: Optional[int]
    components: tuple[tuple[int, int, int, int], ...]
    mp_attributes: Optional[int]
    mp_type: Optional[int]
    segments: tuple[SegmentInfo, ...]
    iso21496: tuple[Iso21496Record, ...]
    icc: Optional[IccSummary]


@dataclass(frozen=True)
class JpegHdrInfo:
    is_jpeg: bool
    is_mpo: bool
    frames: tuple[FrameInfo, ...]
    mp_entries: tuple[MpEntryInfo, ...]
    mpf_version: Optional[str]
    number_of_images: Optional[int]
    gain_map_frame_index: Optional[int]
    gain_map: Optional[GainMapMetadata]
    unindexed_tail_offset: int
    unindexed_tail_size: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without retaining source bytes."""

        return _json_value(self)


@dataclass(frozen=True)
class _FrameScan:
    end: int
    segments: tuple[SegmentInfo, ...]
    width: Optional[int]
    height: Optional[int]
    components: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class _TiffEntry:
    tag: int
    type_id: int
    count: int
    raw: bytes


@dataclass(frozen=True)
class _MpfIndex:
    version: Optional[str]
    number_of_images: Optional[int]
    entries: tuple[MpEntryInfo, ...]


class _Context:
    def __init__(self, strict: bool):
        self.strict = strict
        self.warnings: list[str] = []

    def issue(self, message: str) -> None:
        if self.strict:
            raise MpoMetadataError(message)
        self.warnings.append(message)


def _json_value(value: Any) -> Any:
    if isinstance(value, Rational):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "value": value.value,
        }
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _payload(data: bytes, segment: SegmentInfo) -> bytes:
    return data[segment.payload_offset : segment.payload_offset + segment.payload_size]


def _next_entropy_marker(data: bytes, start: int, limit: int) -> Optional[int]:
    pos = start
    while pos < limit - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker_start = pos
        while pos < limit and data[pos] == 0xFF:
            pos += 1
        if pos >= limit:
            return None
        marker = data[pos]
        if marker == 0x00:
            pos += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            pos += 1
            continue
        return marker_start
    return None


def _scan_frame(
    data: bytes,
    start: int,
    limit: int,
    context: _Context,
    label: str,
) -> Optional[_FrameScan]:
    if start < 0 or start + 2 > len(data) or data[start : start + 2] != b"\xFF\xD8":
        context.issue(f"{label}: missing JPEG SOI at offset {start}")
        return None
    if limit > len(data) or limit <= start + 2:
        context.issue(f"{label}: invalid frame limit {limit} for {len(data)} bytes")
        return None

    segments = [SegmentInfo(0xD8, start, start + 2, start + 2, 0)]
    width: Optional[int] = None
    height: Optional[int] = None
    components: tuple[tuple[int, int, int, int], ...] = ()
    pos = start + 2

    while pos < limit:
        if data[pos] != 0xFF:
            context.issue(f"{label}: expected marker at offset {pos}")
            next_marker = data.find(b"\xFF", pos + 1, limit)
            if next_marker < 0:
                return None
            pos = next_marker

        marker_start = pos
        while pos < limit and data[pos] == 0xFF:
            pos += 1
        if pos >= limit:
            context.issue(f"{label}: truncated marker at frame end")
            return None
        marker = data[pos]
        pos += 1

        if marker == 0x00:
            context.issue(f"{label}: stuffed byte outside entropy data at {marker_start}")
            continue
        if marker in _STANDALONE_MARKERS:
            segment = SegmentInfo(marker, marker_start, pos, pos, 0)
            segments.append(segment)
            if marker == 0xD9:
                return _FrameScan(pos, tuple(segments), width, height, components)
            if marker == 0xD8 and marker_start != start:
                context.issue(f"{label}: nested SOI at offset {marker_start}")
            continue

        if pos + 2 > limit:
            context.issue(f"{label}: truncated segment length at {marker_start}")
            return None
        segment_length = struct.unpack_from(">H", data, pos)[0]
        if segment_length < 2:
            context.issue(f"{label}: invalid segment length {segment_length} at {marker_start}")
            return None
        segment_end = pos + segment_length
        if segment_end > limit:
            context.issue(
                f"{label}: segment at {marker_start} ends at {segment_end}, past frame limit {limit}"
            )
            return None
        payload_offset = pos + 2
        payload_size = segment_length - 2
        segment = SegmentInfo(marker, marker_start, segment_end, payload_offset, payload_size)
        segments.append(segment)

        if marker in _SOF_MARKERS and width is None:
            raw = _payload(data, segment)
            if len(raw) < 6:
                context.issue(f"{label}: truncated SOF at {marker_start}")
                return None
            precision = raw[0]
            parsed_height, parsed_width = struct.unpack_from(">HH", raw, 1)
            count = raw[5]
            required = 6 + 3 * count
            if precision == 0 or count == 0 or len(raw) < required:
                context.issue(f"{label}: invalid SOF payload at {marker_start}")
                return None
            width, height = parsed_width, parsed_height
            components = tuple(
                (raw[6 + 3 * i], raw[7 + 3 * i] >> 4, raw[7 + 3 * i] & 0x0F, raw[8 + 3 * i])
                for i in range(count)
            )

        pos = segment_end
        if marker == 0xDA:
            next_marker = _next_entropy_marker(data, pos, limit)
            if next_marker is None:
                context.issue(f"{label}: entropy data has no terminating marker")
                return None
            pos = next_marker

    context.issue(f"{label}: JPEG frame has no EOI before {limit}")
    return None


def _parse_ifd(
    tiff: bytes,
    endian: str,
    offset: int,
    context: _Context,
    label: str,
) -> Optional[dict[int, _TiffEntry]]:
    if offset < 8 or offset + 2 > len(tiff):
        context.issue(f"{label}: IFD offset {offset} is out of bounds")
        return None
    count = struct.unpack_from(endian + "H", tiff, offset)[0]
    if count > 512:
        context.issue(f"{label}: unreasonable IFD entry count {count}")
        return None
    directory_end = offset + 2 + 12 * count + 4
    if directory_end > len(tiff):
        context.issue(f"{label}: truncated IFD directory")
        return None

    result: dict[int, _TiffEntry] = {}
    for index in range(count):
        entry_offset = offset + 2 + 12 * index
        tag, type_id, value_count = struct.unpack_from(endian + "HHI", tiff, entry_offset)
        type_size = _TIFF_TYPE_SIZES.get(type_id)
        if type_size is None:
            context.issue(f"{label}: unsupported TIFF type {type_id} for tag 0x{tag:04X}")
            continue
        byte_count = type_size * value_count
        value_field = tiff[entry_offset + 8 : entry_offset + 12]
        if byte_count <= 4:
            raw = value_field[:byte_count]
        else:
            value_offset = struct.unpack(endian + "I", value_field)[0]
            if value_offset < 0 or value_offset + byte_count > len(tiff):
                context.issue(
                    f"{label}: value for tag 0x{tag:04X} is out of bounds "
                    f"({value_offset}+{byte_count}>{len(tiff)})"
                )
                continue
            raw = tiff[value_offset : value_offset + byte_count]
        if tag in result:
            context.issue(f"{label}: duplicate TIFF tag 0x{tag:04X}")
            continue
        result[tag] = _TiffEntry(tag, type_id, value_count, raw)
    return result


def _parse_mpf(
    data: bytes,
    segment: SegmentInfo,
    context: _Context,
) -> Optional[_MpfIndex]:
    payload = _payload(data, segment)
    if not payload.startswith(MPF_IDENTIFIER):
        return None
    tiff = payload[len(MPF_IDENTIFIER) :]
    label = f"MPF APP2 at {segment.offset}"
    if len(tiff) < 8:
        context.issue(f"{label}: truncated TIFF header")
        return None
    if tiff[:2] == b"II":
        endian = "<"
    elif tiff[:2] == b"MM":
        endian = ">"
    else:
        context.issue(f"{label}: invalid TIFF byte order {tiff[:2]!r}")
        return None
    if struct.unpack_from(endian + "H", tiff, 2)[0] != 42:
        context.issue(f"{label}: invalid TIFF magic")
        return None
    ifd_offset = struct.unpack_from(endian + "I", tiff, 4)[0]
    entries = _parse_ifd(tiff, endian, ifd_offset, context, label)
    if entries is None:
        return None

    version: Optional[str] = None
    version_entry = entries.get(0xB000)
    if version_entry is not None:
        if version_entry.type_id != 7 or version_entry.count != 4:
            context.issue(f"{label}: MPFVersion must be four UNDEFINED bytes")
        else:
            version = version_entry.raw.decode("ascii", "replace")

    number_of_images: Optional[int] = None
    number_entry = entries.get(0xB001)
    if number_entry is not None:
        if number_entry.type_id != 4 or number_entry.count != 1 or len(number_entry.raw) != 4:
            context.issue(f"{label}: NumberOfImages must be one LONG")
        else:
            number_of_images = struct.unpack(endian + "I", number_entry.raw)[0]

    mp_entry = entries.get(0xB002)
    if mp_entry is None:
        context.issue(f"{label}: missing MPEntry tag")
        return None
    if mp_entry.type_id != 7 or mp_entry.count == 0 or mp_entry.count % 16:
        context.issue(f"{label}: MPEntry must be a non-empty UNDEFINED array divisible by 16")
        return None
    if len(mp_entry.raw) != mp_entry.count:
        context.issue(f"{label}: truncated MPEntry array")
        return None

    tiff_base = segment.payload_offset + len(MPF_IDENTIFIER)
    parsed_entries = []
    for index in range(mp_entry.count // 16):
        attributes, size, data_offset, dep1, dep2 = struct.unpack_from(
            endian + "IIIHH", mp_entry.raw, 16 * index
        )
        absolute_offset = 0 if data_offset == 0 else tiff_base + data_offset
        parsed_entries.append(
            MpEntryInfo(index, attributes, size, data_offset, absolute_offset, dep1, dep2)
        )

    if number_of_images is not None and number_of_images != len(parsed_entries):
        context.issue(
            f"{label}: NumberOfImages={number_of_images}, but MPEntry contains {len(parsed_entries)} entries"
        )
    return _MpfIndex(version, number_of_images, tuple(parsed_entries))


def _read_rational(
    body: bytes,
    offset: int,
    signed: bool,
    context: _Context,
    label: str,
) -> tuple[Optional[Rational], int]:
    if offset + 8 > len(body):
        context.issue(f"{label}: truncated rational at byte {offset}")
        return None, len(body)
    numerator = struct.unpack_from(">i" if signed else ">I", body, offset)[0]
    denominator = struct.unpack_from(">I", body, offset + 4)[0]
    if denominator == 0:
        context.issue(f"{label}: zero rational denominator at byte {offset}")
        return None, offset + 8
    return Rational(numerator, denominator), offset + 8


def _parse_gain_map_body(
    body: bytes,
    context: _Context,
    label: str,
) -> Optional[GainMapMetadata]:
    if len(body) < 5:
        context.issue(f"{label}: gain-map metadata is only {len(body)} bytes")
        return None
    minimum_version, writer_version = struct.unpack_from(">HH", body, 0)
    flags = body[4]
    is_multichannel = bool(flags & 0x80)
    use_base_colour_space = bool(flags & 0x40)
    if flags & 0x3F:
        context.issue(f"{label}: reserved ISO 21496 flags are non-zero (0x{flags:02X})")
    channel_count = 3 if is_multichannel else 1
    expected_size = 5 + 16 + channel_count * 40
    if len(body) != expected_size:
        context.issue(
            f"{label}: metadata size is {len(body)}, expected exactly {expected_size} bytes"
        )
        return None

    offset = 5
    base_headroom, offset = _read_rational(body, offset, False, context, label)
    alternate_headroom, offset = _read_rational(body, offset, False, context, label)
    if base_headroom is None or alternate_headroom is None:
        return None
    channels = []
    for _ in range(channel_count):
        gain_min, offset = _read_rational(body, offset, True, context, label)
        gain_max, offset = _read_rational(body, offset, True, context, label)
        gamma, offset = _read_rational(body, offset, False, context, label)
        base_offset, offset = _read_rational(body, offset, True, context, label)
        alternate_offset, offset = _read_rational(body, offset, True, context, label)
        values = (gain_min, gain_max, gamma, base_offset, alternate_offset)
        if any(value is None for value in values):
            return None
        channels.append(GainMapChannel(*values))  # type: ignore[arg-type]
    return GainMapMetadata(
        minimum_version,
        writer_version,
        flags,
        is_multichannel,
        use_base_colour_space,
        base_headroom,
        alternate_headroom,
        tuple(channels),
    )


def _parse_iso_records(
    data: bytes,
    segments: Iterable[SegmentInfo],
    context: _Context,
) -> tuple[Iso21496Record, ...]:
    records = []
    for segment in segments:
        if segment.marker != 0xE2:
            continue
        payload = _payload(data, segment)
        if not payload.startswith(ISO_21496_IDENTIFIER):
            continue
        body = payload[len(ISO_21496_IDENTIFIER) :]
        if len(body) == 4:
            records.append(Iso21496Record(segment.offset, 4, True, None))
            continue
        label = f"ISO 21496 APP2 at {segment.offset}"
        metadata = _parse_gain_map_body(body, context, label)
        records.append(Iso21496Record(segment.offset, len(body), False, metadata))
    return tuple(records)


def _decode_icc_text(tag: bytes, context: _Context, label: str) -> Optional[str]:
    if len(tag) < 8:
        context.issue(f"{label}: truncated ICC text tag")
        return None
    kind = tag[:4]
    if kind == b"desc":
        if len(tag) < 12:
            context.issue(f"{label}: truncated ICC desc tag")
            return None
        size = struct.unpack_from(">I", tag, 8)[0]
        if size == 0 or 12 + size > len(tag):
            context.issue(f"{label}: invalid ICC desc string length {size}")
            return None
        return tag[12 : 12 + size].rstrip(b"\x00").decode("latin-1", "replace")
    if kind == b"text":
        return tag[8:].split(b"\x00", 1)[0].decode("latin-1", "replace")
    if kind == b"mluc":
        if len(tag) < 16:
            context.issue(f"{label}: truncated ICC mluc tag")
            return None
        count, record_size = struct.unpack_from(">II", tag, 8)
        if count == 0 or record_size < 12 or 16 + count * record_size > len(tag):
            context.issue(f"{label}: invalid ICC mluc record table")
            return None
        strings = []
        for index in range(count):
            record = 16 + index * record_size
            length, offset = struct.unpack_from(">II", tag, record + 4)
            if offset + length > len(tag) or length % 2:
                context.issue(f"{label}: invalid ICC mluc string range")
                continue
            strings.append(tag[offset : offset + length].decode("utf-16-be", "replace"))
        return strings[0] if strings else None
    return None


def _parse_icc_summary(
    profile: bytes,
    chunk_count: int,
    context: _Context,
    label: str,
) -> Optional[IccSummary]:
    if len(profile) < 132:
        context.issue(f"{label}: ICC profile is only {len(profile)} bytes")
        return None
    declared_size = struct.unpack_from(">I", profile, 0)[0]
    if declared_size < 132 or declared_size > len(profile):
        context.issue(
            f"{label}: invalid declared ICC size {declared_size} for {len(profile)} collected bytes"
        )
        return None
    if declared_size != len(profile):
        context.issue(
            f"{label}: declared ICC size {declared_size} differs from collected size {len(profile)}"
        )
        if declared_size > len(profile):
            return None
        profile = profile[:declared_size]

    tag_count = struct.unpack_from(">I", profile, 128)[0]
    if tag_count > 4096 or 132 + 12 * tag_count > declared_size:
        context.issue(f"{label}: invalid ICC tag count {tag_count}")
        return None
    tags: dict[bytes, bytes] = {}
    signatures = []
    for index in range(tag_count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + 12 * index)
        if offset < 0 or offset + size > declared_size:
            context.issue(
                f"{label}: ICC tag {signature!r} range {offset}+{size} exceeds {declared_size}"
            )
            continue
        if signature in tags:
            context.issue(f"{label}: duplicate ICC tag {signature!r}")
            continue
        tags[signature] = profile[offset : offset + size]
        signatures.append(signature.decode("latin-1"))

    description = None
    if b"desc" in tags:
        description = _decode_icc_text(tags[b"desc"], context, f"{label} desc")
    copyright_text = None
    if b"cprt" in tags:
        copyright_text = _decode_icc_text(tags[b"cprt"], context, f"{label} cprt")
    return IccSummary(
        declared_size,
        struct.unpack_from(">I", profile, 8)[0],
        profile[12:16].decode("latin-1"),
        profile[16:20].decode("latin-1"),
        profile[20:24].decode("latin-1"),
        description,
        copyright_text,
        tuple(signatures),
        chunk_count,
        sha256(profile).hexdigest(),
    )


def _parse_icc(
    data: bytes,
    segments: Iterable[SegmentInfo],
    context: _Context,
    frame_label: str,
) -> Optional[IccSummary]:
    chunks: dict[int, bytes] = {}
    expected_total: Optional[int] = None
    for segment in segments:
        if segment.marker != 0xE2:
            continue
        payload = _payload(data, segment)
        if not payload.startswith(ICC_IDENTIFIER):
            continue
        if len(payload) < 14:
            context.issue(f"{frame_label}: truncated ICC APP2 at {segment.offset}")
            return None
        sequence, total = payload[12], payload[13]
        if total == 0 or sequence == 0 or sequence > total:
            context.issue(
                f"{frame_label}: invalid ICC chunk {sequence}/{total} at {segment.offset}"
            )
            return None
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            context.issue(
                f"{frame_label}: inconsistent ICC total {total}, expected {expected_total}"
            )
            return None
        if sequence in chunks:
            context.issue(f"{frame_label}: duplicate ICC chunk {sequence}")
            return None
        chunks[sequence] = payload[14:]
    if not chunks:
        return None
    assert expected_total is not None
    missing = [index for index in range(1, expected_total + 1) if index not in chunks]
    if missing:
        context.issue(f"{frame_label}: missing ICC chunks {missing}")
        return None
    profile = b"".join(chunks[index] for index in range(1, expected_total + 1))
    return _parse_icc_summary(profile, expected_total, context, f"{frame_label} ICC")


def _make_frame_info(
    data: bytes,
    index: int,
    offset: int,
    size: int,
    scan: _FrameScan,
    entry: Optional[MpEntryInfo],
    context: _Context,
) -> FrameInfo:
    label = f"frame {index}"
    iso_records = _parse_iso_records(data, scan.segments, context)
    icc = _parse_icc(data, scan.segments, context, label)
    return FrameInfo(
        index,
        offset,
        size,
        scan.width,
        scan.height,
        scan.components,
        entry.attributes if entry is not None else None,
        entry.mp_type if entry is not None else None,
        scan.segments,
        iso_records,
        icc,
    )


def _non_jpeg_info() -> JpegHdrInfo:
    return JpegHdrInfo(False, False, (), (), None, None, None, None, 0, 0, ())


def inspect_bytes(data: bytes, *, strict: bool = False) -> JpegHdrInfo:
    """Inspect a JPEG/MPO byte string without decoding pixels.

    In strict mode malformed metadata raises :class:`MpoMetadataError`.  The
    default tolerant mode returns all safely decoded information and records
    problems in ``warnings``.  The function never reads outside ``data``.
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not data.startswith(b"\xFF\xD8"):
        return _non_jpeg_info()

    context = _Context(strict)
    root_scan = _scan_frame(data, 0, len(data), context, "root frame")
    if root_scan is None:
        return JpegHdrInfo(
            True,
            False,
            (),
            (),
            None,
            None,
            None,
            None,
            0,
            len(data),
            tuple(context.warnings),
        )

    mpf_segments = [
        segment
        for segment in root_scan.segments
        if segment.marker == 0xE2 and _payload(data, segment).startswith(MPF_IDENTIFIER)
    ]
    if not mpf_segments:
        root = _make_frame_info(
            data, 0, 0, root_scan.end, root_scan, None, context
        )
        return JpegHdrInfo(
            True,
            False,
            (root,),
            (),
            None,
            None,
            None,
            None,
            root_scan.end,
            len(data) - root_scan.end,
            tuple(context.warnings),
        )
    if len(mpf_segments) > 1:
        context.issue(f"root frame: found {len(mpf_segments)} MPF APP2 segments")

    mpf = _parse_mpf(data, mpf_segments[0], context)
    if mpf is None or len(mpf.entries) < 2:
        root = _make_frame_info(data, 0, 0, root_scan.end, root_scan, None, context)
        return JpegHdrInfo(
            True,
            False,
            (root,),
            (),
            mpf.version if mpf else None,
            mpf.number_of_images if mpf else None,
            None,
            None,
            root_scan.end,
            len(data) - root_scan.end,
            tuple(context.warnings),
        )

    valid = True
    intervals = []
    scans: list[Optional[_FrameScan]] = []
    for entry in mpf.entries:
        start = entry.absolute_offset
        end = start + entry.size
        if entry.size <= 0 or start < 0 or end > len(data):
            context.issue(
                f"MPEntry {entry.index}: range {start}+{entry.size} exceeds file size {len(data)}"
            )
            valid = False
            scans.append(None)
            continue
        if data[start : start + 2] != b"\xFF\xD8":
            context.issue(f"MPEntry {entry.index}: no SOI at absolute offset {start}")
            valid = False
            scans.append(None)
            continue
        scan = root_scan if start == 0 else _scan_frame(
            data, start, end, context, f"MPEntry {entry.index}"
        )
        if scan is None or scan.end != end:
            if scan is not None:
                context.issue(
                    f"MPEntry {entry.index}: EOI ends at {scan.end}, declared end is {end}"
                )
            valid = False
            scans.append(scan)
            continue
        scans.append(scan)
        intervals.append((start, end, entry.index))

    if mpf.entries[0].absolute_offset != 0:
        context.issue("MPEntry 0 is not the primary image at file offset 0")
        valid = False
    for previous, current in zip(sorted(intervals), sorted(intervals)[1:]):
        if current[0] < previous[1]:
            context.issue(
                f"MPEntry {previous[2]} overlaps MPEntry {current[2]} at offset {current[0]}"
            )
            valid = False

    if not valid or any(scan is None for scan in scans):
        root = _make_frame_info(data, 0, 0, root_scan.end, root_scan, None, context)
        return JpegHdrInfo(
            True,
            False,
            (root,),
            mpf.entries,
            mpf.version,
            mpf.number_of_images,
            None,
            None,
            root_scan.end,
            len(data) - root_scan.end,
            tuple(context.warnings),
        )

    frames = tuple(
        _make_frame_info(
            data,
            index,
            entry.absolute_offset,
            entry.size,
            scans[index],  # type: ignore[arg-type]
            entry,
            context,
        )
        for index, entry in enumerate(mpf.entries)
    )

    full_gain_maps = [
        (frame.index, record.metadata)
        for frame in frames[1:]
        for record in frame.iso21496
        if record.metadata is not None
    ]
    gain_map_frame_index: Optional[int] = None
    gain_map: Optional[GainMapMetadata] = None
    if full_gain_maps:
        if len(full_gain_maps) > 1:
            context.issue(f"MPO has {len(full_gain_maps)} full ISO 21496 metadata records")
        gain_map_frame_index, gain_map = full_gain_maps[0]
        primary = frames[0]
        gain_frame = frames[gain_map_frame_index]
        if (
            primary.width is not None
            and primary.height is not None
            and gain_frame.width is not None
            and gain_frame.height is not None
            and (
                primary.width != 2 * gain_frame.width
                or primary.height != 2 * gain_frame.height
            )
        ):
            context.issue(
                "gain-map dimensions are not exactly half the primary image dimensions"
            )

    indexed_end = max(frame.offset + frame.size for frame in frames)
    return JpegHdrInfo(
        True,
        True,
        frames,
        mpf.entries,
        mpf.version,
        mpf.number_of_images,
        gain_map_frame_index,
        gain_map,
        indexed_end,
        len(data) - indexed_end,
        tuple(context.warnings),
    )


def inspect_file(
    path: Union[str, Path],
    *,
    strict: bool = False,
) -> JpegHdrInfo:
    """Read and inspect one JPEG/MPO file."""

    return inspect_bytes(Path(path).read_bytes(), strict=strict)


def extract_frame(data: bytes, info: JpegHdrInfo, index: int) -> memoryview:
    """Return a zero-copy view of one already validated JPEG frame."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    try:
        frame = info.frames[index]
    except IndexError as error:
        raise IndexError(f"frame index {index} is out of range") from error
    end = frame.offset + frame.size
    if frame.offset < 0 or end > len(data):
        raise MpoMetadataError("frame range no longer matches the supplied data")
    return memoryview(data)[frame.offset:end]


__all__ = [
    "FrameInfo",
    "GainMapChannel",
    "GainMapMetadata",
    "IccSummary",
    "Iso21496Record",
    "JpegHdrInfo",
    "MpEntryInfo",
    "MpoMetadataError",
    "Rational",
    "SegmentInfo",
    "extract_frame",
    "inspect_bytes",
    "inspect_file",
]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect JPEG/MPO, ISO 21496 and per-frame ICC metadata")
    parser.add_argument("paths", nargs="+", help="JPEG/MPO files or directories")
    parser.add_argument("--strict", action="store_true", help="fail on malformed metadata")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args()
    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(item for item in path.iterdir() if item.is_file()))
        else:
            files.append(path)
    result = {str(path): inspect_file(path, strict=args.strict).as_dict() for path in files}
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, allow_nan=False))


if __name__ == "__main__":
    _main()
