"""手写 JPEG 段解析 + TIFF/EXIF IFD 遍历（不依赖 PIL/exiftool）。"""
import struct, sys, os, json

MARKER_NAMES = {
    0xC0: "SOF0", 0xC1: "SOF1", 0xC2: "SOF2", 0xC3: "SOF3", 0xC5: "SOF5",
    0xC6: "SOF6", 0xC7: "SOF7", 0xC9: "SOF9", 0xCA: "SOF10", 0xCB: "SOF11",
    0xCD: "SOF13", 0xCE: "SOF14", 0xCF: "SOF15", 0xC4: "DHT", 0xC8: "JPG",
    0xCC: "DAC", 0xD8: "SOI", 0xD9: "EOI", 0xDA: "SOS", 0xDB: "DQT",
    0xDC: "DNL", 0xDD: "DRI", 0xDE: "DHP", 0xDF: "EXP", 0xFE: "COM",
}
for i in range(16):
    MARKER_NAMES[0xE0 + i] = "APP%d" % i


def scan_segments(data):
    """返回 [(marker_byte, name, offset_of_marker, payload_bytes)]，SOS 后停止解析 entropy 数据但继续找后续标记。"""
    segs = []
    i = 0
    n = len(data)
    if data[0:2] != b"\xff\xd8":
        return segs, "NOT_JPEG"
    segs.append((0xD8, "SOI", 0, b""))
    i = 2
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xFF, 0x00):
            i += 1
            continue
        if 0xD0 <= m <= 0xD7:  # RSTn
            i += 2
            continue
        if m == 0xD8:
            segs.append((m, "SOI", i, b""))
            i += 2
            continue
        if m == 0xD9:
            segs.append((m, "EOI", i, b""))
            i += 2
            continue
        if i + 4 > n:
            break
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + ln]
        segs.append((m, MARKER_NAMES.get(m, "0x%02X" % m), i, payload))
        i = i + 2 + ln
        if m == 0xDA:
            # 跳过 entropy-coded 数据，找下一个非 RST 标记
            j = i
            while j < n - 1:
                if data[j] == 0xFF and data[j + 1] != 0x00 and not (0xD0 <= data[j + 1] <= 0xD7) and data[j + 1] != 0xFF:
                    break
                j += 1
            i = j
    return segs, None


# --- TIFF/EXIF ---
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
TYPE_NAME = {1: "BYTE", 2: "ASCII", 3: "SHORT", 4: "LONG", 5: "RATIONAL", 6: "SBYTE",
             7: "UNDEFINED", 8: "SSHORT", 9: "SLONG", 10: "SRATIONAL", 11: "FLOAT",
             12: "DOUBLE", 13: "IFD"}

TAGS = {
0x00fe:"NewSubfileType",0x0100:"ImageWidth",0x0101:"ImageLength",0x0102:"BitsPerSample",
0x0103:"Compression",0x0106:"PhotometricInterpretation",0x010e:"ImageDescription",
0x010f:"Make",0x0110:"Model",0x0111:"StripOffsets",0x0112:"Orientation",
0x0115:"SamplesPerPixel",0x0116:"RowsPerStrip",0x0117:"StripByteCounts",
0x011a:"XResolution",0x011b:"YResolution",0x011c:"PlanarConfiguration",
0x0128:"ResolutionUnit",0x0131:"Software",0x0132:"DateTime",0x013b:"Artist",
0x013e:"WhitePoint",0x013f:"PrimaryChromaticities",0x0201:"JPEGInterchangeFormat",
0x0202:"JPEGInterchangeFormatLength",0x0211:"YCbCrCoefficients",0x0212:"YCbCrSubSampling",
0x0213:"YCbCrPositioning",0x0214:"ReferenceBlackWhite",0x8298:"Copyright",
0x829a:"ExposureTime",0x829d:"FNumber",0x8769:"ExifIFDPointer",0x8822:"ExposureProgram",
0x8824:"SpectralSensitivity",0x8825:"GPSInfoIFDPointer",0x8827:"ISOSpeedRatings",
0x8828:"OECF",0x8830:"SensitivityType",0x8831:"StandardOutputSensitivity",
0x8832:"RecommendedExposureIndex",0x8833:"ISOSpeed",0x9000:"ExifVersion",
0x9003:"DateTimeOriginal",0x9004:"DateTimeDigitized",0x9010:"OffsetTime",
0x9011:"OffsetTimeOriginal",0x9012:"OffsetTimeDigitized",0x9101:"ComponentsConfiguration",
0x9102:"CompressedBitsPerPixel",0x9201:"ShutterSpeedValue",0x9202:"ApertureValue",
0x9203:"BrightnessValue",0x9204:"ExposureBiasValue",0x9205:"MaxApertureValue",
0x9206:"SubjectDistance",0x9207:"MeteringMode",0x9208:"LightSource",0x9209:"Flash",
0x920a:"FocalLength",0x9214:"SubjectArea",0x927c:"MakerNote",0x9286:"UserComment",
0x9290:"SubSecTime",0x9291:"SubSecTimeOriginal",0x9292:"SubSecTimeDigitized",
0xa000:"FlashpixVersion",0xa001:"ColorSpace",0xa002:"ExifImageWidth",0xa003:"ExifImageHeight",
0xa004:"RelatedSoundFile",0xa005:"InteroperabilityIFDPointer",0xa20b:"FlashEnergy",
0xa20e:"FocalPlaneXResolution",0xa20f:"FocalPlaneYResolution",0xa210:"FocalPlaneResolutionUnit",
0xa214:"SubjectLocation",0xa215:"ExposureIndex",0xa217:"SensingMethod",0xa300:"FileSource",
0xa301:"SceneType",0xa302:"CFAPattern",0xa401:"CustomRendered",0xa402:"ExposureMode",
0xa403:"WhiteBalance",0xa404:"DigitalZoomRatio",0xa405:"FocalLengthIn35mmFilm",
0xa406:"SceneCaptureType",0xa407:"GainControl",0xa408:"Contrast",0xa409:"Saturation",
0xa40a:"Sharpness",0xa40b:"DeviceSettingDescription",0xa40c:"SubjectDistanceRange",
0xa420:"ImageUniqueID",0xa430:"CameraOwnerName",0xa431:"BodySerialNumber",
0xa432:"LensSpecification",0xa433:"LensMake",0xa434:"LensModel",0xa435:"LensSerialNumber",
0xa436:"Title",0xa460:"CompositeImage",0xa461:"SourceImageNumberOfCompositeImage",
0xa462:"SourceExposureTimesOfCompositeImage",0xc4a5:"PrintIM",0xea1c:"Padding",
0x0001:"InteropIndex_or_GPSLatitudeRef",0x0002:"InteropVersion_or_GPSLatitude",
}
GPS_TAGS = {
0x0000:"GPSVersionID",0x0001:"GPSLatitudeRef",0x0002:"GPSLatitude",0x0003:"GPSLongitudeRef",
0x0004:"GPSLongitude",0x0005:"GPSAltitudeRef",0x0006:"GPSAltitude",0x0007:"GPSTimeStamp",
0x0008:"GPSSatellites",0x0009:"GPSStatus",0x000a:"GPSMeasureMode",0x000b:"GPSDOP",
0x000c:"GPSSpeedRef",0x000d:"GPSSpeed",0x000e:"GPSTrackRef",0x000f:"GPSTrack",
0x0010:"GPSImgDirectionRef",0x0011:"GPSImgDirection",0x0012:"GPSMapDatum",
0x0013:"GPSDestLatitudeRef",0x0014:"GPSDestLatitude",0x001b:"GPSProcessingMethod",
0x001d:"GPSDateStamp",0x001f:"GPSHPositioningError",
}
INTEROP_TAGS = {0x0001:"InteropIndex",0x0002:"InteropVersion",0x1000:"RelatedImageFileFormat",
                0x1001:"RelatedImageWidth",0x1002:"RelatedImageLength"}


def read_val(buf, endian, typ, count, valoff_bytes, base_for_offsets=0):
    size = TYPE_SIZE.get(typ, 1) * count
    if size <= 4:
        raw = valoff_bytes[:size]
        off = None
    else:
        off = struct.unpack(endian + "I", valoff_bytes)[0]
        raw = buf[base_for_offsets + off: base_for_offsets + off + size]
    return raw, off, size


def decode_val(raw, endian, typ, count):
    try:
        if typ == 2:
            return raw.split(b"\x00")[0].decode("utf-8", "replace")
        if typ in (1, 6, 7):
            return list(raw)
        if typ == 3:
            return list(struct.unpack(endian + "%dH" % count, raw[:2 * count]))
        if typ == 8:
            return list(struct.unpack(endian + "%dh" % count, raw[:2 * count]))
        if typ == 4:
            return list(struct.unpack(endian + "%dI" % count, raw[:4 * count]))
        if typ == 9:
            return list(struct.unpack(endian + "%di" % count, raw[:4 * count]))
        if typ == 5:
            v = struct.unpack(endian + "%dI" % (2 * count), raw[:8 * count])
            return [(v[2 * i], v[2 * i + 1]) for i in range(count)]
        if typ == 10:
            v = struct.unpack(endian + "%di" % (2 * count), raw[:8 * count])
            return [(v[2 * i], v[2 * i + 1]) for i in range(count)]
        if typ == 11:
            return list(struct.unpack(endian + "%df" % count, raw[:4 * count]))
        if typ == 12:
            return list(struct.unpack(endian + "%dd" % count, raw[:8 * count]))
    except Exception as e:
        return "DECODE_ERR:%r" % (e,)
    return list(raw)


def parse_ifd(buf, endian, ifd_off, tagmap, base=0, depth=0, seen=None):
    """返回 (entries, next_ifd_off). entries: list of dict"""
    if seen is None:
        seen = set()
    if ifd_off in seen or depth > 6:
        return [], 0
    seen.add(ifd_off)
    out = []
    p = base + ifd_off
    if p + 2 > len(buf):
        return [], 0
    ncount = struct.unpack(endian + "H", buf[p:p + 2])[0]
    if ncount > 512:
        return [], 0
    p += 2
    for k in range(ncount):
        if p + 12 > len(buf):
            break
        tag, typ, cnt = struct.unpack(endian + "HHI", buf[p:p + 8])
        vb = buf[p + 8:p + 12]
        raw, off, size = read_val(buf, endian, typ, cnt, vb, base)
        e = {
            "tag": tag, "tag_hex": "0x%04x" % tag,
            "name": tagmap.get(tag, TAGS.get(tag, "Unknown_0x%04x" % tag)),
            "type": TYPE_NAME.get(typ, str(typ)), "type_id": typ, "count": cnt,
            "size": size, "value_offset": off, "entry_offset": p - base,
            "raw": raw,
            "value": decode_val(raw, endian, typ, cnt),
        }
        out.append(e)
        p += 12
    nxt = 0
    if p + 4 <= len(buf):
        nxt = struct.unpack(endian + "I", buf[p:p + 4])[0]
    return out, nxt


def parse_tiff(tiff):
    """tiff: APP1 里去掉 'Exif\\0\\0' 后的 TIFF 块。返回结构化 dict。"""
    res = {"ok": False}
    if len(tiff) < 8:
        return res
    bo = tiff[0:2]
    if bo == b"II":
        endian = "<"
    elif bo == b"MM":
        endian = ">"
    else:
        res["err"] = "bad byte order %r" % bo
        return res
    magic = struct.unpack(endian + "H", tiff[2:4])[0]
    ifd0_off = struct.unpack(endian + "I", tiff[4:8])[0]
    res.update({"ok": True, "byte_order": bo.decode(), "magic": magic, "ifd0_offset": ifd0_off,
                "tiff_len": len(tiff)})
    ifds = {}
    ifd0, nxt = parse_ifd(tiff, endian, ifd0_off, TAGS)
    ifds["IFD0"] = ifd0
    res["ifd1_offset"] = nxt
    # 子 IFD
    def find(entries, name):
        for e in entries:
            if e["name"] == name:
                return e
        return None
    ex = find(ifd0, "ExifIFDPointer")
    if ex and ex["value"]:
        ents, _ = parse_ifd(tiff, endian, ex["value"][0], TAGS)
        ifds["ExifIFD"] = ents
        io = find(ents, "InteroperabilityIFDPointer")
        if io and io["value"]:
            e2, _ = parse_ifd(tiff, endian, io["value"][0], INTEROP_TAGS)
            ifds["InteropIFD"] = e2
    gp = find(ifd0, "GPSInfoIFDPointer")
    if gp and gp["value"]:
        ents, _ = parse_ifd(tiff, endian, gp["value"][0], GPS_TAGS)
        ifds["GPSIFD"] = ents
    if nxt:
        ents, nxt2 = parse_ifd(tiff, endian, nxt, TAGS)
        ifds["IFD1"] = ents
        res["ifd2_offset"] = nxt2
    res["ifds"] = ifds
    res["endian"] = endian
    return res
