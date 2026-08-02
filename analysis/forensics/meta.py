"""解 MPF / EXIF / ICC / APP 段内容。"""
import struct, sys, os

TIFF_TYPES = {1:('B',1),2:('s',1),3:('H',2),4:('I',4),5:('II',8),
              6:('b',1),7:('B',1),8:('h',2),9:('i',4),10:('ii',8),
              11:('f',4),12:('d',8)}

EXIF_TAGS = {
 0x010e:'ImageDescription',0x010f:'Make',0x0110:'Model',0x0112:'Orientation',
 0x011a:'XResolution',0x011b:'YResolution',0x0128:'ResolutionUnit',
 0x0131:'Software',0x0132:'DateTime',0x013b:'Artist',0x8298:'Copyright',
 0x8769:'ExifIFD',0x8825:'GPSIFD',0xa005:'InteropIFD',
 0x829a:'ExposureTime',0x829d:'FNumber',0x8822:'ExposureProgram',0x8827:'ISO',
 0x9000:'ExifVersion',0x9003:'DateTimeOriginal',0x9004:'DateTimeDigitized',
 0x9201:'ShutterSpeedValue',0x9202:'ApertureValue',0x9204:'ExposureBias',
 0x9205:'MaxApertureValue',0x9207:'MeteringMode',0x9208:'LightSource',
 0x9209:'Flash',0x920a:'FocalLength',0x927c:'MakerNote',0x9286:'UserComment',
 0xa000:'FlashpixVersion',0xa001:'ColorSpace',0xa002:'PixelXDimension',
 0xa003:'PixelYDimension',0xa20e:'FocalPlaneXRes',0xa402:'ExposureMode',
 0xa403:'WhiteBalance',0xa405:'FocalLengthIn35mm',0xa406:'SceneCaptureType',
 0xa430:'CameraOwnerName',0xa431:'BodySerialNumber',0xa432:'LensSpecification',
 0xa433:'LensMake',0xa434:'LensModel',0x0100:'ImageWidth',0x0101:'ImageLength',
 0x0102:'BitsPerSample',0x0103:'Compression',0x0106:'PhotometricInterp',
 0x0201:'JPEGInterchangeFormat',0x0202:'JPEGInterchangeFormatLength',
 0x0213:'YCbCrPositioning',0x9101:'ComponentsConfiguration',
 0x9102:'CompressedBitsPerPixel',0x9203:'BrightnessValue',0x9290:'SubSecTime',
 0x9291:'SubSecTimeOriginal',0x9292:'SubSecTimeDigitized',0xa401:'CustomRendered',
 0xa404:'DigitalZoomRatio',0xa407:'GainControl',0xa408:'Contrast',
 0xa409:'Saturation',0xa40a:'Sharpness',0xa420:'ImageUniqueID',
 0x882a:'TimeZoneOffset',0x9010:'OffsetTime',0x9011:'OffsetTimeOriginal',
 0x0116:'RowsPerStrip',0x0117:'StripByteCounts',0x0111:'StripOffsets',
 0xb001:'MPFVersion',0xb002:'NumberOfImages',0xb003:'MPEntry',
 0xb004:'ImageUIDList',0xb005:'TotalFrames',0xb000:'MPFVersionTag',
}


def read_ifd(buf, base, off, endian, depth=0, seen=None, maxd=3):
    """解析 TIFF IFD，返回 [(tag, type, count, value)]"""
    if seen is None:
        seen = set()
    out = []
    if off in seen or depth > maxd or off + 2 > len(buf):
        return out
    seen.add(off)
    n = struct.unpack(endian + 'H', buf[off:off+2])[0]
    if n > 1000:
        return out
    for i in range(n):
        p = off + 2 + i*12
        if p + 12 > len(buf):
            break
        tag, typ, cnt = struct.unpack(endian + 'HHI', buf[p:p+8])
        raw = buf[p+8:p+12]
        if typ not in TIFF_TYPES:
            out.append((tag, typ, cnt, '<badtype>'))
            continue
        _, esz = TIFF_TYPES[typ]
        total = esz * cnt
        if total > 4:
            voff = struct.unpack(endian + 'I', raw)[0]
            data = buf[voff:voff+total]
        else:
            data = raw[:total]
        val = decode_val(typ, cnt, data, endian)
        out.append((tag, typ, cnt, val))
    return out


def decode_val(typ, cnt, data, endian):
    try:
        if typ == 2:
            return data.split(b'\x00')[0].decode('latin1')
        if typ in (1, 7):
            return data[:32].hex() if cnt > 8 else list(data)
        if typ == 3:
            return list(struct.unpack(endian + '%dH' % (len(data)//2), data[:len(data)//2*2]))
        if typ == 4:
            return list(struct.unpack(endian + '%dI' % (len(data)//4), data[:len(data)//4*4]))
        if typ == 5:
            v = struct.unpack(endian + '%dI' % (len(data)//4), data[:len(data)//4*4])
            return [(v[i], v[i+1]) for i in range(0, len(v)-1, 2)]
        if typ == 10:
            v = struct.unpack(endian + '%di' % (len(data)//4), data[:len(data)//4*4])
            return [(v[i], v[i+1]) for i in range(0, len(v)-1, 2)]
        if typ == 9:
            return list(struct.unpack(endian + '%di' % (len(data)//4), data[:len(data)//4*4]))
    except Exception as e:
        return '<err %s>' % e
    return data[:16].hex()
