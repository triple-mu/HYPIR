"""手工解析 JPEG 段结构 + TIFF/EXIF IFD 树。不依赖 PIL/exiftool。"""
import struct, sys, os, json, zlib, binascii

# ---------------- JPEG 段扫描 ----------------
MARKERS = {
    0xC0:'SOF0',0xC1:'SOF1',0xC2:'SOF2',0xC3:'SOF3',0xC5:'SOF5',0xC6:'SOF6',0xC7:'SOF7',
    0xC9:'SOF9',0xCA:'SOF10',0xCB:'SOF11',0xCD:'SOF13',0xCE:'SOF14',0xCF:'SOF15',
    0xC4:'DHT',0xCC:'DAC',0xD8:'SOI',0xD9:'EOI',0xDA:'SOS',0xDB:'DQT',0xDC:'DNL',
    0xDD:'DRI',0xDE:'DHP',0xDF:'EXP',0xFE:'COM',
}
for i in range(16):
    MARKERS[0xE0+i] = 'APP%d' % i

def scan_jpeg(data):
    """返回段列表 [(marker_name, offset, length, payload)]，SOS 之后只到 EOI。"""
    segs = []
    i = 0
    n = len(data)
    if data[0:2] != b'\xff\xd8':
        return None
    segs.append(('SOI', 0, 0, b''))
    i = 2
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i+1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x00:
            i += 2
            continue
        name = MARKERS.get(m, 'M%02X' % m)
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7 or m == 0x01:
            segs.append((name, i, 0, b''))
            i += 2
            if m == 0xD9:
                break
            continue
        if i + 4 > n:
            break
        L = struct.unpack('>H', data[i+2:i+4])[0]
        payload = data[i+4:i+2+L]
        segs.append((name, i, L, payload))
        i = i + 2 + L
        if m == 0xDA:
            # 跳过熵编码数据，找下一个非 RST 标记
            j = i
            while j < n - 1:
                if data[j] == 0xFF and data[j+1] != 0x00 and not (0xD0 <= data[j+1] <= 0xD7):
                    break
                j += 1
            i = j
    return segs

def parse_sof(payload):
    prec = payload[0]
    h, w = struct.unpack('>HH', payload[1:5])
    nc = payload[5]
    comps = []
    for k in range(nc):
        cid, hv, tq = payload[6+k*3], payload[7+k*3], payload[8+k*3]
        comps.append((cid, hv >> 4, hv & 15, tq))
    return dict(prec=prec, w=w, h=h, ncomp=nc, comps=comps)

def parse_dqt(payload):
    out = []
    i = 0
    while i < len(payload):
        pq = payload[i] >> 4
        tq = payload[i] & 15
        i += 1
        if pq == 0:
            tbl = list(payload[i:i+64]); i += 64
        else:
            tbl = list(struct.unpack('>64H', payload[i:i+128])); i += 128
        out.append((tq, pq, tbl))
    return out

# ---------------- TIFF / EXIF ----------------
TYPE_SIZE = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8,13:4}
TYPE_NAME = {1:'BYTE',2:'ASCII',3:'SHORT',4:'LONG',5:'RATIONAL',6:'SBYTE',7:'UNDEFINED',
             8:'SSHORT',9:'SLONG',10:'SRATIONAL',11:'FLOAT',12:'DOUBLE',13:'IFD'}

TAGS = {
 0x0100:'ImageWidth',0x0101:'ImageLength',0x0102:'BitsPerSample',0x0103:'Compression',
 0x0106:'PhotometricInterpretation',0x010E:'ImageDescription',0x010F:'Make',0x0110:'Model',
 0x0111:'StripOffsets',0x0112:'Orientation',0x0115:'SamplesPerPixel',0x0116:'RowsPerStrip',
 0x0117:'StripByteCounts',0x011A:'XResolution',0x011B:'YResolution',0x011C:'PlanarConfiguration',
 0x0128:'ResolutionUnit',0x0131:'Software',0x0132:'DateTime',0x013B:'Artist',0x013E:'WhitePoint',
 0x013F:'PrimaryChromaticities',0x0201:'JPEGInterchangeFormat',0x0202:'JPEGInterchangeFormatLength',
 0x0211:'YCbCrCoefficients',0x0212:'YCbCrSubSampling',0x0213:'YCbCrPositioning',
 0x0214:'ReferenceBlackWhite',0x8298:'Copyright',0x829A:'ExposureTime',0x829D:'FNumber',
 0x8769:'ExifIFDPointer',0x8773:'InterColorProfile',0x8822:'ExposureProgram',
 0x8824:'SpectralSensitivity',0x8825:'GPSInfoIFDPointer',0x8827:'ISOSpeedRatings',
 0x8828:'OECF',0x8830:'SensitivityType',0x8831:'StandardOutputSensitivity',
 0x8832:'RecommendedExposureIndex',0x8833:'ISOSpeed',0x9000:'ExifVersion',
 0x9003:'DateTimeOriginal',0x9004:'DateTimeDigitized',0x9010:'OffsetTime',
 0x9011:'OffsetTimeOriginal',0x9012:'OffsetTimeDigitized',
 0x9101:'ComponentsConfiguration',0x9102:'CompressedBitsPerPixel',0x9201:'ShutterSpeedValue',
 0x9202:'ApertureValue',0x9203:'BrightnessValue',0x9204:'ExposureBiasValue',
 0x9205:'MaxApertureValue',0x9206:'SubjectDistance',0x9207:'MeteringMode',0x9208:'LightSource',
 0x9209:'Flash',0x920A:'FocalLength',0x9214:'SubjectArea',0x927C:'MakerNote',
 0x9286:'UserComment',0x9290:'SubSecTime',0x9291:'SubSecTimeOriginal',0x9292:'SubSecTimeDigitized',
 0xA000:'FlashpixVersion',0xA001:'ColorSpace',0xA002:'ExifImageWidth',0xA003:'ExifImageHeight',
 0xA004:'RelatedSoundFile',0xA005:'InteroperabilityIFDPointer',0xA20B:'FlashEnergy',
 0xA20E:'FocalPlaneXResolution',0xA20F:'FocalPlaneYResolution',0xA210:'FocalPlaneResolutionUnit',
 0xA214:'SubjectLocation',0xA215:'ExposureIndex',0xA217:'SensingMethod',0xA300:'FileSource',
 0xA301:'SceneType',0xA302:'CFAPattern',0xA401:'CustomRendered',0xA402:'ExposureMode',
 0xA403:'WhiteBalance',0xA404:'DigitalZoomRatio',0xA405:'FocalLengthIn35mmFilm',
 0xA406:'SceneCaptureType',0xA407:'GainControl',0xA408:'Contrast',0xA409:'Saturation',
 0xA40A:'Sharpness',0xA40B:'DeviceSettingDescription',0xA40C:'SubjectDistanceRange',
 0xA420:'ImageUniqueID',0xA430:'CameraOwnerName',0xA431:'BodySerialNumber',
 0xA432:'LensSpecification',0xA433:'LensMake',0xA434:'LensModel',0xA435:'LensSerialNumber',
 0xA460:'CompositeImage',0xA461:'SourceImageNumberOfCompositeImage',
 0xA462:'SourceExposureTimesOfCompositeImage',0xA500:'Gamma',
 0xC4A5:'PrintIM', 0x00FE:'NewSubfileType', 0x9C9B:'XPTitle',0x9C9C:'XPComment',
 0x9C9D:'XPAuthor',0x9C9E:'XPKeywords',0x9C9F:'XPSubject',
}
GPSTAGS = {
 0x00:'GPSVersionID',0x01:'GPSLatitudeRef',0x02:'GPSLatitude',0x03:'GPSLongitudeRef',
 0x04:'GPSLongitude',0x05:'GPSAltitudeRef',0x06:'GPSAltitude',0x07:'GPSTimeStamp',
 0x08:'GPSSatellites',0x09:'GPSStatus',0x0A:'GPSMeasureMode',0x0B:'GPSDOP',
 0x0C:'GPSSpeedRef',0x0D:'GPSSpeed',0x0E:'GPSTrackRef',0x0F:'GPSTrack',
 0x10:'GPSImgDirectionRef',0x11:'GPSImgDirection',0x12:'GPSMapDatum',
 0x13:'GPSDestLatitudeRef',0x14:'GPSDestLatitude',0x1B:'GPSProcessingMethod',
 0x1C:'GPSAreaInformation',0x1D:'GPSDateStamp',0x1E:'GPSDifferential',0x1F:'GPSHPositioningError',
}

def read_ifd(buf, off, endian, tagmap, base=0):
    """返回 (entries, next_ifd_off)。entries: list of dict"""
    E = endian
    if off + 2 > len(buf):
        return [], 0
    cnt = struct.unpack(E+'H', buf[off:off+2])[0]
    entries = []
    p = off + 2
    for k in range(cnt):
        if p + 12 > len(buf):
            break
        tag, typ, num = struct.unpack(E+'HHI', buf[p:p+8])
        valoff_raw = buf[p+8:p+12]
        tsz = TYPE_SIZE.get(typ, 0)
        total = tsz * num if tsz else 0
        inline = total <= 4
        if inline:
            raw = valoff_raw[:total] if total else b''
            dataoff = p + 8
        else:
            dataoff = struct.unpack(E+'I', valoff_raw)[0] + base
            raw = buf[dataoff:dataoff+total] if dataoff + total <= len(buf) else buf[dataoff:]
        entries.append(dict(
            tag=tag, name=tagmap.get(tag, 'Unknown_0x%04X' % tag), type=typ,
            typename=TYPE_NAME.get(typ, str(typ)), count=num, entry_off=p,
            data_off=dataoff, nbytes=total, inline=inline, raw=raw,
            value=decode_val(typ, num, raw, E)))
        p += 12
    nxt = 0
    if p + 4 <= len(buf):
        nxt = struct.unpack(E+'I', buf[p:p+4])[0]
    return entries, nxt

def decode_val(typ, num, raw, E):
    try:
        if typ == 2:
            s = raw.split(b'\x00')[0]
            return s.decode('utf-8', 'replace')
        if typ in (1, 6, 7):
            return raw
        if typ == 3:
            v = list(struct.unpack(E+'%dH' % (len(raw)//2), raw[:len(raw)//2*2]))
        elif typ == 8:
            v = list(struct.unpack(E+'%dh' % (len(raw)//2), raw[:len(raw)//2*2]))
        elif typ == 4:
            v = list(struct.unpack(E+'%dI' % (len(raw)//4), raw[:len(raw)//4*4]))
        elif typ == 9:
            v = list(struct.unpack(E+'%di' % (len(raw)//4), raw[:len(raw)//4*4]))
        elif typ == 5:
            n = len(raw)//8
            v = [struct.unpack(E+'II', raw[i*8:i*8+8]) for i in range(n)]
            v = ['%d/%d' % t for t in v]
        elif typ == 10:
            n = len(raw)//8
            v = [struct.unpack(E+'ii', raw[i*8:i*8+8]) for i in range(n)]
            v = ['%d/%d' % t for t in v]
        elif typ == 11:
            v = list(struct.unpack(E+'%df' % (len(raw)//4), raw[:len(raw)//4*4]))
        elif typ == 12:
            v = list(struct.unpack(E+'%dd' % (len(raw)//8), raw[:len(raw)//8*8]))
        else:
            return raw
        return v[0] if len(v) == 1 else v
    except Exception as e:
        return 'ERR:%s' % e

def parse_tiff(tiff, label=''):
    """解析 TIFF 块（EXIF APP1 去掉 'Exif\0\0' 之后），返回 IFD 树。"""
    if len(tiff) < 8:
        return None
    bo = tiff[0:2]
    if bo == b'II':
        E = '<'
    elif bo == b'MM':
        E = '>'
    else:
        return None
    magic = struct.unpack(E+'H', tiff[2:4])[0]
    ifd0_off = struct.unpack(E+'I', tiff[4:8])[0]
    res = dict(byteorder=bo.decode(), magic=magic, ifd0_off=ifd0_off, tiff_len=len(tiff), ifds=[])
    seen = set()
    off = ifd0_off
    idx = 0
    while off and off not in seen and off < len(tiff):
        seen.add(off)
        ents, nxt = read_ifd(tiff, off, E, TAGS)
        ifd = dict(name='IFD%d' % idx, off=off, entries=ents, next=nxt, sub={})
        for e in ents:
            if e['name'] == 'ExifIFDPointer':
                so = e['value'] if isinstance(e['value'], int) else None
                if so and so < len(tiff):
                    se, _ = read_ifd(tiff, so, E, TAGS)
                    ifd['sub']['ExifIFD'] = dict(off=so, entries=se)
                    for e2 in se:
                        if e2['name'] == 'InteroperabilityIFDPointer' and isinstance(e2['value'], int) and e2['value'] < len(tiff):
                            ie, _ = read_ifd(tiff, e2['value'], E, TAGS)
                            ifd['sub']['InteropIFD'] = dict(off=e2['value'], entries=ie)
            if e['name'] == 'GPSInfoIFDPointer':
                so = e['value'] if isinstance(e['value'], int) else None
                if so and so < len(tiff):
                    ge, _ = read_ifd(tiff, so, E, GPSTAGS)
                    ifd['sub']['GPSIFD'] = dict(off=so, entries=ge)
        res['ifds'].append(ifd)
        off = nxt
        idx += 1
    res['E'] = E
    return res
