"""手工 JPEG 段解析器：不依赖 PIL，直接扫 marker。"""
import os, sys, json, struct, hashlib, glob

MARKER_NAMES = {
    0xC0: 'SOF0', 0xC1: 'SOF1', 0xC2: 'SOF2', 0xC3: 'SOF3',
    0xC5: 'SOF5', 0xC6: 'SOF6', 0xC7: 'SOF7', 0xC9: 'SOF9',
    0xCA: 'SOF10', 0xCB: 'SOF11', 0xCD: 'SOF13', 0xCE: 'SOF14', 0xCF: 'SOF15',
    0xC4: 'DHT', 0xCC: 'DAC', 0xD8: 'SOI', 0xD9: 'EOI', 0xDA: 'SOS',
    0xDB: 'DQT', 0xDC: 'DNL', 0xDD: 'DRI', 0xDE: 'DHP', 0xDF: 'EXP',
    0xFE: 'COM',
}
for i in range(16):
    MARKER_NAMES[0xE0 + i] = 'APP%d' % i

ZIGZAG = [
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63]


def parse(path):
    data = open(path, 'rb').read()
    r = {'path': path, 'size': len(data),
         'md5': hashlib.md5(data).hexdigest(),
         'magic': data[:4].hex(),
         'dqt': {}, 'apps': [], 'sof': None, 'dht': [], 'dri': None,
         'scans': [], 'com': [], 'markers': [], 'trailing': 0}
    if data[:2] != b'\xff\xd8':
        r['error'] = 'not JPEG SOI'
        return r
    i = 2
    n = len(data)
    while i < n:
        # 找 marker
        if data[i] != 0xFF:
            i += 1
            continue
        j = i
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break
        m = data[j]
        i = j + 1
        if m == 0x00 or (0xD0 <= m <= 0xD7):
            continue
        if m == 0xD9:  # EOI
            r['markers'].append('EOI')
            r['trailing'] = n - i
            break
        if m == 0x01:
            continue
        if i + 2 > n:
            break
        seglen = struct.unpack('>H', data[i:i+2])[0]
        seg = data[i+2:i+seglen]
        name = MARKER_NAMES.get(m, 'M%02X' % m)
        r['markers'].append(name)
        if m == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq = seg[p] >> 4
                tq = seg[p] & 15
                p += 1
                if pq == 0:
                    tbl = list(seg[p:p+64]); p += 64
                else:
                    tbl = list(struct.unpack('>64H', seg[p:p+128])); p += 128
                # 存 zigzag 顺序（文件中的顺序）
                r['dqt'][tq] = {'prec': pq, 'zz': tbl}
        elif m == 0xC4:  # DHT
            p = 0
            while p + 17 <= len(seg):
                tc = seg[p] >> 4; th = seg[p] & 15
                counts = list(seg[p+1:p+17])
                nsym = sum(counts)
                r['dht'].append({'class': tc, 'id': th, 'nsym': nsym})
                p += 17 + nsym
        elif m == 0xDD:
            r['dri'] = struct.unpack('>H', seg[:2])[0]
        elif 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            prec = seg[0]
            h, w = struct.unpack('>HH', seg[1:5])
            nc = seg[5]
            comps = []
            for c in range(nc):
                cid = seg[6+c*3]
                hv = seg[7+c*3]
                tq = seg[8+c*3]
                comps.append({'id': cid, 'h': hv >> 4, 'v': hv & 15, 'tq': tq})
            r['sof'] = {'marker': name, 'prec': prec, 'w': w, 'h': h,
                        'comps': comps,
                        'progressive': name in ('SOF2', 'SOF6', 'SOF10', 'SOF14')}
        elif 0xE0 <= m <= 0xEF:
            r['apps'].append({'marker': name, 'len': seglen,
                              'head16': seg[:16].hex(),
                              'head_ascii': ''.join(chr(b) if 32 <= b < 127 else '.' for b in seg[:16]),
                              'off': i - 2})
        elif m == 0xFE:
            r['com'].append(seg[:64].decode('latin1'))
        elif m == 0xDA:
            ns = seg[0]
            sc = {'ncomp': ns, 'comps': []}
            for c in range(ns):
                sc['comps'].append({'id': seg[1+c*2], 'dc': seg[2+c*2] >> 4, 'ac': seg[2+c*2] & 15})
            base = 1 + ns*2
            sc['Ss'], sc['Se'] = seg[base], seg[base+1]
            sc['Ah'], sc['Al'] = seg[base+2] >> 4, seg[base+2] & 15
            r['scans'].append(sc)
            # 跳过熵编码数据
            i = i + seglen
            while i < n - 1:
                if data[i] == 0xFF and data[i+1] != 0x00 and not (0xD0 <= data[i+1] <= 0xD7):
                    break
                i += 1
            continue
        i += seglen
    r['nscans'] = len(r['scans'])
    return r


def dqt_natural(zz):
    """zigzag 顺序 -> 自然 8x8 行主序"""
    out = [0]*64
    for k in range(64):
        out[ZIGZAG[k]] = zz[k]
    return out


if __name__ == '__main__':
    files = []
    for pat in sys.argv[1:]:
        files.extend(sorted(glob.glob(pat)))
    res = [parse(f) for f in files]
    json.dump(res, open('/tmp/jpegparse.json', 'w'))
    print(len(res))
