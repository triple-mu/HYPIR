import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegexif import *

TEST = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集'
CASES = [1,2,3,4,5,38,64,65,66]

MPTAGS = {0xB000:'MPFVersion',0xB001:'NumberOfImages',0xB002:'MPEntry',
          0xB003:'ImageUIDList',0xB004:'TotalFrames',0xB101:'MPIndividualNum',
          0xB201:'PanOrientation',0xB202:'PanOverlapH',0xB203:'PanOverlapV',
          0xB204:'BaseViewpointNum',0xB205:'ConvergenceAngle',0xB206:'BaselineLength',
          0xB207:'VerticalDivergence',0xB208:'AxisDistanceX',0xB209:'AxisDistanceY',
          0xB20A:'AxisDistanceZ',0xB20B:'YawAngle',0xB20C:'PitchAngle',0xB20D:'RollAngle'}

MPTYPE = {0x030000:'Baseline MP Primary Image', 0x010001:'Large Thumbnail(VGA)',
          0x010002:'Large Thumbnail(full HD)', 0x020001:'Multi-frame Panorama',
          0x020002:'Multi-frame Disparity', 0x020003:'Multi-angle', 0x000000:'Undefined'}

print('#' * 110)
print('# 1. MPF (MPO) 结构 + 多图像布局')
print('#' * 110)
for c in CASES:
    p = os.path.join(TEST, 'case%d.jpg' % c)
    data = open(p, 'rb').read()
    # find MPF APP2
    segs = scan_jpeg(data)
    mpf_payload = None; mpf_off = None
    for name, off, L, pl in segs:
        if name == 'APP2' and pl[:4] == b'MPF\x00':
            mpf_payload = pl[4:]
            mpf_off = off + 4 + 4   # marker(2)+len(2)+'MPF\0'(4) -> TIFF base
    print('--- case%d (%d bytes) ---' % (c, len(data)))
    if mpf_payload:
        t = parse_tiff(mpf_payload)
        E = t['E']
        ents, _ = read_ifd(mpf_payload, t['ifd0_off'], E, MPTAGS)
        for e in ents:
            v = e['value']
            if e['name'] == 'MPEntry':
                raw = e['raw']
                n = len(raw)//16
                print('   MPEntry (%d 项):' % n)
                for k in range(n):
                    attr, sz, off_, d1, d2 = struct.unpack(E+'IIIHH', raw[k*16:k*16+16])
                    typ = attr & 0xFFFFFF
                    fmt = (attr >> 24) & 0x7
                    rep = (attr >> 29) & 1
                    abs_off = off_ + mpf_off if off_ else 0
                    print('     #%d type=0x%06X(%s) fmt=%d 代表图=%d size=%d rel_off=%d abs_off=%d' % (
                        k, typ, MPTYPE.get(typ, '?'), fmt, rep, sz, off_, abs_off))
            else:
                if isinstance(v, bytes): v = v.hex()
                print('   %-18s %s' % (e['name'], v))
    # 全文件扫 SOI/EOI
    sois = []
    i = 0
    while True:
        i = data.find(b'\xff\xd8\xff', i)
        if i < 0: break
        sois.append(i); i += 2
    print('   全文件 FFD8FF 出现位置: %s' % sois[:12])
    print('   文件尾 4 字节: %s' % data[-4:].hex())

print()
print('#' * 110)
print('# 2. TIFF 块字节覆盖分析（判断 EXIF 是重写还是就地打补丁）')
print('#' * 110)
for c in CASES:
    p = os.path.join(TEST, 'case%d.jpg' % c)
    data = open(p, 'rb').read()
    segs = scan_jpeg(data)
    tiff = None
    for name, off, L, pl in segs:
        if name == 'APP1' and pl[:6] == b'Exif\x00\x00':
            tiff = pl[6:]; break
    t = parse_tiff(tiff)
    cov = bytearray(len(tiff))
    cov[0:8] = b'\x01' * 8   # TIFF header
    def mark(a, b):
        for i in range(max(0,a), min(len(tiff), b)): cov[i] = 1
    def do_ifd(ents, off):
        mark(off, off + 2 + 12*len(ents) + 4)
        for e in ents:
            if not e['inline']:
                mark(e['data_off'], e['data_off'] + e['nbytes'])
    for ifd in t['ifds']:
        do_ifd(ifd['entries'], ifd['off'])
        for sn, sub in ifd['sub'].items():
            do_ifd(sub['entries'], sub['off'])
    gaps = []
    i = 0
    while i < len(cov):
        if cov[i] == 0:
            j = i
            while j < len(cov) and cov[j] == 0: j += 1
            gaps.append((i, j-i, bytes(tiff[i:j])))
            i = j
        else: i += 1
    print('case%-4d TIFF_len=%d  未覆盖字节=%d  gap数=%d' % (c, len(tiff), sum(g[1] for g in gaps), len(gaps)))
    for a, l, b in gaps:
        printable = ''.join(chr(x) if 32 <= x < 127 else '.' for x in b)
        print('    gap @%-5d len=%-4d hex=%s  ascii=%r' % (a, l, b[:48].hex(), printable[:48]))
