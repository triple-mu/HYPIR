import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpegexif import *

TEST = '/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集'
CASES = [1,2,3,4,5,38,64,65,66]

def show_entries(ents, prefix=''):
    for e in ents:
        v = e['value']
        if isinstance(v, bytes):
            v = v[:64].hex() + ('...' if len(e['raw']) > 64 else '')
        s = str(v)
        if len(s) > 200: s = s[:200] + '...'
        print('%s  0x%04X %-28s %-9s cnt=%-6d nb=%-6d off=%-8d = %s' % (
            prefix, e['tag'], e['name'], e['typename'], e['count'], e['nbytes'], e['data_off'], s))

for c in CASES:
    p = os.path.join(TEST, 'case%d.jpg' % c)
    data = open(p, 'rb').read()
    segs = scan_jpeg(data)
    print('=' * 100)
    print('### case%d.jpg  size=%d' % (c, len(data)))
    print('--- 段布局 (offset) ---')
    for name, off, L, pl in segs:
        ident = pl.split(b'\x00')[0][:34].decode('latin1') if pl else ''
        print('  @%-9d %-6s len=%-7d %s' % (off, name, L, ident))
    # EXIF
    tiff = None
    for name, off, L, pl in segs:
        if name == 'APP1' and pl[:6] == b'Exif\x00\x00':
            tiff = pl[6:]
            break
    if tiff is None:
        continue
    t = parse_tiff(tiff)
    print('--- TIFF header: byteorder=%s magic=%d ifd0_off=%d tiff_len=%d ---' % (
        t['byteorder'], t['magic'], t['ifd0_off'], t['tiff_len']))
    for ifd in t['ifds']:
        print('  [%s] off=%d nentries=%d next=%d' % (ifd['name'], ifd['off'], len(ifd['entries']), ifd['next']))
        show_entries(ifd['entries'], '   ')
        for sname, sub in ifd['sub'].items():
            print('    <%s> off=%d nentries=%d' % (sname, sub['off'], len(sub['entries'])))
            show_entries(sub['entries'], '     ')
    # 缩略图
    for ifd in t['ifds']:
        jo = jl = None
        for e in ifd['entries']:
            if e['name'] == 'JPEGInterchangeFormat': jo = e['value']
            if e['name'] == 'JPEGInterchangeFormatLength': jl = e['value']
        if jo and jl:
            thumb = tiff[jo:jo+jl]
            out = '/home/ubuntu/workspace/contest/CSIG-2026/Moebius/forensics/thumb_case%d_%s.jpg' % (c, ifd['name'])
            open(out, 'wb').write(thumb)
            ts = scan_jpeg(thumb)
            sof = [parse_sof(pl) for n,o,L,pl in ts if n.startswith('SOF')]
            print('  >>> 缩略图 %s: off=%d len=%d 实际字节=%d SOF=%s -> %s' % (
                ifd['name'], jo, jl, len(thumb), sof, out))
