"""从 PD12M 元数据里筛出「手机型 12MP + 域内内容 + 原生锐利」的图并下载。

用法:
  python3 pd12m_select.py index   # 只读 125 个元数据分片(约 2.3GB 传输), 产出 urls.txt
  python3 pd12m_select.py fetch    # 按 urls.txt 下载并做锐度筛选, 落盘 out/
"""
import io, os, re, sys, json, random
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pyarrow.parquet as pq

Image.MAX_IMAGE_PIXELS = None
BASE = 'https://huggingface.co/datasets/Spawning/PD12M/resolve/main/metadata/pd12m.%03d.parquet'
OUT = 'out'
HF_THRESH = 0.005          # 高频能量比阈值, 见报告 §3 标定

CONTENT = re.compile(r'\b(building|skyscraper|facade|high-rise|architecture|architectural|'
                     r'office building|cathedral|temple|pagoda|tower|skyline|cityscape|downtown|'
                     r'flower|blossom|petal|leaf|leaves|foliage|plant|bloom|floral|'
                     r'sign|signage|storefront|billboard|sculpture|statue|carving|monument|'
                     r'street|sidewalk|pavement|alley)\b', re.I)
REJECT = re.compile(r'\b(woman|man|person|people|portrait|girl|boy|child|face|smiling|posing|'
                    r'food|dish|meal|cake|pizza|illustration|painting|drawing|logo|icon|map|'
                    r'chart|diagram)\b', re.I)


class HTTPFile(io.RawIOBase):
    """pyarrow 用的最小可 seek HTTP 文件, 让 parquet 只拉需要的列。"""
    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True)
        self.url, self.size, self.pos = r.url, int(r.headers['Content-Length']), 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        end = min(self.pos + n, self.size) - 1
        if end < self.pos: return b''
        d = self.s.get(self.url, headers={'Range': f'bytes={self.pos}-{end}'}).content
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def index():
    def one(i):
        for _ in range(3):
            try:
                t = pq.ParquetFile(HTTPFile(BASE % i)).read(
                    columns=['url', 'caption', 'width', 'height'])
                break
            except Exception:
                t = None
        if t is None: return []
        u = t.column('url').to_pylist(); c = t.column('caption').to_pylist()
        w = t.column('width').to_pylist(); h = t.column('height').to_pylist()
        keep = []
        for k in range(len(u)):
            le, se = max(w[k], h[k]), min(w[k], h[k])
            if not (3700 <= le <= 4400 and abs(le / se - 4 / 3) < 0.012): continue
            cap = c[k] or ''
            if REJECT.search(cap) or not CONTENT.search(cap): continue
            keep.append(u[k])
        return keep

    urls = []
    with ThreadPoolExecutor(12) as ex:
        for k, r in enumerate(ex.map(one, range(125))):
            urls += r
            if k % 20 == 0: print(f'{k}/125  累计 {len(urls)}', flush=True)
    open('urls.txt', 'w').write('\n'.join(urls))
    print(f'共 {len(urls)} 条 -> urls.txt (预计 {len(urls) * 5 / 1000:.0f} GB)')


def hf_max(im, s=512, grid=4):
    """取 4x4 网格中最锐 patch 的高频能量比; 用最大值以避开虚化背景导致的误杀。"""
    W, H = im.size
    g = np.asarray(im.convert('L')).astype(np.float64)
    win = np.hanning(s)[:, None] * np.hanning(s)[None, :]
    best = 0.0
    for iy in range(grid):
        for ix in range(grid):
            y, x = int((H - s) * (iy + .5) / grid), int((W - s) * (ix + .5) / grid)
            c = g[y:y + s, x:x + s]
            if c.shape != (s, s) or c.std() <= 8: continue
            F = np.fft.fftshift(np.abs(np.fft.fft2((c - c.mean()) * win)) ** 2)
            cy, cx = s // 2, s // 2
            yy, xx = np.indices(F.shape)
            r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
            idx = np.clip((r * 64).astype(int), 0, 63)
            p = np.bincount(idx.ravel(), F.ravel(), minlength=64) / np.bincount(idx.ravel(), minlength=64)
            p /= p[1]
            best = max(best, p[32:].sum() / p[1:].sum())
    return best


def fetch():
    os.makedirs(OUT, exist_ok=True)
    urls = [u.strip() for u in open('urls.txt') if u.strip()]
    random.seed(0); random.shuffle(urls)
    stat = {'ok': 0, 'soft': 0, 'err': 0}

    def one(u):
        name = u.rsplit('/', 1)[-1]
        p = os.path.join(OUT, name)
        if os.path.exists(p): return 'ok'
        try:
            b = requests.get(u, timeout=60, headers={'User-Agent': 'Mozilla/5.0'}).content
            im = Image.open(io.BytesIO(b)); im.load()
        except Exception:
            return 'err'
        if hf_max(im) < HF_THRESH: return 'soft'
        open(p, 'wb').write(b)
        return 'ok'

    with ThreadPoolExecutor(16) as ex:
        for k, r in enumerate(ex.map(one, urls)):
            stat[r] += 1
            if k % 200 == 0: print(k, stat, flush=True)
    print(stat)


if __name__ == '__main__':
    {'index': index, 'fetch': fetch}[sys.argv[1]]()
