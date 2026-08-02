import json, sys, time, urllib.parse
import requests

UA = "CSIGDataBot/0.0 (https://github.com/csig2026-data; sonlin@nvidia.com)"
S = requests.Session()
S.headers.update({"User-Agent": UA})
API = "https://commons.wikimedia.org/w/api.php"

def api(params, host=API):
    p = dict(params); p.setdefault("format", "json"); p.setdefault("formatversion", 2)
    for att in range(4):
        try:
            r = S.get(host, params=p, timeout=60)
            if r.status_code == 200:
                return r.json()
            print("HTTP", r.status_code, r.text[:200], file=sys.stderr)
        except Exception as e:
            print("ERR", e, file=sys.stderr)
        time.sleep(2 * (att + 1))
    return None

def hits(srsearch, ns=6):
    d = api({"action": "query", "list": "search", "srsearch": srsearch,
             "srnamespace": ns, "srlimit": 1, "srinfo": "totalhits", "srprop": ""})
    if d is None: return None
    if "error" in d: return "ERR:" + d["error"].get("code", "?") + ":" + d["error"].get("info", "")[:120]
    return d["query"]["searchinfo"]["totalhits"]
