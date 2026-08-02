import re
# 中国品牌：Make 字段 + 常见 Model 前缀/代号
CN_MAKE = re.compile(r"^(huawei|honor|xiaomi|redmi|poco|oppo|vivo|oneplus|realme|meizu|zte|nubia|lenovo|"
                     r"coolpad|gionee|smartisan|hisense|tcl|doogee|umidigi|blackview|cubot|ulefone|"
                     r"letv|leeco|360|qiku|nokia \(hmd\)|infinix|tecno|itel|dji|hmd)", re.I)
CN_MODEL = re.compile(r"^(cph\d|pd[a-z0-9]{4}|v\d{4}[a-z]|rmx\d|mi \d|mi note|mi max|mi pad|mi a\d|"
                      r"redmi|poco|m20\d\d[a-z0-9]|2[0-9]{5}[a-z0-9]+|23\d{3}[a-z0-9]+|22\d{3}[a-z0-9]+|"
                      r"[a-z]{3}-[atwlnc][lx]\d{2}|[a-z]{3}-[atwln]\d{2}|nth-|els-|vog-|lya-|ana-|ela-|"
                      r"tas-|noh-|jad-|alp-|clt-|eml-|war-|jsn-|pot-|mar-|lld-|dub-|jny-|brq-|nam-|"
                      r"kb2\d\d\d|le2\d\d\d|in2\d\d\d|hd1\d\d\d|gm1\d\d\d|a000\d|be2\d\d\d)", re.I)
# 主流手机品牌（含非中国）：用来算「手机 vs 相机」
PHONE_MAKE = re.compile(r"^(apple|samsung|google|motorola|lg electronics|lge|nokia|htc|sony ericsson|"
                        r"asus|essential|fairphone|sharp|kyocera|blackberry)", re.I)

def is_cn_phone(make, model):
    m = (make or "").strip(); mo = (model or "").strip()
    if CN_MAKE.match(m): return True
    if m.lower() in ("", "unknown") and CN_MODEL.match(mo): return True
    if CN_MODEL.match(mo) and not PHONE_MAKE.match(m) and m.lower() not in ("nikon","canon","sony","fujifilm","olympus imaging corp.","panasonic"): return True
    return False

def is_phone(make, model):
    m = (make or "").strip()
    if PHONE_MAKE.match(m): return True
    return is_cn_phone(make, model)
