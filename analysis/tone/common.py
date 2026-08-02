import numpy as np, cv2, os
VAL = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/验证集"
TEST = "/home/ubuntu/workspace/contest/CSIG-2026/赛题二/测试集"
PAIRS = {
    "case1": (f"{VAL}/case1_lq.jpg", f"{VAL}/case1_gt.png"),
    "case2": (f"{VAL}/case2_lq.jpg", f"{VAL}/case2_gt.png"),
    "case3": (f"{VAL}/case3_lq.jpg", f"{VAL}/case3_gt.jpg"),
}
def imread(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert a is not None, p
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # uint8 RGB
