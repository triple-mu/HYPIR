"""100 张测试图统计量的分布图，标注验证集 3 LQ / 3 GT 的落点。

形式：横向 strip plot（每张图一个点，纵向抖动避免重叠）。
两个系列（重编码 / 相机原生）用已验证的分类色；验证集 LQ/GT 用中性墨色的
参考标记 + 直接标注，不占分类色槽。
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone")
from content_labels import L

D = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone"
S = {r["name"]: r for r in json.load(open(f"{D}/stats_all.json"))}
M = {r["name"]: r for r in json.load(open(f"{D}/maniqa_all.json"))}
H = json.load(open(f"{D}/hfr2.json"))
hfr = {int(k): v for k, v in H["test"].items()}
NATIVE = {1, 2, 3, 4, 5, 38, 64, 65, 66}

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b0"
C_RE, C_NAT = "#2a78d6", "#eb6834"

PANELS = [
    ("HFR 纹理条件锐度 (log)", lambda i: hfr[i],
     lambda n: H["val"][n.split("case")[1].split("_")[0] + n[-2:]], True),
    ("MANIQA (整图, 20x224 crop)", lambda i: M[f"test_case{i}"]["maniqa_full"],
     lambda n: M[n]["maniqa_full"], False),
    ("拉普拉斯方差 (log)", lambda i: S[f"test_case{i}"]["lap_var"],
     lambda n: S[n]["lap_var"], True),
    ("梯度能量 mean|∇|", lambda i: S[f"test_case{i}"]["grad_mean"],
     lambda n: S[n]["grad_mean"], False),
    ("亮度均值", lambda i: S[f"test_case{i}"]["lum_mean"],
     lambda n: S[n]["lum_mean"], False),
    ("饱和度均值 (HSV S)", lambda i: S[f"test_case{i}"]["sat_mean"],
     lambda n: S[n]["sat_mean"], False),
    ("暗部 0 值像素占比 %", lambda i: S[f"test_case{i}"]["frac_zero_any"] * 100,
     lambda n: S[n]["frac_zero_any"] * 100, False),
    ("亮部 255 截断占比 %", lambda i: S[f"test_case{i}"]["frac_255_any"] * 100,
     lambda n: S[n]["frac_255_any"] * 100, False),
    (">0.25 cyc/px 能量占比 (log)", lambda i: S[f"test_case{i}"]["psd_e_gt025"],
     lambda n: S[n]["psd_e_gt025"], True),
]

fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 2.05 * len(PANELS)))
fig.patch.set_facecolor(SURF)
rng = np.random.default_rng(0)

for ax, (title, f, fv, logx) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    re_i = [i for i in range(1, 101) if i not in NATIVE]
    na_i = sorted(NATIVE)
    for idx, col, lab, z in ((re_i, C_RE, "测试集·被重编码 (91)", 2),
                             (na_i, C_NAT, "测试集·相机原生 (9)", 3)):
        x = np.array([f(i) for i in idx], float)
        y = rng.uniform(-0.30, 0.30, len(x))
        ax.scatter(x, y, s=34, c=col, alpha=0.80, linewidths=0.8,
                   edgecolors=SURF, zorder=z)
    for i in na_i:  # 原生图直接标号（次级编码，不靠颜色识别）
        ax.annotate(str(i), (f(i), 0.42), ha="center", va="bottom",
                    fontsize=6.5, color=INK2, zorder=4)
    for k, (mk, nm) in enumerate((("lq", "LQ"), ("gt", "GT"))):
        xs = [fv(f"val_case{i}_{mk}") for i in (1, 2, 3)]
        yy = -0.62 if mk == "lq" else 0.62
        ax.scatter(xs, [yy] * 3, s=95, marker="v" if mk == "lq" else "^",
                   c=INK if mk == "gt" else "none", edgecolors=INK,
                   linewidths=1.6, zorder=5)
        for j, xv in enumerate(xs):
            ax.annotate(f"{nm}{j+1}", (xv, yy), xytext=(0, -11 if mk == "lq" else 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=INK, fontweight="bold", zorder=6)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    ax.grid(axis="x", color=MUTED, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)

handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=C_RE, mec=SURF, label="测试集·被重编码 (91)"),
           Line2D([], [], marker="o", ls="", ms=7, mfc=C_NAT, mec=SURF, label="测试集·相机原生 (9)"),
           Line2D([], [], marker="v", ls="", ms=9, mfc="none", mec=INK, mew=1.6, label="验证集 LQ (3)"),
           Line2D([], [], marker="^", ls="", ms=9, mfc=INK, mec=INK, label="验证集 GT (3)")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, 1.005), labelcolor=INK)
fig.suptitle("CSIG-2026 赛题二：100 张测试图统计分布 vs 验证集 LQ/GT 落点",
             fontsize=13, color=INK, y=1.028, x=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = "/home/ubuntu/workspace/contest/CSIG-2026/analysis_tone/dist_test_vs_val.png"
fig.savefig(out, dpi=145, facecolor=SURF, bbox_inches="tight")
print(out)
