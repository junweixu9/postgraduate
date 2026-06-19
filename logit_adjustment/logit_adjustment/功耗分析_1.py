# -*- coding: utf-8 -*-
"""
紧凑版：AES 操作区间内多条功耗轨迹与 S 盒区分性示意图

核心表达：
    1. 横轴只保留 AES 操作区间；
    2. 多条轨迹对应不同 S 盒中间值 / HW 类别；
    3. 在第一轮 S 盒区域，不同 HW 类别轨迹分离明显；
    4. 在 ShiftRows、MixColumns 和后续轮区域，类别差异较弱；
    5. 尽量减少上下左右空白，提高图面空间利用率。

依赖：
    pip install numpy matplotlib mksci-font
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

try:
    from mksci_font import config_font
except ImportError as e:
    raise ImportError("未检测到 mksci-font，请先执行：pip install mksci-font") from e


# ============================================================
# 1. 字体与输出设置
# ============================================================

FONTSIZE = 13

config_font({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE - 1.4,
    "figure.titlesize": FONTSIZE,
})

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = Path("./sca_trace_fig_compact_tight")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "SCA_multi_traces_sbox_compact_tight.png"
OUT_PDF = OUT_DIR / "SCA_multi_traces_sbox_compact_tight.pdf"
OUT_SVG = OUT_DIR / "SCA_multi_traces_sbox_compact_tight.svg"


# ============================================================
# 2. 基础函数
# ============================================================

np.random.seed(2026)


def gaussian(x, mu, sigma, amp):
    """高斯峰，用于模拟局部操作引起的功耗变化。"""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def smooth_noise(n, scale=0.05, k=11):
    """平滑噪声，使轨迹更接近连续功耗曲线。"""
    z = np.random.normal(0, scale, n)
    kernel = np.ones(k) / k
    return np.convolve(z, kernel, mode="same")


def clean_axis(ax):
    """去掉上边框和右边框，使图更适合论文排版。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 3. 构造紧凑 AES 操作时间轴
# ============================================================

# 只保留 AES 操作区间，横轴压缩为 0~700
x = np.linspace(0, 700, 701)

# 第一轮 S 盒兴趣区
roi_start, roi_end = 185, 270
roi_center = (roi_start + roi_end) / 2

# AES 操作阶段
segments = [
    (0, 105, "轮密钥加"),
    (105, 270, "第一轮 S 盒"),
    (270, 365, "ShiftRows"),
    (365, 520, "MixColumns"),
]

# 公共功耗形状：所有轨迹共享
base = (
    0.07 * np.sin(2 * np.pi * x / 115)
    + 0.055 * np.sin(2 * np.pi * x / 34)
)

# 非 S 盒区域的公共峰：不同 HW 类别基本重叠
common_peaks = [
    (35, 9, 0.55),
    (88, 13, 0.78),
    (150, 14, 0.82),
    (300, 15, 0.62),
    (380, 18, 0.78),
    (480, 18, 0.70),
    (565, 20, 0.60),
    (635, 18, 0.58),
]

for mu, sig, amp in common_peaks:
    base += gaussian(x, mu, sig, amp)

# S 盒区域的公共功耗
sbox_common = gaussian(x, roi_center, 18, 0.72)

# S 盒区域的类别相关泄漏形状
# 只有这一部分随 HW 类别明显变化。
sbox_leak_shape = (
    gaussian(x, roi_center - 5, 12, 1.00)
    + 0.42 * gaussian(x, roi_center + 17, 9, 1.00)
)


# ============================================================
# 4. 生成多条轨迹
# ============================================================

hw_classes = np.arange(9)      # HW0 ~ HW8
traces_per_hw = 5              # 每个 HW 类别生成多条轨迹

colors = plt.cm.tab10(np.linspace(0, 1, len(hw_classes)))

all_traces = []
all_labels = []

for hw in hw_classes:
    hw_norm = hw / 8.0

    for _ in range(traces_per_hw):
        noise = smooth_noise(len(x), scale=0.105, k=9)
        scale = 1.0 + np.random.normal(0, 0.015)

        # 所有轨迹共享 AES 操作整体形状
        trace = scale * (base + sbox_common)

        # 关键：只在 S 盒区域加入明显的类别相关差异
        class_amp = 0.10 + 0.60 * hw_norm
        trace += class_amp * sbox_leak_shape

        # 轻微采集抖动
        jitter = np.random.normal(0, 1.3)
        trace += gaussian(x, roi_center + jitter, 12, 0.06 * hw_norm)

        trace += noise

        all_traces.append(trace)
        all_labels.append(hw)

all_traces = np.array(all_traces)
all_labels = np.array(all_labels)


# ============================================================
# 5. 自动收紧 y 轴范围，减少上下空白
# ============================================================

q_low = np.percentile(all_traces, 0.5)
q_high = np.percentile(all_traces, 99.7)
y_span = q_high - q_low

ylim_low = q_low - 0.055 * y_span

# 顶部阶段条靠近曲线，避免上方留太多空白
y_bar = q_high + 0.080 * y_span
bar_h = 0.070 * y_span

ylim_high = y_bar + bar_h + 0.035 * y_span


# ============================================================
# 6. 绘图
# ============================================================

fig, ax = plt.subplots(figsize=(9.2, 2.05), dpi=200)

roi_color = "#E53935"
shade_color = "#FDECEC"
aes_shade = "#F5F5F5"
segment_color = "#ECEFF1"
grid_color = "#E0E0E0"
text_color = "#263238"

# 背景与 ROI
ax.axvspan(0, 700, color=aes_shade, zorder=0)
ax.axvspan(roi_start, roi_end, color=shade_color, zorder=1)

# 多条轨迹：细线表示单条轨迹，粗线表示同一 HW 类别的平均轨迹
for hw, color in zip(hw_classes, colors):
    idx = np.where(all_labels == hw)[0]

    for i in idx:
        ax.plot(
            x,
            all_traces[i],
            color=color,
            lw=0.50,
            alpha=0.22,
            zorder=2,
        )

    mean_trace = all_traces[idx].mean(axis=0)
    ax.plot(
        x,
        mean_trace,
        color=color,
        lw=1.05,
        alpha=0.95,
        zorder=3,
    )

# 顶部 AES 阶段条
for a, b, label in segments:
    ax.add_patch(
        Rectangle(
            (a, y_bar),
            b - a,
            bar_h,
            fc=segment_color,
            ec="white",
            lw=0.5,
            zorder=5,
        )
    )
    ax.text(
        (a + b) / 2,
        y_bar + bar_h / 2,
        label,
        ha="center",
        va="center",
        fontsize=FONTSIZE - 0.5,
        zorder=6,
    )

# ROI 红框：缩小面积，避免覆盖过多绘图区
roi_box_x_margin = 8
roi_box_y_bottom_margin = 0.080 * y_span
roi_box_y_top_margin = 0.240 * y_span

roi_box_left = roi_start + roi_box_x_margin
roi_box_width = (roi_end - roi_start) - 2 * roi_box_x_margin

roi_box_bottom = ylim_low + roi_box_y_bottom_margin
roi_box_top = y_bar + bar_h - roi_box_y_top_margin
roi_box_height = roi_box_top - roi_box_bottom

ax.add_patch(
    Rectangle(
        (roi_box_left, roi_box_bottom+0.2),
        roi_box_width,
        roi_box_height,
        fill=False,
        ec=roi_color,
        lw=1.5,
        zorder=7,
    )
)

# 核心标注：放在曲线附近，避免占用额外空白
ax.annotate(
    "不同 S 盒中间值 / HW 类别\n在该区域轨迹分离明显",
    xy=(roi_center, q_high - 0.030 * y_span),
    xytext=(roi_end + 35, q_high - 0.030 * y_span-0.2),
    arrowprops=dict(arrowstyle="->", color=roi_color, lw=1.2),
    color=roi_color,
    ha="left",
    va="center",
    fontsize=FONTSIZE,
    zorder=8,
)

# 辅助标注：尽量靠近曲线
other_x = 445
other_y = all_traces[:, np.argmin(np.abs(x - other_x))].mean()


# 紧凑图例：放在图内右侧，避免增加额外空白
legend_handles = [
    Line2D([0], [0], color=colors[i], lw=1.3, label=f"HW{i}")
    for i in hw_classes
]

legend = ax.legend(
    handles=legend_handles,
    loc="center right",
    bbox_to_anchor=(0.995, 0.72),
    ncol=3,
    frameon=True,
    fontsize=FONTSIZE - 1.35,
    handlelength=1.45,
    columnspacing=0.52,
    borderpad=0.22,
    labelspacing=0.22,
)

legend.get_frame().set_alpha(0.88)
legend.get_frame().set_linewidth(0.5)


# ============================================================
# 7. 坐标轴与版式
# ============================================================

ax.set_xlim(0, 700)
ax.set_ylim(ylim_low, ylim_high)

ax.set_xlabel("AES 操作采样点", labelpad=2)
ax.set_ylabel("归一化功耗", labelpad=3)

ax.set_xticks(np.arange(0, 701, 100))
ax.tick_params(axis="both", labelsize=FONTSIZE, pad=2)

ax.grid(axis="y", color=grid_color, lw=0.5, alpha=0.6)

clean_axis(ax)

# 进一步压缩画布内部边距
fig.subplots_adjust(
    left=0.070,
    right=0.995,
    top=0.985,
    bottom=0.175,
)

# 保存时裁剪外部空白
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.015)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.015)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.015)

plt.show()

print(f"PNG 已保存至：{OUT_PNG}")
print(f"PDF 已保存至：{OUT_PDF}")
print(f"SVG 已保存至：{OUT_SVG}")