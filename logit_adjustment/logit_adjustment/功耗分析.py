# -*- coding: utf-8 -*-
"""
SCA 功耗曲线、兴趣区截取与轨迹对齐示意图

字体要求：
    中文：宋体
    英文：Times New Roman
    图片字体：五号，即 10.5 pt
    每个子图的图题：放在对应子图的中央下方

依赖安装：
    pip install numpy matplotlib mksci-font

运行：
    python draw_sca_trace_roi_alignment_mksci_caption_bottom.py

输出：
    ./sca_trace_fig_mksci_caption_bottom/SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.png
    ./sca_trace_fig_mksci_caption_bottom/SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.pdf
    ./sca_trace_fig_mksci_caption_bottom/SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.svg

说明：
    本图是功耗分析流程示意图，不是某一公开数据集的原始实测曲线。
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 核心：使用 mksci-font 配置“中文宋体、英文 Times New Roman”
try:
    from mksci_font import config_font
except ImportError as e:
    raise ImportError(
        "未检测到 mksci-font，请先在终端执行：pip install mksci-font"
    ) from e


# ============================================================
# 1. 字体与输出设置
# ============================================================

# 五号字体约为 10.5 pt
FONTSIZE = 10.5

# mksci-font 核心配置：
# 中文宋体，英文 Times New Roman，并统一字号为五号
config_font({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "figure.titlesize": FONTSIZE,
})

# 负号正常显示
matplotlib.rcParams["axes.unicode_minus"] = False

# PDF/SVG 尽量保留可编辑文字
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# 数学公式尽量接近 Times New Roman 风格
matplotlib.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = Path("./sca_trace_fig_mksci_caption_bottom_fix_label")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.png"
OUT_PDF = OUT_DIR / "SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.pdf"
OUT_SVG = OUT_DIR / "SCA_power_trace_ROI_alignment_schematic_caption_bottom_fix_label.svg"


# ============================================================
# 2. 生成示意性功耗曲线
# ============================================================

np.random.seed(2026)


def gaussian(x, mu, sigma, amp):
    """高斯峰，用于模拟密码操作引起的局部功耗变化。"""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def smooth_noise(n, scale=0.05, k=21):
    """平滑噪声，使曲线更接近功耗轨迹的连续波动。"""
    z = np.random.normal(0, scale, n)
    kernel = np.ones(k) / k
    return np.convolve(z, kernel, mode="same")


def clean_axis(ax):
    """去掉上边框和右边框，使图更适合论文排版。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bottom_caption(ax, text, y=-0.34):
    """
    在单个子图的正下方居中添加子图题。

    参数：
        ax   : 当前子图坐标轴
        text : 子图题文本
        y    : 子图题相对于坐标轴的位置，负数表示在坐标轴下方
    """
    ax.text(
        0.5, y, text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


# 完整轨迹采样点
x = np.linspace(0, 1000, 1001)

# 基础波动 + 平滑噪声
trace = (
    0.08 * np.sin(2 * np.pi * x / 140)
    + 0.08 * np.sin(2 * np.pi * x / 37)
    + smooth_noise(len(x), scale=0.18, k=13)
)

# AES 加密过程中不同局部操作的示意峰值
# 注意：这里是示意曲线，不对应真实设备的固定采样点。
operation_peaks = [
    (120, 10, 0.60),   # 触发/轮密钥加附近
    (170, 13, 0.80),
    (230, 16, 0.90),
    (305, 13, 1.35),   # 目标 S 盒泄漏区域：重点突出
    (370, 18, 0.80),
    (455, 22, 0.75),
    (550, 18, 0.90),
    (645, 22, 0.70),
    (735, 24, 0.75),
    (835, 18, 0.65),
    (910, 16, 0.55),
]

for mu, sig, amp in operation_peaks:
    trace += gaussian(x, mu, sig, amp)

# 目标 S 盒泄漏窗口，即 ROI / 攻击窗口
roi_start, roi_end = 265, 350
roi_center = (roi_start + roi_end) / 2

# 用于子图 (b) 的局部显示范围
roi_show_mask = (x >= 235) & (x <= 385)
x_roi = x[roi_show_mask]
trace_roi = trace[roi_show_mask]


# ============================================================
# 3. 生成对齐前后的多条局部轨迹
# ============================================================

local_x = np.linspace(-60, 60, 240)

# 对齐前：同一泄漏峰值在不同轨迹中有时间偏移
shifts = [-17, -9, 0, 11, 20]
amps = [1.00, 0.92, 1.05, 0.96, 1.02]

traces_before = []
traces_after = []

for s, a in zip(shifts, amps):
    noise = smooth_noise(len(local_x), scale=0.09, k=7)

    # 对齐前：峰值中心位于 s
    before = (
        gaussian(local_x, s, 8.5, a)
        + 0.25 * gaussian(local_x, s + 19, 10, 0.5)
        + noise
    )

    # 对齐后：主峰被校准到 0 附近
    after = (
        gaussian(local_x, 0, 8.5, a)
        + 0.25 * gaussian(local_x, 19, 10, 0.5)
        + 0.9 * noise
    )

    traces_before.append(before)
    traces_after.append(after)


# ============================================================
# 4. 绘图
# ============================================================

fig = plt.figure(figsize=(12.0, 10.0), dpi=200)

# 增大 hspace，给每个子图下方的子图题留出空间
gs = fig.add_gridspec(
    3, 1,
    height_ratios=[1.25, 1.00, 1.15],
    hspace=1.08
)

# 配色：论文中比较稳妥的低饱和色
line_color = "#263238"
roi_color = "#E53935"
shade_color = "#FDECEC"
aes_shade = "#F5F5F5"
segment_color = "#ECEFF1"
grid_color = "#E0E0E0"
multi_colors = ["#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00", "#00838F"]


# --------------------------
# (a) 完整功耗曲线
# --------------------------
ax1 = fig.add_subplot(gs[0])
ax1.plot(x, trace, color=line_color, lw=1.20)

# AES 完整加密过程背景
ax1.axvspan(80, 940, color=aes_shade, zorder=0)

# 目标 ROI 阴影与红框
ax1.axvspan(roi_start, roi_end, color=shade_color, zorder=0)
ax1.add_patch(
    Rectangle(
        (roi_start, trace.min() - 0.16),
        roi_end - roi_start,
        trace.max() - trace.min() + 0.32,
        fill=False,
        ec=roi_color,
        lw=1.6,
    )
)

# AES 操作阶段示意条
segments = [
    (80, 185, "触发与轮密钥加"),
    (185, 350, "第一轮 S 盒"),
    (350, 465, "ShiftRows"),
    (465, 620, "MixColumns"),
    (620, 940, "后续轮操作"),
]

y_top = trace.max() + 0.46
for a, b, label in segments:
    ax1.add_patch(
        Rectangle(
            (a, y_top - 0.18),
            b - a,
            0.16,
            fc=segment_color,
            ec="white",
            lw=0.6,
        )
    )
    ax1.text(
        (a + b) / 2,
        y_top - 0.10,
        label,
        ha="center",
        va="center",
        fontsize=FONTSIZE,
    )

ax1.annotate(
    "目标操作区域\nSbox($p\\oplus k$)",
    xy=(roi_center, trace[np.argmin(np.abs(x - roi_center))] + 0.22),
    xytext=(405, y_top - 0.55),
    arrowprops=dict(arrowstyle="->", color=roi_color, lw=1.3),
    color=roi_color,
    ha="left",
    va="center",
    fontsize=FONTSIZE,
)

ax1.set_xlim(0, 1000)
ax1.set_ylim(trace.min() - 0.28, trace.max() + 0.74)
ax1.set_xlabel("采样点", fontsize=FONTSIZE, labelpad=4)
ax1.set_ylabel("归一化功耗", fontsize=FONTSIZE)
ax1.tick_params(axis="both", labelsize=FONTSIZE)
ax1.grid(axis="y", color=grid_color, lw=0.5, alpha=0.6)
clean_axis(ax1)

add_bottom_caption(
    ax1,
    "(a) 完整功耗曲线：从完整 AES 加密轨迹中定位目标 S 盒泄漏区域",
    y=-0.32
)


# --------------------------
# (b) 兴趣区截取
# --------------------------
ax2 = fig.add_subplot(gs[1])
ax2.plot(x_roi, trace_roi, color=line_color, lw=1.25)

ax2.axvspan(roi_start, roi_end, color=shade_color, zorder=0)
ax2.add_patch(
    Rectangle(
        (roi_start, trace_roi.min() - 0.10),
        roi_end - roi_start,
        trace_roi.max() - trace_roi.min() + 0.20,
        fill=False,
        ec=roi_color,
        lw=1.7,
    )
)

ax2.annotate(
    "截取ROI $\\boldsymbol{x}_i$",
    xy=(roi_center, trace_roi.max() * 0.90),
    xytext=(248, trace_roi.max() + 0.28),
    arrowprops=dict(arrowstyle="->", color=roi_color, lw=1.25),
    color=roi_color,
    fontsize=FONTSIZE,
    ha="left",
)

ax2.text(
    roi_start + 5,
    trace_roi.min() + 0.10,
    "保留与目标中间值相关的局部泄漏，\n减少无关采样点和噪声干扰",
    va="bottom",
    ha="left",
    fontsize=FONTSIZE,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BDBDBD", lw=0.8),
)

ax2.set_xlim(x_roi.min(), x_roi.max())
ax2.set_ylim(trace_roi.min() - 0.18, trace_roi.max() + 0.42)
ax2.set_xlabel("采样点", fontsize=FONTSIZE, labelpad=4)
ax2.set_ylabel("归一化功耗", fontsize=FONTSIZE)
ax2.tick_params(axis="both", labelsize=FONTSIZE)
ax2.grid(axis="y", color=grid_color, lw=0.5, alpha=0.6)
clean_axis(ax2)

add_bottom_caption(
    ax2,
    "(b) 兴趣区截取：以第一轮 S 盒输出为目标截取局部攻击窗口",
    y=-0.38
)


# --------------------------
# (c) 轨迹对齐前后对比
# --------------------------
inner = gs[2].subgridspec(1, 2, wspace=0.25)
ax3a = fig.add_subplot(inner[0])
ax3b = fig.add_subplot(inner[1])

# 为了让多条轨迹都看得清楚，给每条轨迹增加一个小的纵向偏移
vertical_offset = 0.18

for idx, y in enumerate(traces_before):
    ax3a.plot(
        local_x,
        y + idx * vertical_offset,
        lw=1.10,
        color=multi_colors[idx],
        alpha=0.92,
    )

for idx, y in enumerate(traces_after):
    ax3b.plot(
        local_x,
        y + idx * vertical_offset,
        lw=1.10,
        color=multi_colors[idx],
        alpha=0.92,
    )

for ax, title in [
    (ax3a, "对齐前：泄漏峰值存在时间偏移"),
    (ax3b, "对齐后：相同操作的峰值集中"),
]:
    ax.set_xlim(-60, 60)

    # 提高 y 轴上限，为上方说明文字和曲线峰值留出空间
    ax.set_ylim(-0.15, 2.25)

    ax.axvline(0, color="#616161", lw=1.0, ls="--")

    # 这两个文字是左右小图的说明，不是整幅子图(c)的图题。
    # 使用 set_title + pad，明显拉开文字与曲线的距离。
    ax.set_title(title, fontsize=FONTSIZE, pad=16)

    ax.set_xlabel("相对采样点", fontsize=FONTSIZE, labelpad=4)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.grid(axis="y", color=grid_color, lw=0.5, alpha=0.6)
    clean_axis(ax)

ax3a.set_ylabel("多条轨迹", fontsize=FONTSIZE)
ax3b.set_yticklabels([])

# 用浅色箭头表示对齐前的偏移
for s in shifts:
    ax3a.annotate(
        "",
        xy=(s, 1.75),
        xytext=(0, 1.75),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#9E9E9E", alpha=0.6),
    )

# 红色注释移动到右下方，避免与顶部黑色说明文字重叠
ax3b.annotate(
    "目标泄漏位置重合",
    xy=(0, 1.48),          # 箭头指向对齐后的主峰位置
    xytext=(24, 1.18),     # 文字放到右下方，避开顶部黑色文字
    arrowprops=dict(
        arrowstyle="->",
        color=roi_color,
        lw=1.1,
        shrinkA=2,
        shrinkB=2,
    ),
    color=roi_color,
    fontsize=FONTSIZE,
    ha="left",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.20",
        fc="white",
        ec="none",
        alpha=0.85,
    ),
    zorder=5,
)

# 对于 (c) 这种由左右两个小图组成的组合子图，
# 使用 fig.text 放在两个坐标轴整体的中央下方。
# 先绘制一次，确保坐标轴位置已经确定。
fig.canvas.draw()
bbox_left = ax3a.get_position()
bbox_right = ax3b.get_position()
x_center_c = (bbox_left.x0 + bbox_right.x1) / 2
y_bottom_c = min(bbox_left.y0, bbox_right.y0) - 0.075

fig.text(
    x_center_c,
    y_bottom_c,
    "(c) 轨迹对齐前后对比：将同一密码操作对应的局部泄漏校准到相近采样位置",
    fontsize=FONTSIZE,
    fontweight="bold",
    ha="center",
    va="top",
)

# ============================================================
# 5. 保存文件
# ============================================================

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
plt.close(fig)

print("绘图完成，文件已保存到：")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())
print(OUT_SVG.resolve())
print()
print("当前 Matplotlib 字体配置：")
print("font.family =", matplotlib.rcParams.get("font.family"))
print("font.size =", matplotlib.rcParams.get("font.size"))
