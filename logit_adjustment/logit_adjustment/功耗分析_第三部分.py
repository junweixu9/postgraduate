# -*- coding: utf-8 -*-
"""
SCA 轨迹对齐前后对比示意图

字体要求：
    中文：宋体
    英文：Times New Roman
    图片字体：五号，即 10.5 pt

依赖安装：
    pip install numpy matplotlib mksci-font

输出：
    ./sca_alignment_fig/SCA_trace_alignment_before_after.png
    ./sca_alignment_fig/SCA_trace_alignment_before_after.pdf
    ./sca_alignment_fig/SCA_trace_alignment_before_after.svg

说明：
    本图是轨迹对齐流程示意图，不是某一公开数据集的原始实测曲线。
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    from mksci_font import config_font
except ImportError as e:
    raise ImportError(
        "未检测到 mksci-font，请先在终端执行：pip install mksci-font"
    ) from e


# ============================================================
# 1. 字体与输出设置
# ============================================================

FONTSIZE = 10.5

config_font({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "figure.titlesize": FONTSIZE,
})

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = Path("./sca_alignment_fig")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "SCA_trace_alignment_before_after.png"
OUT_PDF = OUT_DIR / "SCA_trace_alignment_before_after.pdf"
OUT_SVG = OUT_DIR / "SCA_trace_alignment_before_after.svg"


# ============================================================
# 2. 基础函数
# ============================================================

np.random.seed(2026)


def gaussian(x, mu, sigma, amp):
    """高斯峰，用于模拟局部泄漏峰值。"""
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


# ============================================================
# 3. 生成对齐前后的多条局部轨迹
# ============================================================

local_x = np.linspace(-60, 60, 240)

# 对齐前：同一泄漏峰值在不同轨迹中存在时间偏移
shifts = [-17, -9, 0, 11, 20]
amps = [1.00, 0.92, 1.05, 0.96, 1.02]

traces_before = []
traces_after = []

for s, a in zip(shifts, amps):
    noise = smooth_noise(len(local_x), scale=0.09, k=7)

    before = (
        gaussian(local_x, s, 8.5, a)
        + 0.25 * gaussian(local_x, s + 19, 10, 0.5)
        + noise
    )

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

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(8.6, 2.8),
    dpi=200,
    sharey=True,
)

roi_color = "#E53935"
grid_color = "#E0E0E0"
multi_colors = ["#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00", "#00838F"]

vertical_offset = 0.18

# --------------------------
# 左图：对齐前
# --------------------------
for idx, y in enumerate(traces_before):
    ax1.plot(
        local_x,
        y + idx * vertical_offset,
        lw=1.10,
        color=multi_colors[idx],
        alpha=0.92,
    )

# 对齐前的偏移箭头
for s in shifts:
    ax1.annotate(
        "",
        xy=(s, 1.75),
        xytext=(0, 1.75),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.75,
            color="#9E9E9E",
            alpha=0.6,
        ),
    )

ax1.axvline(0, color="#616161", lw=1.0, ls="--")
ax1.set_title("对齐前：泄漏峰值存在时间偏移", fontsize=FONTSIZE, pad=10)
ax1.set_xlabel("相对采样点", fontsize=FONTSIZE, labelpad=4)
ax1.set_ylabel("多条轨迹", fontsize=FONTSIZE)


# --------------------------
# 右图：对齐后
# --------------------------
for idx, y in enumerate(traces_after):
    ax2.plot(
        local_x,
        y + idx * vertical_offset,
        lw=1.10,
        color=multi_colors[idx],
        alpha=0.92,
    )

ax2.axvline(0, color="#616161", lw=1.0, ls="--")
ax2.set_title("对齐后：相同操作的峰值集中", fontsize=FONTSIZE, pad=10)
ax2.set_xlabel("相对采样点", fontsize=FONTSIZE, labelpad=4)

ax2.annotate(
    "目标泄漏位置重合",
    xy=(0, 1.48),
    xytext=(24, 1.18),
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


# ============================================================
# 5. 坐标轴与版式
# ============================================================

for ax in [ax1, ax2]:
    ax.set_xlim(-60, 60)
    ax.set_ylim(-0.15, 2.25)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.grid(axis="y", color=grid_color, lw=0.5, alpha=0.6)
    clean_axis(ax)

ax2.set_yticklabels([])

# # 整体图题放在下方
# fig.text(
#     0.5,
#     0.015,
#     "轨迹对齐前后对比：将同一密码操作对应的局部泄漏校准到相近采样位置",
#     fontsize=FONTSIZE,
#     fontweight="bold",
#     ha="center",
#     va="bottom",
# )

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.86,
    bottom=0.22,
    wspace=0.22,
)

# ============================================================
# 6. 保存文件
# ============================================================

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.02)

plt.close(fig)

print("绘图完成，文件已保存到：")
print(OUT_PNG.resolve())
print(OUT_PDF.resolve())
print(OUT_SVG.resolve())