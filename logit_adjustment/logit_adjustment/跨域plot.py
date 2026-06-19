import numpy as np
import matplotlib.pyplot as plt

# 核心配置：保持高DPI保证小图清晰
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 更换随机种子（保证数据分布与之前版本不同）
np.random.seed(100)

# 生成基础数据：横轴0-5（截半）
x = np.linspace(0, 5, 300)
y = np.zeros_like(x)

# 调整后的峰值参数（分布与原版本差异显著）
n_peaks = 8
for _ in range(n_peaks):
    center = np.random.uniform(0.2, 4.8)
    height = np.random.uniform(0.3, 1)
    width = np.random.uniform(0.1, 0.3)
    y += height * np.exp(-abs((x - center)/width)**1.5)

# 调整底部波纹
base_wave = 0.4 * np.sin(8 * x) + 0.15
y = np.maximum(y, base_wave)

# 超小画布尺寸（3×1.8）
fig, ax = plt.subplots(figsize=(3, 1.8))
# 核心：波形线条改为红色（调整色值为正红色，更醒目）
ax.plot(x, y, color='#FF0000', linewidth=1.2)  # 纯红色 #FF0000

# 样式纯净化：横轴0-5，无坐标轴/刻度/图例
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 5)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
# 刻度标记改为更深的红色（与波形红形成层次）
for tick_pos in np.linspace(1, 4.5, 5):
    ax.plot(tick_pos, 1.2, '|', color='#CC0000', markersize=4, transform=ax.get_xaxis_transform())

# 无冗余空白
plt.tight_layout(pad=0.05)

# 保存红色版图片
plt.savefig('red_style_small_wave.png', bbox_inches='tight', pad_inches=0.02)
plt.show()