import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 数据
hamming_weights = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sample_counts = [166, 1568, 5500, 10848, 13747, 11023, 5416, 1532, 200]
percentages = [0.3, 3.1, 11.0, 21.7, 27.5, 22.0, 10.8, 3.1, 0.4]

# 创建图形
fig, ax = plt.subplots(figsize=(8.5, 7))

# 绘制渐变色柱状图
colors = plt.cm.viridis(np.linspace(0, 1, len(hamming_weights)))
bars = ax.bar(hamming_weights, sample_counts, color=colors)

# 添加顶部标签
for bar, count, percentage in zip(bars, sample_counts, percentages):
    ax.annotate(f'{count} ({percentage}%)',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')

# 图表配置
ax.set_title('汉明重量标签分布图',fontsize=20)
ax.set_xlabel('HW',fontsize=20)
ax.set_ylabel('类别数量',fontsize=20)
ax.set_xticks(hamming_weights)

plt.tight_layout()
plt.show()