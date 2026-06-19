import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mksci_font import config_font

# 如果您在本地运行且需要弹窗显示，可以保留 TkAgg
# matplotlib.use('TkAgg')
config_font({"font.size": 24})  # 可同时设置字号

# --- Dataset A 数据 (保持不变) ---
traces_A = np.array([50, 100, 150, 200, 250, 300, 350, 400])

# 各方法对应的猜测熵
cer_A = np.array([36.0, 9.0, 5.0, 1.8, 1.0, 0.5, 0.2, 0.0])
flr_A = np.array([21.0, 7.0, 2.5, 0.9, 0.5, 0.2, 0.0, 0.0])
cela_A = np.array([5.2, 1.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
ce_A = np.array([48.0, 40.0, 35.0, 28.0, 16.5, 8.5, 5.5, 3.2])
flr_center_A = np.array([23.0, 7.5, 4.0, 1.9, 1.0, 0.5, 0.2, 0.0])
flr_softmax_A = np.array([25.0, 8.5, 5.0, 2.8, 1.5, 0.8, 0.5, 0.0])
knll_A = np.array([12.8, 4.5, 2.8, 0.5, 0.2, 0.0, 0.0, 0.0])

# 创建图表
plt.figure(figsize=(10, 7.5))

# 绘制每种方法的数据 (保持原样式设置)
plt.plot(traces_A, cer_A, marker='o', linestyle='-', label='CER', markersize=7)
plt.plot(traces_A, flr_A, marker='s', linestyle='--', label='FLR', markersize=7)

plt.plot(traces_A, ce_A, marker='x', linestyle=':', label='CE', markersize=7, color="green")
plt.plot(traces_A, flr_center_A, marker='d', linestyle='-', label='FLR-Center', markersize=7)
plt.plot(traces_A, flr_softmax_A, marker='p', linestyle='--', label='FLR-Softmax', markersize=7)
plt.plot(traces_A, knll_A, marker='*', linestyle='-.', label='KNLL', markersize=8)
plt.plot(traces_A, cela_A, marker='^', linestyle='-.', label='CE-LA (ours)', linewidth=2.5, markersize=9, color="red")
# --- 修改点 1：增大轴标签字体 (改为 24) ---
plt.xlabel('攻击轨迹数量', fontsize=24)
plt.ylabel('猜测熵', fontsize=24)

# --- 修改点 2：增大图例字体 (改为 24) ---
plt.legend(fontsize=24, loc='upper right', ncol=1)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# --- 修改点 3：增大刻度字体 (改为 24) ---
plt.xticks(traces_A, fontsize=24, rotation=0)
plt.yticks(fontsize=24)

# 调整坐标轴范围
plt.xlim(traces_A.min() - 20, traces_A.max() + 20)

# 合并数据以计算Y轴上限
all_data_A = np.concatenate([
    cer_A, flr_A, cela_A, ce_A,
    flr_center_A, flr_softmax_A, knll_A
])

# Y轴范围调整
plt.ylim(-2, all_data_A.max() * 1.05)

# 使用 plt.subplots_adjust() 稍微增大边距以防大字体被切掉
plt.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.93)

# 保存图片
file_name = 'dataset_A_reproduced_extra_large_font.png'
plt.savefig(file_name, bbox_inches='tight', pad_inches=0.05, dpi=300)
print(f"图表已保存为 {file_name}")

# 显示图表
plt.show()