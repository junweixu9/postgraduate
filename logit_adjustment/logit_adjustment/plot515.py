import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mksci_font import config_font

# 如果您在本地运行且需要弹窗显示，可以保留 TkAgg
# matplotlib.use('TkAgg')
config_font({"font.size": 24})  # 可同时设置字号

# 数据集 AR 的实验结果 (保持您提供的数据不变)
traces_AR = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

# 蓝色圆圈实线 (CER)
cer_AR = np.array([62.0, 42.5, 32.2, 19.6, 12.5, 5.0, 1.7, 1.2, 6.1, 1.5])

# 橙色方块虚线 (FLR)
flr_AR = np.array([34.2, 7.5, 2.1, 4.2, 1.0, 0.6, 0.2, 0.0, 0.0, 0.0])

# 红色三角点划线 (CE-LA, Ours)
cela_AR = np.array([15.2, 2.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 绿色 X 点线 (CE)
# 注意：图片中CE起始点约为45，且下降较慢
ce_AR = np.array([45.0, 36.0, 37.1, 25.0, 22.4, 21.5, 16.8, 11.5, 8.5, 5.1])

# 绿色菱形实线 (FLR-Center)
flr_center_AR = np.array([39.5, 29.4, 25.2, 9.5, 5.7, 3.6, 2.8, 2.5, 2.4, 2.4])

# 红色五边形虚线 (FLR-Softmax)
flr_softmax_AR = np.array([40.8, 30.5, 24.6, 10.4, 7.5, 5.1, 4.4, 3.7, 3.1, 3.0])

# 紫色星形点划线 (KNLL)
knll_AR = np.array([20.4, 6.7, 1.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 创建图表
plt.figure(figsize=(10, 7.5))

# 绘制每种方法的数据
plt.plot(traces_AR, cer_AR, marker='o', linestyle='-', label='CER', markersize=7)
plt.plot(traces_AR, flr_AR, marker='s', linestyle='--', label='FLR', markersize=7)
plt.plot(traces_AR, cela_AR, marker='^', linestyle='-.', label='CE-LA (ours)', linewidth=2.5, markersize=9, color="red")
plt.plot(traces_AR, ce_AR, marker='x', linestyle=':', label='CE', markersize=7, color="green")
plt.plot(traces_AR, flr_center_AR, marker='d', linestyle='-', label='FLR-Center', markersize=7)
plt.plot(traces_AR, flr_softmax_AR, marker='p', linestyle='--', label='FLR-Softmax', markersize=7)
plt.plot(traces_AR, knll_AR, marker='*', linestyle='-.', label='KNLL', markersize=8)

# 添加图表标题和轴标签
# --- 修改点 1：增大轴标签字体 (改为 24) ---
plt.xlabel('攻击轨迹数量', fontsize=24)
plt.ylabel('猜测熵', fontsize=24)

# 添加图例
# --- 修改点 2：增大图例字体 (改为 24) ---
plt.legend(fontsize=24, loc='upper right', ncol=1)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 设置坐标轴刻度字体大小
# --- 修改点 3：增大刻度字体 (改为 24) ---
plt.xticks(traces_AR, fontsize=24, rotation=0)
plt.yticks(fontsize=24)

# 调整坐标轴范围
plt.xlim(traces_AR.min() - 20, traces_AR.max() + 20)
all_data_C_updated = np.concatenate([
    cer_AR, flr_AR, cela_AR, ce_AR,
    flr_center_AR, flr_softmax_AR, knll_AR
])
# Y轴下限
plt.ylim(-5, all_data_C_updated[all_data_C_updated >= 0].max() * 1.05 if all_data_C_updated[all_data_C_updated >= 0].size > 0 else 10)

# 使用 plt.subplots_adjust() 稍微增大边距以防大字体被切掉
plt.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.93)

# 保存图片
file_name = 'dataset_C_guessing_entropy_updated_extra_large_24.png'
plt.savefig(file_name, bbox_inches='tight', pad_inches=0.05, dpi=300)
print(f"图表已保存为 {file_name}")

# 显示图表
plt.show()