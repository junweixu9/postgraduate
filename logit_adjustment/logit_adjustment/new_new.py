import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc("font", family='SimSun')
# 生成数据
num_traces = np.arange(0, 2001, 50)  # x轴：0到2000，步长50
original_ranks = 160 / (1 + num_traces / 100)  # 原始轨迹排名函数
mixup_ranks = original_ranks + 10 * np.exp(-num_traces / 300)  # 混合轨迹排名函数

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)  # 图形尺寸1000x600像素

# 绘制曲线
plt.plot(num_traces, original_ranks,
         label='Original Traces',
         color='#1f77b4',  # 默认蓝色
         linewidth=2)

plt.plot(num_traces, mixup_ranks,
         label='Mixup Traces',
         color='#ff7f0e',  # 默认橙色
         linewidth=2,
         linestyle='--')   # 虚线样式

# 添加标注
# plt.title('(b) Hardware Model Performance', fontsize=12, pad=20)
plt.xlabel('能量轨迹数量', fontsize=10)
plt.ylabel('正确密钥的排名', fontsize=10)

# 设置坐标轴范围
plt.xlim(0, 2000)
plt.ylim(0, max(mixup_ranks)*1.1)

# 图例和网格
plt.legend(frameon=True, fontsize=10)
plt.grid(linestyle=':', linewidth=0.5, alpha=0.8)

# 保存并显示
plt.tight_layout()
plt.savefig('hardware_model_plot.png', bbox_inches='tight')
plt.show()