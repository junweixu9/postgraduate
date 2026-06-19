import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置参数
correct_key = 100  # 正确密钥的位置（可自定义）
key_space = np.arange(200)  # 密钥空间 0~255

# 生成相关性数据
correlation = np.zeros(200)  # 初始化所有相关性为 0
correlation[correct_key] = 0.87  # 正确密钥处设置最大值
noise = np.random.normal(0, 0.05, size=200)  # 添加轻微噪声
correlation += noise  # 将噪声叠加到相关性上

# 绘图
plt.figure(figsize=(9, 7))
plt.plot(key_space, correlation, color='blue', linewidth=2)

# 添加红色箭头和备注

# 图形设置

plt.ylabel('相关系数',fontsize=22)
plt.title('在各密钥下的中间值与能量变化的相关性',fontsize=22)
ax = plt.gca()
ax.set_xticks(ax.get_xticks())  # 保留刻度线
ax.set_xticklabels(['' for _ in ax.get_xticks()])  # 隐藏数字

plt.legend()
plt.tight_layout()
plt.show()