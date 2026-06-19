import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 参数设置
x = np.linspace(0, 8, 800)  # x轴范围
mean = 4  # 所有高斯分布的均值相同
std_devs = [0.05,0.1,0.3, 0.4, 0.5, 0.6, 0.7,0.75,0.85]  # 不同的标准差

# 生成带噪声的高斯曲线
gaussian_data = [np.exp(-((x - mean)**2) / (2 * 1)) * s for s in std_devs]
noisy_data = [g + np.random.normal(0, 0.001, g.shape) for g in gaussian_data]

# 第二部分参数
x_1 = np.linspace(8, 12, 400)  # 第二部分 x 范围
mean_1 = 9  # 第二部分均值
std_devs_1 = [0.3] * 9  # 第二部分标准差（与第一条曲线相同）

# 生成第二部分数据
gaussian_data_1 = [np.exp(-((x_1 - mean_1)**2) / (2 * s**2))*0.7for s in std_devs_1]
noisy_data_1 = [g + np.random.normal(0, 0.001, g.shape) for g in gaussian_data_1]

# 创建图形
plt.figure(figsize=(10, 8))

# 定义颜色映射
cmap = plt.cm.get_cmap('tab10', len(noisy_data))

# 绘制所有曲线
for i in range(len(noisy_data)):
    plt.plot(x, noisy_data[i], color=cmap(i), label=f'HW{i}')
    plt.plot(x_1, noisy_data_1[i], color=cmap(i), linestyle='--')

# 图形设置
plt.xlabel('时间',fontsize=22)
plt.ylabel('能量',fontsize=22)
plt.title('中间值与能量变化的关系',fontsize=22)

plt.ylim(0, 1)
# 图例设置
legend = plt.legend(loc='upper right',fontsize=22)
plt.tight_layout()
plt.show()