import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = [
    'Baseline (CE_loss)',
    'CE_loss with Intra-class CutMix',
    'Cer_loss',
    'Focal_loss',
    'Cer_Focal_loss',
    'Cer_Focal_Center_loss',
    'CE_loss_based_key'
]
tge0_values = [1560, 310, 540, 1480, 570, 531, 496]  # 泛化能力/TGE0的值
time_values = [680.5, 1001.3, 1308.4, 560.1, 806.6, 1100.6, 690.2]  # 训练时间/s的值

# 设置位置和宽度
n_methods = len(methods)
ind = np.arange(n_methods)  # 柱状图的位置
width = 0.5  # 每个柱子的宽度

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制泛化能力柱状图
bars1 = ax.bar(ind - width/2, tge0_values, width, label='泛化能力/TGE0', color='skyblue')
# 在图例中添加泛化能力的标签

# 绘制训练时间柱状图
bars2 = ax.bar(ind + width/2, time_values, width, label='训练时间/s', color='lightgreen')
# 在图例中添加训练时间的标签

# 添加标签、标题等
ax.set_xlabel('现有方法')
ax.set_ylabel('值')
ax.set_xticks(ind)
ax.set_xticklabels(methods, rotation=45, ha='right')  # 使方法名称垂直显示

# 添加图例
ax.legend()

plt.title('不同方法的泛化能力和训练时间对比')
plt.tight_layout()  # 自动调整布局以避免重叠

# 显示图表
plt.show()