# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg')
# # 设置字体
# import numpy as np
# matplotlib.rc("font",family='SimSun')
# # 转换为 NumPy 数组
#
# cm = np.ones((9,9))
# f = plt.figure(figsize=(15, 10), dpi=300)
# ax = plt.subplot(111)
#
# sns.set(font_scale=1.1)  # 将混淆矩阵中的数字字体变大**
# sns.heatmap(cm, cmap='Blues', annot=True, fmt='g')
# ax.xaxis.tick_top()  # 将x轴刻度放置再top位置
# ax.set_ylabel('真实标签', fontsize=14)
# ax.set_xlabel('预测标签', fontsize=14)
# ax.tick_params(axis='y', labelsize=14, labelrotation=45)  # y轴
# ax.tick_params(axis='x', labelsize=14)
# plt.show()
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
matplotlib.use('TkAgg')

matplotlib.rc("font", family='SimSun')

cm = np.random.randint(0, 1001, size=(9, 9))
plt.figure(figsize=(10, 8), dpi=120)
ax = plt.subplot(111)

sns.set(font_scale=1.5)

# 使用更柔和的色板，添加边框线
sns.heatmap(cm,
            cmap='binary',  # 关键修改：从白到黑的渐变
            annot=True,
            fmt='g',
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "aspect": 30})

ax.xaxis.tick_top()
ax.set_ylabel('真实标签', fontsize=22, labelpad=12)
ax.set_xlabel('预测标签', fontsize=22, labelpad=12)

# 正确设置 y 轴标签旋转和对齐（替换原来的 rotation_mode）
ax.tick_params(axis='y',
               labelsize=22,
               labelrotation=45,
               pad=5,
               labelleft=True)

# 手动调整 y 轴标签的对齐方式（替代 rotation_mode 的功能）
plt.setp(ax.get_yticklabels(), rotation=45, ha='right')

# 调整 x 轴标签对齐
ax.tick_params(axis='x', labelsize=22, pad=5)
ax.xaxis.set_tick_params(rotation=0)

# 添加标题并调整间距
ax.set_title('混淆矩阵', fontsize=22, pad=20, fontweight='bold')

# 调整颜色条字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# 自动调整布局
plt.tight_layout()

plt.savefig('hah_2.png', bbox_inches='tight', dpi=140, pad_inches=0.1)
plt.show()
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# matplotlib.use('TkAgg')
#
# matplotlib.rc("font", family='SimSun')
#
# cm = np.random.randint(0, 1001, size=(9, 9))
# plt.figure(figsize=(10, 8), dpi=110)
# ax = plt.gca()  # 直接获取当前坐标轴，避免使用 plt.subplot(111)
#
# sns.set(font_scale=1.5)
#
# # 绘制热力图并减少颜色条的间距
# sns.heatmap(cm,
#             cmap='binary',
#             annot=True,
#             fmt='g',
#             linewidths=0.5,
#             linecolor='white',
#             cbar_kws={"shrink": 0.8, "pad": 0.01},  # 减少颜色条与热力图的距离
#             annot_kws={"size": 14})  # 调整注释字体大小
#
# ax.xaxis.tick_top()
# ax.set_ylabel('真实标签', fontsize=21, labelpad=12)
# ax.set_xlabel('预测标签', fontsize=21, labelpad=12)
#
# # 调整 y 轴标签旋转和对齐
# ax.tick_params(axis='y',
#                labelsize=21,
#                labelrotation=45,
#                pad=5,
#                labelleft=True)
# plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
#
# # 调整 x 轴标签对齐
# ax.tick_params(axis='x', labelsize=21, pad=5)
# ax.xaxis.set_tick_params(rotation=0)
#
# # 添加标题并调整间距
# ax.set_title('混淆矩阵', fontsize=21, pad=20)
#
# # 调整颜色条字体大小
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=12)
#
# # 关键调整：手动设置子图边距并强制紧凑布局
# plt.subplots_adjust(
#     left=0.15,  # 左边距
#     right=0.95,  # 右边距
#     bottom=0.15,  # 底边距
#     top=0.9,  # 顶边距
#     wspace=0.2,  # 子图水平间距
#     hspace=0.2  # 子图垂直间距
# )
#
# # 使用 tight_layout 并调整 pad 参数
# plt.tight_layout(pad=0.5, rect=(0, 0, 1, 1))  # rect 参数控制布局区域
#
# plt.show()



# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# matplotlib.use('TkAgg')
#
# matplotlib.rc("font", family='SimSun')
#
# cm = np.random.randint(0, 1001, size=(9, 9))
# plt.figure(figsize=(10, 8), dpi=110)
# ax = plt.gca()  # 直接使用当前坐标轴
#
# sns.set(font_scale=1.5)
#
# # 关键修改1：调整颜色条的间距（pad参数）
# sns.heatmap(cm,
#             cmap='binary',
#             annot=True,
#             fmt='g',
#             linewidths=0.5,
#             linecolor='white',
#             cbar_kws={"shrink": 0.8, "pad": 0.01},  # 减少颜色条与热力图的距离
#             annot_kws={"size": 14})
#
# ax.xaxis.tick_top()
# ax.set_ylabel('真实标签', fontsize=21, labelpad=12)
# ax.set_xlabel('预测标签', fontsize=21, labelpad=12)
#
# ax.tick_params(axis='y',
#                labelsize=21,
#                labelrotation=45,
#                pad=5,
#                labelleft=True)
# plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
#
# ax.tick_params(axis='x', labelsize=21, pad=5)
# ax.xaxis.set_tick_params(rotation=0)
#
# ax.set_title('混淆矩阵', fontsize=21, pad=20)
#
# # 关键修改2：手动调整子图边距
# plt.subplots_adjust(
#     right=0.85  # 减少右侧边距，防止颜色条外的空白
# )
#
# # 关键修改3：使用更严格的紧凑布局
# plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整右侧边界（0.85对应right参数）
#
# plt.show()