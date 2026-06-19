# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import os
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import Adam, RMSprop
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
#
# # 参数设置
# total_time = 8  # 总时间(秒)
# dt = 0.01          # 时间步长
# t0 = 3           # 变化发生的中心时间
# sigma = 0.4        # 高斯变化的标准差
# base_value = 0   # 基础能量值
# amplitude1 = 0.8   # 第一个轨迹的变化幅度
# amplitude2 = 2.0   # 第二个轨迹的变化幅度
#
# # 生成时间序列
# t = np.arange(0, total_time, dt)
#
# # 创建高斯变化函数
# def gaussian_change(t, t0, sigma, amplitude):
#     return amplitude * np.exp(-(t - t0)**2 / (2 * sigma**2))
#
# # 计算两个轨迹的能量值
# energy1 = base_value + gaussian_change(t, t0, sigma, amplitude1)
# energy2 = base_value + gaussian_change(t, t0, sigma, amplitude2)
#
# # 创建图形
# plt.figure(figsize=(4.5, 2.5), dpi=100)
#
# # 绘制能量轨迹
# plt.plot(t, energy1, 'b-', linewidth=2.5, label='HW1')
# plt.plot(t, energy2, 'r--', linewidth=2.0, label='HW3')
#
# # 标记变化时刻
#
#
# # 添加标记区域
#
# # 设置图形属性
# plt.title('不同HW的能量轨迹变化比较', fontsize=18, pad=5)
# plt.legend(loc='upper right', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.ylim(0, 2)
# plt.xlim(1,5)
#
#
# plt.legend(loc='upper left', fontsize=18)  # 将字体大小从10增加到14
#
# plt.tight_layout()
# plt.show()




# def ascad_f_id_cnn(loss, metric):
#     img_input = Input(shape=(700, 1))
#     x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
#     x = BatchNormalization()(x)
#     x = AveragePooling1D(25, strides=25)(x)
#     x = Flatten(name='flatten')(x)
#     x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(256, activation='softmax')(x)
#     model = Model(img_input, x)
#     # optimizer = Adam(lr=5e-3)
#     model.compile(loss=loss, optimizer='adam', metrics=metric)
#     model.summary()
#     return model
#
# def ascad_f_hw_cnn_rs(loss, metric):
#     img_input = Input(shape=(700, 1))
#     x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
#     x = AveragePooling1D(4, strides=4)(x)
#     x = Flatten(name='flatten')(x)
#     x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(9)(x)
#     model = Model(img_input, x)
#     # optimizer = Adam(lr=5e-3)
#     model.compile(loss=loss, optimizer='adam', metrics=metric)
#     model.summary()
#     return model
#
# model = ascad_f_id_cnn(None,"crossentropy")
# model = ascad_f_hw_cnn_rs(None,"crossentropy")


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import os
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import Adam, RMSprop
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
#
# # 参数设置
# total_time = 8  # 总时间(秒)
# dt = 0.01          # 时间步长
# t0 = 3           # 变化发生的中心时间
# sigma = 0.4        # 高斯变化的标准差
# base_value = 0   # 基础能量值
# amplitude1 = 0.8   # 第一个轨迹的变化幅度
# amplitude2 = 2.0   # 第二个轨迹的变化幅度
#
# # 生成时间序列
# t = np.arange(0, total_time, dt)
#
# # 创建高斯变化函数
# def gaussian_change(t, t0, sigma, amplitude):
#     return amplitude * np.exp(-(t - t0)**2 / (2 * sigma**2))
#
# # 计算两个轨迹的能量值
# energy1 = base_value + gaussian_change(t, t0, sigma, amplitude1)
# energy2 = base_value + gaussian_change(t, t0, sigma, amplitude2)
#
# # 创建图形
# plt.figure(figsize=(4.5, 2.5), dpi=100)
#
# # 绘制能量轨迹
# plt.plot(t, energy1, 'b-', linewidth=2.5, label='HW1')
# plt.plot(t, energy2, 'r--', linewidth=2.0, label='HW3')
#
# # 标记变化时刻
#
#
# # 添加标记区域
#
# # 设置图形属性
# plt.title('不同HW的能量轨迹变化比较', fontsize=18, pad=5)
# plt.legend(loc='upper right', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.ylim(0, 2)
# plt.xlim(1,5)
#
#
# plt.legend(loc='upper left', fontsize=18)  # 将字体大小从10增加到14
#
# plt.tight_layout()
# plt.show()




# def ascad_f_id_cnn(loss, metric):
#     img_input = Input(shape=(700, 1))
#     x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
#     x = BatchNormalization()(x)
#     x = AveragePooling1D(25, strides=25)(x)
#     x = Flatten(name='flatten')(x)
#     x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(256, activation='softmax')(x)
#     model = Model(img_input, x)
#     # optimizer = Adam(lr=5e-3)
#     model.compile(loss=loss, optimizer='adam', metrics=metric)
#     model.summary()
#     return model
#
# def ascad_f_hw_cnn_rs(loss, metric):
#     img_input = Input(shape=(700, 1))
#     x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
#     x = AveragePooling1D(4, strides=4)(x)
#     x = Flatten(name='flatten')(x)
#     x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
#     x = Dense(9)(x)
#     model = Model(img_input, x)
#     # optimizer = Adam(lr=5e-3)
#     model.compile(loss=loss, optimizer='adam', metrics=metric)
#     model.summary()
#     return model
#
# model = ascad_f_id_cnn(None,"crossentropy")
# model = ascad_f_hw_cnn_rs(None,"crossentropy")


import numpy as np
import matplotlib.pyplot as plt

# 生成标签（0到10）
labels = np.arange(0, 9)
mean = 4
std = 2
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 生成分布式标签的概率分布（高斯分布，峰值调整为0.8）
distributed_probs = np.exp(-0.5 * ((labels - mean) / std) ** 2) * 0.4

# 生成独热标签（在标签5处为1，其他为0）
one_hot = np.zeros_like(labels)
one_hot[4] = 1
labels = [0, 1, 2, 3, 3.75, 5, 6, 7, 8]
# 绘制图表
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(labels, distributed_probs, marker='o', color='blue', linestyle='--', label='分布式标签')
for i in range(len(labels)):
    plt.vlines(labels[i], 0, distributed_probs[i], colors='blue', linestyles='solid',linewidth=2.5)
plt.plot(4, 1, 'ko', label='独热标签', markersize=10)
plt.vlines(4, 0, 1, linestyles='solid', colors='black',linewidth=2.5)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('标签', fontsize=17)
plt.ylabel('概率', fontsize=17)
plt.title('独热标签与分布式标签的比较',fontsize=17)
plt.legend(loc='upper right',fontsize=17)
plt.ylim(0, 1.1)
plt.tick_params(axis='both', which='both', labelsize=14)
plt.tight_layout()
plt.show()

# 如果需要保存图片，取消下面一行的注释
# plt.savefig('label_distribution_comparison.png', dpi=300, bbox_inches='tight')








