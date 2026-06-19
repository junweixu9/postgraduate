from data_load_ascad_50000 import read_data
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 3000
    epoch_count = 0
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 50
    learning_rate = 5e-3
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    rank_logs = []
    all_rank_logs = []
    corr_logs = []
    all_corr_logs = []
    loss_logs = []
    kl_loss_logs = []
    count = 0
    best_weights = None
    tro = 1
    adjust_flag = True
    visualize_hw_traces = True  # 是否进行可视化
    peak_search_range = (0, 700)  # 搜索峰值的时间范围
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)
    Y_profiling = Y_profiling[:, :9]
    # 设置绘图风格
    # 设置绘图风格
    try:
        # 方法1: 使用系统自带的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

        print("中文支持已启用")
    except Exception as e:
        print(f"设置中文支持时出错: {e}")
        print("将使用英文显示")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (14, 10),
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14
    })

    # 可视化HW标签对应的能量轨迹
    if visualize_hw_traces and leakage_model == 'HW':
        print("\n" + "=" * 80)
        print("开始可视化HW标签对应的能量轨迹...")
        print(f"共有 {len(Y_profiling)} 条训练轨迹")

        # 将one-hot标签转换为整数标签
        hw_labels = np.argmax(Y_profiling, axis=1)
        unique, counts = np.unique(hw_labels, return_counts=True)
        print("HW标签分布:")
        for hw, count in zip(unique, counts):
            print(f"  HW={hw}: {count}条轨迹 ({count / len(hw_labels):.2%})")

        # 1. 绘制每个HW组的平均能量轨迹
        # plt.figure(figsize=(14, 10))

        # 计算每个时间点的总方差（用于寻找关键点）
        total_variance = np.var(X_profiling, axis=0)
        peak_time = np.argmax(total_variance[peak_search_range[0]:peak_search_range[1]]) + peak_search_range[0]
        print(f"检测到关键时间点: {peak_time} (该点方差最大)")

        # for hw in range(9):
        #     # 获取当前HW的所有轨迹
        #     hw_mask = (hw_labels == hw)
        #     if np.sum(hw_mask) == 0:
        #         print(f"警告: HW={hw}没有轨迹!")
        #         continue
        #
        #     hw_traces = X_profiling[hw_mask]
        #
        #     # 计算平均轨迹
        #     mean_trace = np.mean(hw_traces, axis=0).flatten()  # 确保是一维数组
        #
        #     # 计算标准差范围
        #     std_trace = np.std(hw_traces, axis=0).flatten()  # 确保是一维数组
        #     upper_bound = mean_trace + std_trace
        #     lower_bound = mean_trace - std_trace
        #
        #     # 确保所有数组都是一维的
        #     x_vals = np.arange(len(mean_trace))
            #
            # # 绘制平均轨迹和标准差范围
            # plt.plot(x_vals, mean_trace, label=f'HW={hw}', linewidth=2.5)
            # plt.fill_between(x_vals, lower_bound, upper_bound, alpha=0.15)

        # # 标记关键时间点
        # plt.axvline(peak_time, color='red', linestyle='--', alpha=0.7,
        #             label=f'关键点 (样本 {peak_time})')
        # plt.title('Average Energy Trace as a Function of Hamming Weight', pad=20)
        # plt.xlabel('time')
        # plt.ylabel('energy')
        # plt.legend(title='HW', loc='upper right')
        #
        # plt.show()

        # 2. 绘制每个HW组的示例轨迹
        plt.figure(figsize=(14, 12))
        plt.suptitle('Representative Energy Trace for Each Hamming Weight', fontsize=18)

        for hw in range(9):
            plt.subplot(3, 3, hw + 1)

            # 获取当前HW的所有轨迹
            hw_mask = (hw_labels == hw)
            if np.sum(hw_mask) == 0:
                plt.title(f'HW = {hw} (无数据)')
                continue

            hw_traces = X_profiling[hw_mask]

            # 随机选择10条轨迹
            num_samples = min(2, len(hw_traces))
            sample_indices = np.random.choice(len(hw_traces), num_samples, replace=False)

            # 绘制示例轨迹
            for idx in sample_indices:
                plt.plot(hw_traces[idx].flatten(), alpha=0.6, linewidth=1)  # 确保是一维数组

            # 添加平均轨迹
            mean_trace = np.mean(hw_traces, axis=0).flatten()  # 确保是一维数组
            # plt.plot(mean_trace, 'k-', linewidth=2.5, label='平均值')

            # 标记关键时间点
            # plt.axvline(peak_time, color='red', linestyle='--', alpha=0.7)

            plt.title(f'HW = {hw} (n={len(hw_traces)})')
            plt.xlabel('time')
            plt.ylabel('energy')

            if hw == 0:
                plt.legend(loc='upper right')

        plt.tight_layout()

        plt.show()

        # 3. 分析在关键时间点的线性关系
        peak_values = X_profiling[:, peak_time].flatten()  # 确保是一维数组

        # 计算每个HW组的平均值和标准差
        hw_means = []
        hw_stds = []
        hw_counts = []
        valid_hw = []

        for hw in range(9):
            hw_mask = (hw_labels == hw)
            if np.sum(hw_mask) == 0:
                continue

            values = peak_values[hw_mask].flatten()  # 确保是一维数组
            hw_means.append(np.mean(values))
            hw_stds.append(np.std(values))
            hw_counts.append(len(values))
            valid_hw.append(hw)

        # 线性回归分析
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_hw, hw_means
        )

        # 绘制线性关系图
        plt.figure(figsize=(12, 8))

        # 绘制数据点和误差条
        plt.errorbar(
            valid_hw,
            hw_means,
            yerr=hw_stds,
            fmt='o',
            markersize=8,
            capsize=5,
            label='平均能量 ± 1 标准差'
        )

        # 添加样本数量标签
        for hw, mean, count in zip(valid_hw, hw_means, hw_counts):
            plt.text(hw, mean + 0.5 * max(hw_stds), f'n={count}',
                     ha='center', fontsize=10)

        # 绘制回归线
        regression_line = intercept + slope * np.array(valid_hw)
        plt.plot(valid_hw, regression_line, 'r--', linewidth=2.5,
                 label=f'线性拟合: y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value ** 2:.3f}')

        # 添加统计信息
        plt.text(min(valid_hw) + 0.5, max(hw_means) * 0.8,
                 f'R² = {r_value ** 2:.3f}\np值 = {p_value:.2e}',
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

        plt.title(f'关键点能量与汉明重量的关系 (样本 {peak_time})', pad=15)
        plt.xlabel('汉明重量')
        plt.ylabel('能量消耗')
        plt.xticks(valid_hw)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 4. 计算并绘制R²随时间变化
        r_squared_values = []
        for t in range(X_profiling.shape[1]):
            values = X_profiling[:, t].flatten()  # 确保是一维数组
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(hw_labels, values)
                r_squared_values.append(r_value ** 2)
            except:
                r_squared_values.append(0)

        plt.figure(figsize=(14, 6))
        plt.plot(r_squared_values, linewidth=1.5)
        plt.axvline(peak_time, color='red', linestyle='--', alpha=0.7,
                    label=f'关键点 (样本 {peak_time})')
        plt.title('线性关系强度 (R²) 随时间变化')
        plt.xlabel('时间样本')
        plt.ylabel('R² 值')
        plt.ylim([0, max(r_squared_values) * 1.1])
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("=" * 80)
        print("可视化完成，结果已保存为PNG文件")
        print("=" * 80)