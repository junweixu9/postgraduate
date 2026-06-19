from data_load_ascad_50000 import read_data
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def visualize_hw_analysis(X_profiling, Y_profiling, leakage_model='HW'):
    """
    可视化HW泄露模型的能量轨迹特征分析
    包括：平均轨迹、分布特性、线性关系分析、关键时间点检测
    """

    # 数据预处理
    hw_labels = np.argmax(Y_profiling, axis=1)
    unique, counts = np.unique(hw_labels, return_counts=True)
    print(f"HW标签分布: {dict(zip(unique, counts))}")

    # 1. 关键时间点检测 (方差最大点)
    total_variance = np.var(X_profiling, axis=0)
    peak_time = np.argmax(total_variance[-200:]) + len(total_variance) - 200
    print(f"检测到关键时间点: {peak_time} (最后200个采样点中方差最大)")

    # 2. 平均轨迹与标准差可视化
    plt.figure(figsize=(16, 10))

    # 颜色映射配置
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 9)]

    for hw in range(9):
        mask = (hw_labels == hw)
        if not np.any(mask):
            continue

        traces = X_profiling[mask]
        mean_trace = np.mean(traces, axis=0)  # shape: (700,)
        std_trace = np.std(traces, axis=0)  # shape: (700,)

        # 确保是1D数组（防止意外的维度扩展）
        mean_trace = np.squeeze(mean_trace)
        std_trace = np.squeeze(std_trace)

        lower_bound = mean_trace - std_trace
        upper_bound = mean_trace + std_trace

        # 绘制平均轨迹和置信区间
        plt.plot(mean_trace,
                 label=f'HW={hw}',
                 color=colors[hw],
                 linewidth=2)
        plt.fill_between(np.arange(len(mean_trace)),
                         lower_bound,
                         upper_bound,
                         color=colors[hw],
                         alpha=0.2)

    # 标注关键时间点
    plt.axvline(peak_time, color='red', linestyle='--',
                alpha=0.7, linewidth=2, label=f'Key Point (Sample {peak_time})')

    # 图形美化
    plt.title('Average Power Traces by Hamming Weight with 1σ Confidence Bands', fontsize=16)
    plt.xlabel('Time Samples', fontsize=14)
    plt.ylabel('Power Consumption', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Hamming Weight', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('hw_average_traces.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 典型轨迹可视化
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Representative Power Traces for Each Hamming Weight', fontsize=20, y=0.95)

    for hw, ax in enumerate(axes.flat):
        mask = (hw_labels == hw)
        if not np.any(mask):
            ax.set_title(f'HW={hw} (No Data)', fontsize=12)
            continue

        traces = X_profiling[mask]
        sample_indices = np.random.choice(len(traces), min(10, len(traces)), replace=False)

        # 绘制样本轨迹
        for idx in sample_indices:
            ax.plot(traces[idx], alpha=0.4, color=colors[hw], linewidth=1)

        # 绘制平均轨迹
        mean_trace = np.mean(traces, axis=0)
        ax.plot(mean_trace, color='black', linewidth=2, label='Mean')

        # 标注关键时间点
        ax.axvline(peak_time, color='red', linestyle='--', alpha=0.7)

        # 添加统计信息
        ax.set_title(f'HW={hw} (n={len(traces)})', fontsize=14)
        ax.set_xlabel('Time Samples', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.grid(True, alpha=0.3)

        if hw == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('hw_sample_traces.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 线性关系分析
    peak_values = X_profiling[:, peak_time]

    # 计算各HW组的统计参数
    hw_stats = []
    valid_hw = []

    for hw in range(9):
        mask = (hw_labels == hw)
        if not np.any(mask):
            continue

        values = peak_values[mask]
        mean = np.mean(values)
        std = np.std(values)
        count = len(values)

        hw_stats.append((hw, mean, std, count))
        valid_hw.append(hw)

    # 线性回归分析
    hw_list = [x[0] for x in hw_stats]
    mean_list = [x[1] for x in hw_stats]

    slope, intercept, r_value, p_value, std_err = stats.linregress(hw_list, mean_list)

    # 可视化线性关系
    plt.figure(figsize=(12, 8))
    plt.errorbar(hw_list, mean_list,
                 yerr=[x[2] for x in hw_stats],
                 fmt='o',
                 markersize=8,
                 capsize=5,
                 label='Mean Power ± 1σ',
                 color='blue')

    # 添加样本数量标注
    for i, (hw, mean, _, count) in enumerate(hw_stats):
        plt.text(hw, mean + 0.5 * max([x[2] for x in hw_stats]),
                 f'n={count}',
                 ha='center', fontsize=10)

    # 绘制回归线
    x_vals = np.array(valid_hw)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r--', linewidth=2,
             label=f'y = {slope:.3f}x + {intercept:.3f}\nR²={r_value ** 2:.4f}')

    # 添加统计信息
    plt.text(min(valid_hw) + 0.5, max(mean_list) * 0.8,
             f'R² = {r_value ** 2:.4f}\np-value = {p_value:.2e}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f'Linear Relationship at Key Point (Sample {peak_time})', fontsize=16)
    plt.xlabel('Hamming Weight', fontsize=14)
    plt.ylabel('Power Consumption', fontsize=14)
    plt.xticks(valid_hw)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hw_linear_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. R²时序分析
    r_squared_values = []
    for t in range(X_profiling.shape[1]):
        values = X_profiling[:, t]
        try:
            _, _, r, _, _ = stats.linregress(hw_labels, values)
            r_squared_values.append(r ** 2)
        except:
            r_squared_values.append(0)

    plt.figure(figsize=(14, 6))
    plt.plot(r_squared_values, color='blue', linewidth=1.5)
    plt.axvline(peak_time, color='red', linestyle='--', alpha=0.7,
                label=f'Key Point (Sample {peak_time})')
    plt.title('Linear Relationship Strength (R²) Over Time', fontsize=16)
    plt.xlabel('Time Samples', fontsize=14)
    plt.ylabel('R² Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('r_squared_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'peak_time': peak_time,
        'r_squared_peak': r_squared_values[peak_time],
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value
        }
    }


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
    visualize_hw_analysis(X_profiling, Y_profiling, leakage_model='HW')
