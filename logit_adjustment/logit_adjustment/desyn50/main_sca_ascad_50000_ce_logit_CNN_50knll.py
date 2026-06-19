from data_load_ascad_50000_100 import read_data
import DL_model_100 as DL_model
from SCA_util_knl_new import perform_attacks
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
from datetime import datetime
from clr import OneCycleLR
import matplotlib.pyplot as plt

class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(all, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(all, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)

            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            all_corr_logs.append(avg_corr_current)

            if not corr_logs:
                corr_logs.append(avg_corr_current)
                best_weights = self.model.get_weights()
            else:
                if corr_logs[-1] < avg_corr_current:
                    corr_logs.append(avg_corr_current)
                    best_weights = self.model.get_weights()
                    count = 0
                else:
                    count = count + 1
                    print(count)
                    if count == 30:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                print("corr:", avg_corr_current)


            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("corr:", avg_corr_current)


def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling = np.argmax(Y_profiling[:, :9], 1)
    label_freq = {}
    for key in Y_profiling:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments

def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss

# def compute_loss_parameters(Y_profiling):
#     """compute the base probabilities"""
#
#     Y_profiling = np.argmax(Y_profiling[:, :9], 1)
#     label_freq = {}
#     for key in Y_profiling:
#         label_freq[key] = label_freq.get(key, 0) + 1
#     label_freq = dict(sorted(label_freq.items()))
#     label_freq_array = np.array(list(label_freq.values()))
#     label_freq_array = label_freq_array / label_freq_array.sum()
#     scaling_factors = np.exp(np.power(label_freq_array + 1e-12, -0.25))
#
#     # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
#     # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
#     # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
#     return tf.constant(label_freq_array, dtype=tf.float32), tf.constant(scaling_factors, dtype=tf.float32)
#
# def new_custom_loss(y_true, y_pred):
#     y_true = tf.cast(y_true[:, :classes], dtype=tf.float32)
#     logits = y_pred  # y_pred 是模型的原始输出 (logits)
#
#     # 1. 计算所有类别的基础分数: π_{y'} * e^(f_{y'})
#     exp_logits = tf.exp(logits)
#     base_scores = exp_logits * PI_ARRAY  # 维度: (batch_size, classes)
#
#     # 2. 识别每个样本的真实类别 y
#     true_class_indices = tf.argmax(y_true, axis=1)  # 维度: (batch_size,)
#
#     # 3. 为每个样本获取其对应的缩放因子 e^(π_y^(-1/4))
#     # tf.gather 从 SCALING_FACTORS 中根据真实类别索引挑出对应的缩放因子
#     scaling_factors_per_sample = tf.gather(SCALING_FACTORS, true_class_indices)
#     scaling_factors_per_sample = tf.reshape(scaling_factors_per_sample, (-1, 1))  # 维度: (batch_size, 1)
#
#     # 4. 构建损失函数的分母
#     # 4.1 计算真实类别的分数: π_y * e^f_y
#     score_for_true_class = tf.reduce_sum(base_scores * y_true, axis=1, keepdims=True)  # 维度: (batch_size, 1)
#
#     # 4.2 计算所有其他类别的分数之和: Σ_{y'≠y} π_{y'} * e^f_{y'}
#     mask_for_other_classes = 1.0 - y_true
#     scores_for_other_classes = base_scores * mask_for_other_classes
#     sum_scores_for_other_classes = tf.reduce_sum(scores_for_other_classes, axis=1, keepdims=True)  # 维度: (batch_size, 1)
#
#     # 4.3 对其他类别的分数和进行缩放
#     scaled_sum_for_other_classes = sum_scores_for_other_classes * scaling_factors_per_sample
#
#     # 4.4 最终的分母
#     denominator = score_for_true_class + scaled_sum_for_other_classes
#
#     # 5. 计算最终的概率值 P_y = Numerator / Denominator
#     # Numerator 就是 score_for_true_class
#     prob_y = score_for_true_class / (denominator + 1e-12)  # 加上epsilon避免除以零
#
#     # 6. 计算损失 -log(P_y)
#     loss = -tf.math.log(prob_y + 1e-12)  # 加上epsilon避免log(0)
#
#     return loss

if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 5000
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 50
    learning_rate = 1e-2
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
    adjust_flag = False
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

    adjustments = compute_adjustment(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    # PI_ARRAY, SCALING_FACTORS = compute_loss_parameters(Y_profiling)

    """创建神经网络模型"""
    """ select the output_metric function """
    lr_manager = OneCycleLR(len(X_profiling[45000:]), 256, 1e-2, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        all(validation=(X_profiling[45000:], plt_profiling[45000:], Y_profiling[45000:])), lr_manager
    ]
    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    history = model.fit(x=X_profiling[:45000], y=Y_profiling[:45000, :9], batch_size=batch_size, verbose=2, epochs=epoch,
                        callbacks=callback
                        )
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    predictions = model.predict(X_attack[:10000])
    # predictions = predictions - 0.25 * adjustments
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[:10000], predictions, "attack_traces",
                                    leakage_model, dataset, 10000)
    log = open('F:/result/desyn/a/knl_desync100_new.txt', mode='a',
               encoding='utf-8')
    print(file=log)
    print(file=log)
    print(experiment_time, file=log)

    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key], file=log)

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)

    current_time_end = datetime.now()
    day_end = current_time_end.day
    hour_end = current_time_end.hour
    minute_end = current_time_end.minute
    experiment_time_end = 'time_is{}_{}_{}'.format(int(day_end),
                                                   int(hour_end),
                                                   int(minute_end)
                                                   )
    print(experiment_time, file=log)

    print(experiment_time_end, file=log)
    print("learning_rate_", learning_rate,
          "___architecture_", attack_model,
          "___dataset_", dataset,
          "___epochs_", epoch,
          "___num_profiling_traces_", num_profiling_traces,
          "___num_attack_traces_", num_traces_attacks,
          "___sigma_hw_", sigma_hw,
          "___sigma_id_", sigma_id,
          "___adjust_flag_", adjust_flag,
          "___tro_", tro, file=log)

    log.close()
    def generate_image(text="程序已跑完", output_path="completed_matplotlib.png"):
        # 设置图像参数
        fig = plt.figure(figsize=(6, 3), dpi=280)  # 图像尺寸（宽600px，高300px）
        ax = plt.axes([0, 0, 1, 1], frameon=False)  # 全屏显示，无边框
        plt.axis('off')  # 隐藏坐标轴

        # 设置背景颜色（白色）
        ax.set_facecolor('white')

        # 设置中文字体（需要指定字体路径）
        # 中文字体路径示例（根据系统调整）：
        # Windows: 'C:/Windows/Fonts/simhei.ttf'
        # Linux: '/usr/share/fonts/truetype/arphic/uming.ttc'
        # macOS: '/System/Library/Fonts/Supplemental/Song.ttf'
        font_path = 'simhei.ttf'  # 如果使用默认字体可删除此行和下面的 FontProperties

        # 自定义字体（中文字体需要指定路径）
        font_properties = {
            'family': 'SimHei',  # 中文字体名称（需系统支持）
            'size': 24,  # 字体大小
            'color': 'black',  # 字体颜色
            'weight': 'bold'  # 粗体
        }

        # 如果需要指定字体路径，使用FontProperties
        # from matplotlib.font_manager import FontProperties
        # prop = FontProperties(fname=font_path)
        # font_properties['fontproperties'] = prop

        # 添加文本（居中显示）
        ax.text(
            0.5,  # x坐标（0-1比例）
            0.5,  # y坐标（0-1比例）
            text,
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            **font_properties
        )

        plt.show()  # 显示图形


    generate_image()

