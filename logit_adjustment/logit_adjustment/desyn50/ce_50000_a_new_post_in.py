from data_load_ascad_50000 import read_data
from SCA_util_in import perform_attacks
import DL_model
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR
import numpy as np
from tensorflow.keras import backend as K

import requests
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
            global epoch_count
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric, Y_attack_valid = self.validation[0], \
                self.validation[1], \
                self.validation[2]

            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)

            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks, Y_attack_valid)

            all_corr_logs.append(avg_corr_current)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                print("corr:", avg_corr_current)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("corr:", avg_corr_current)

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
                    if count == 10:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)


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
    num_traces_attacks = 2000
    epoch_count = 1
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

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)

    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000], Y_profiling))
        # all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
    ]
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :9], batch_size=128, verbose=2,
                        epochs=epoch,
                        callbacks=callback
                        )
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    predictions = model.predict(X_attack[5000:])
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks, Y_attack[5000:])

    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')