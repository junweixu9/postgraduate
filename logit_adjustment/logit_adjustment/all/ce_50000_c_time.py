import time

from data_load_ascad_50000 import read_data
from SCA_util_standard import perform_attacks
import DL_model
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR
import numpy as np
from tensorflow.keras import backend as K
import matplotlib
from scipy.special import softmax
import requests
matplotlib.use('TkAgg')
# 设置字体
import tensorflow as tf

matplotlib.rc("font", family='SimSun')

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
            X_attack_valid_metric, all_valid_plt_attack_metric, Y_attack_valid= self.validation[0], \
                self.validation[1], \
                self.validation[2]

            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)

            epoch_count = epoch_count + 1

            y_pred_valid_metric = softmax(y_pred_valid_metric, 1)
            avg_rank_current, avg_attack_traces = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])


            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))




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
    # adjustments = np.log(label_freq_array + 1e-12)

    return adjustments


def compute_adjustment_ldl(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling_part = np.sum(Y_profiling[:, :9], 0)
    Y_profiling_all = np.sum(Y_profiling_part)
    label_freq_array = Y_profiling_part / Y_profiling_all
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    return adjustments


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss

def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    # print("y_pred.shape: ", y_pred.shape)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    alpha = np.array(0.25, dtype=np.float32)
    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)
    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, 2) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))


def flr_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 10
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = categorical_focal_loss_fixed(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / (average_fake_k_loss + 1e-40)
    return loss_cer

def cer_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 10
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = tf.keras.losses.categorical_crossentropy(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / (average_fake_k_loss + 1e-40)
    return loss_cer

def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_true = K.clip(y_true, K.epsilon(), 1)
    if adjust_flag:
        y_pred = y_pred + 1 * adjustments
    y_pred = tf.nn.softmax(y_pred, 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


if __name__ == '__main__':

    """变量配置"""
    dataset = 'CHES_CTF'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 2000
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
    count = 1
    best_weights = None
    tro = 1
    adjust_flag = True
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

    """创建神经网络模型"""
    """ select the output_metric function """

    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       flr_adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    # model = build_optimized_clrm_model(input_shape=(700, 1), num_classes=9)
    # model.compile(loss=adjustment_loss, optimizer='adam', metrics=None)
    # model.summary()

    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
         lr_manager
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
    start_time = time.perf_counter()
    history = model.fit(x=X_profiling[:45000], y=Y_profiling[:45000, :9], batch_size=128, verbose=2,
                        epochs=epoch,
                        callbacks=callback
                        )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time} 秒")
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    start_time = time.perf_counter()
    predictions = model.predict(X_attack[:5000])
    predictions = softmax(predictions,1)

    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")

    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack[:5000], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)

    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"密钥恢复: {execution_time} 秒")

    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))


    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')