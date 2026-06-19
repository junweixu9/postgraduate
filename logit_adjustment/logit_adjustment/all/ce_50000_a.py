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

# from tensorflow.keras.layers import (
#     Input,
#     Conv1D,
#     BatchNormalization,
#     Activation,
#     MaxPooling1D,
#     Bidirectional,
#     LSTM,
#     Concatenate,
#     GlobalAveragePooling1D,
#     Flatten,
#     Dense,
#     Dropout,
#     Add,
# )
# from tensorflow.keras.models import Model


# def soft_threshold(x, threshold):
#     """
#     软阈值函数。
#     """
#     return tf.math.sign(x) * tf.math.maximum(tf.math.abs(x) - threshold, 0.0)
#
#
# def resnet_module(x, num_filters):  # 将num_filters作为参数
#     """
#     带有自适应软阈值的残差网络模块。
#     """
#     # 主路径
#     res = BatchNormalization(name="res_bn_1")(x)
#     res = Activation("selu", name="res_selu_1")(res)
#     res = Conv1D(
#         filters=num_filters, kernel_size=3, strides=2, padding="same", name="res_conv_1"
#     )(res)
#
#     res = BatchNormalization(name="res_bn_2")(res)
#     res = Activation("selu", name="res_selu_2")(res)
#     res = Conv1D(
#         filters=num_filters, kernel_size=3, strides=1, padding="same", name="res_conv_2"
#     )(res)
#
#     # 自适应软阈值子网络
#     avg_pool = GlobalAveragePooling1D(name="res_gap_for_threshold")(x)
#     threshold_dense1 = Dense(units=max(32, num_filters // 4), activation="relu", name="res_dense_1_for_threshold")(
#         avg_pool)
#     threshold_dense2 = Dense(units=num_filters, activation="sigmoid", name="res_dense_2_for_threshold")(
#         threshold_dense1)
#     threshold = tf.expand_dims(threshold_dense2, axis=1)
#
#     # 应用软阈值
#     res_soft_thresholded = soft_threshold(res, threshold)
#
#     # 快捷连接
#     shortcut = Conv1D(
#         filters=num_filters, kernel_size=1, strides=2, padding="same", name="shortcut_conv"
#     )(x)
#     shortcut = BatchNormalization(name="shortcut_bn")(shortcut)
#
#     # 添加
#     output = Add(name="res_add")([shortcut, res_soft_thresholded])
#     return output
#
#
# def build_optimized_clrm_model(input_shape=(700, 1), num_classes=9):
#     """
#     为9分类任务优化的CLRM模型架构。
#     """
#     input_layer = Input(shape=input_shape, name="input_layer")
#
#     # --- 优化的 CNN 模块 (滤波器减半) ---
#     x = Conv1D(filters=64, kernel_size=3, strides=2, padding="same", name="conv1d_1")(input_layer)  # 128 -> 64
#     x = BatchNormalization(name="bn_1")(x)
#     x = Activation("selu", name="selu_1")(x)
#     x = MaxPooling1D(pool_size=2, strides=2, name="maxpool_1")(x)
#
#     x = Conv1D(filters=128, kernel_size=3, strides=2, padding="same", name="conv1d_2")(x)  # 256 -> 128
#     x = BatchNormalization(name="bn_2")(x)
#     x = Activation("selu", name="selu_2")(x)
#     x = MaxPooling1D(pool_size=2, strides=2, name="maxpool_2")(x)
#
#     x = Conv1D(filters=256, kernel_size=3, strides=2, padding="same", name="conv1d_3")(x)  # 512 -> 256
#     x = BatchNormalization(name="bn_3")(x)
#     x = Activation("selu", name="selu_3")(x)
#     x = MaxPooling1D(pool_size=2, strides=2, name="maxpool_3")(x)
#
#     # --- 优化的 LSTM 模块 (单元数减半) ---
#     lstm_out = Bidirectional(LSTM(64, return_sequences=True, name="bidirectional_lstm"))(x)  # 128 -> 64
#     bn_lstm = BatchNormalization(name="bn_lstm")(lstm_out)
#     act_lstm = Activation("tanh", name="tanh_lstm")(bn_lstm)
#
#     # --- LSTM后的最大池化层 ---
#     pool_after_lstm = MaxPooling1D(pool_size=2, strides=2, name="maxpool_after_lstm")(act_lstm)
#
#     # --- 优化的残差网络模块 ---
#     # 输入维度是 (batch, steps, 128)，我们将ResNet滤波器设为128
#     resnet_out = resnet_module(pool_after_lstm, num_filters=128)  # 256 -> 128
#
#     # --- 优化的最终分类头 ---
#     bn_head = BatchNormalization(name="bn_head")(resnet_out)
#     act_head = Activation("selu", name="selu_head")(bn_head)
#     gap = GlobalAveragePooling1D(name="global_avg_pool")(act_head)
#     flatten = Flatten(name="flatten_layer")(gap)
#
#     # --- 【关键优化】大幅缩减全连接层 ---
#     dense1 = Dense(units=64, activation="relu", name="dense_1")(flatten)  # 4096 -> 512
#     dense2 = Dense(units=32, activation="relu", name="dense_2")(dense1)
#     dense3 = Dense(units=16, activation="relu", name="dense_2")(dense2)
#     # dense2 = Dense(units=9, activation="relu", name="dense_2")(dense1)
#
#     # dropout_output = Dropout(0.4, name="dropout_layer")(dense2)  # Dropout率可根据实际情况调整
#
#     # --- 【必须】修改输出层为9类 ---
#     output_layer = Dense(units=num_classes, activation="softmax", name="output_layer")(dense3)
#
#     # 创建模型
#     model = Model(inputs=input_layer, outputs=output_layer, name="Optimized_CLRM_Model")
#
#     return model




matplotlib.rc("font", family='SimSun')


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



def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments
        # if epoch_count <= 10:
        #
        #     y_pred = y_pred + 1 * adjustments * 1
        #
        # elif 10 < epoch_count < 36:
        #
        #     y_pred = y_pred + 1 * adjustments * 0.7
        # elif 36 <= epoch_count <= 50:
        #
        #     y_pred = y_pred + 1 * adjustments * 0.6
    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss

def compute_adjustment_1(Y_profiling, tro, classes=9, eps=1e-12):
    """
    Logit Adjustment: add tau * log(pi) to logits (Menon et al. 2020).
    Y_profiling: one-hot 或 soft label，默认取前 classes 维并 argmax 得到类别
    tro: tau
    """
    y = np.argmax(Y_profiling[:, :classes], axis=1)
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    pi = counts / np.maximum(counts.sum(), 1.0)
    adjustments = np.log(np.power(pi, tro) + eps)  # log(pi^tau)
    return adjustments.astype(np.float32)

def make_adjustment_loss(
    classes,
    adjustments,          # shape: (classes,)
    adjust_flag=True,
    smooth_flag=True,
    lam_smooth=1e-3,      # 平滑强度：建议从 1e-4~1e-2 网格搜
    smooth_type="l2",     # "l2" or "tv"
    edge_downweight=0.5,  # 边界(0-1, 7-8)的平滑权重，防止把0/8“抹平”
    eps=1e-12
):
    adjustments = tf.constant(adjustments, dtype=tf.float32)  # (classes,)

    # 构造相邻差分权重 w_k (k=0..classes-2)
    # 默认中间为1，边界为 edge_downweight
    w = np.ones((classes - 1,), dtype=np.float32)
    if classes >= 2:
        w[0] = edge_downweight
        w[-1] = edge_downweight
    w = tf.constant(w, dtype=tf.float32)  # (classes-1,)

    def adjustment_loss_1(y_true, y_pred):
        """
        y_true: one-hot (B, classes)
        y_pred: logits (B, classes)  ——注意：这里假设模型输出 logits
        """
        y_true = y_true[:, :classes]
        logits = y_pred

        # 1) Logit Adjustment（对logits加偏置）
        if adjust_flag:
            logits = logits + adjustments  # broadcast to (B, classes)

        # 2) softmax
        p = tf.nn.softmax(logits, axis=1)

        # 3) 基础交叉熵
        ce = tk.backend.categorical_crossentropy(y_true, p)

        # 4) 相邻平滑正则（表达 HW 相邻相关）
        if smooth_flag:
            diff = p[:, 1:] - p[:, :-1]     # (B, classes-1)

            if smooth_type.lower() == "tv":
                smooth = tf.reduce_sum(w * tf.abs(diff), axis=1)     # (B,)
            else:  # "l2"
                smooth = tf.reduce_sum(w * tf.square(diff), axis=1)  # (B,)

            loss = ce + lam_smooth * smooth
        else:
            loss = ce

        return tf.reduce_mean(loss)

    return adjustment_loss_1


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
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

    adjustments = compute_adjustment_1(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    loss_fn = make_adjustment_loss(
        classes=classes,
        adjustments=adjustments,
        adjust_flag=True,
        smooth_flag=True,
        lam_smooth=1e-2,
        smooth_type="tv",  # 或 "tv"
        edge_downweight=0.2
    )

    """创建神经网络模型"""
    """ select the output_metric function """

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       loss_fn, learning_rate, model=attack_model,
                                                       model_size=model_size)
    # model = build_optimized_clrm_model(input_shape=(700, 1), num_classes=9)
    # model.compile(loss=adjustment_loss, optimizer='adam', metrics=None)
    # model.summary()

    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
         lr_manager]
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
    predictions = softmax(predictions,1)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    log = open('C:\paper_final_revise\ce-la-smooth.txt', mode='a',
               encoding='utf-8')

    print(file=log)
    print(file=log)

    current_time_mid = datetime.now()
    day_mid = current_time_mid.day
    hour_mid = current_time_mid.hour
    minute_mid = current_time_mid.minute
    experiment_time_mid = 'time_is{}_{}_{}'.format(int(day_mid),
                                                   int(hour_mid),
                                                   int(minute_mid)
                                                   )

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
    print(experiment_time_mid, file=log)
    print(experiment_time_end, file=log)
    print("learning_rate_", adjustment_loss,
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

    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')