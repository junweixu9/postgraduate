from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import DL_model
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')
# 设置字体

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Cropping1D
from tensorflow.keras.models import Model

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
            X_attack_valid_metric, all_valid_plt_attack_metric, Y_attack_valid, Y_profiling = self.validation[0], \
                self.validation[1], \
                self.validation[2], self.validation[3]

            # Y_profiling = Y_profiling[:50000, :9]
            # Y_profiling_int = tf.argmax(Y_profiling, 1)
            # y_pred_train_metric = self.model.predict(X_profiling)
            # y_pred_train_metric = tf.nn.softmax(y_pred_train_metric, 1)
            # y_pred_train_metric_int = tf.argmax(y_pred_train_metric, 1)
            # train_correct = tf.equal(y_pred_train_metric_int, Y_profiling_int)
            # train_accuracy = tf.reduce_mean(tf.cast(train_correct, tf.float32))
            # print('train Accuracy:', train_accuracy)

            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            y_pred_valid_metric_int = tf.argmax(y_pred_valid_metric, 1)
            Y_attack_valid_int = tf.argmax(Y_attack_valid[:, :9], 1)

            correct = tf.equal(y_pred_valid_metric_int, Y_attack_valid_int)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('valid Accuracy:', accuracy)

            Y_true = Y_attack_valid_int.numpy()  # 真实标签
            Y_pred = y_pred_valid_metric_int.numpy()  # 预测标签

            # 计算每个标签的准确率
            labels = range(9)
            accuracies = {}

            for label in labels:
                # 真实标签为当前label的样本掩码
                mask = (Y_true == label)
                total = np.sum(mask)

                if total == 0:
                    # 处理无样本的标签（根据需求可选）
                    acc = 0.0  # 或 np.nan
                else:
                    # 统计预测正确的样本数
                    correct = np.sum(Y_pred[mask] == label)
                    acc = correct / total

                accuracies[label] = acc

            # 输出结果
            for label in labels:
                print(f"标签 {label} 的准确率: {accuracies[label]:.4f}")

            epoch_count = epoch_count + 1
            # if epoch_count % 10 == 0:

            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            # avg_attack_traces = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
            #                                     'attack_traces',
            #                                     leakage_model, dataset,
            #                                     num_traces_attacks)
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


def logit_adjustment_focal_loss(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    # print("y_pred.shape: ", y_pred.shape)
    # Clip the prediction value to prevent NaN's and Inf's
    # if adjust_flag:
    #     y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    alpha = np.array([0.25], dtype=np.float32)
    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)
    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, 2) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))


def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_true = K.clip(y_true, K.epsilon(), 1)
    if adjust_flag:
        y_pred = y_pred + 1 * adjustments
    y_pred = tf.nn.softmax(y_pred, 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def build_denoising_autoencoder(input_shape=(700, 1)):
    # 定义论文中选择的激活函数
    activation_function = 'selu'

    # --- 编码器 (Encoder) ---
    # 编码器使用卷积和池化层来压缩输入
    input_trace = Input(shape=input_shape)

    # 编码器模块 1
    # 对应表2中的 Conv1 和 MaxPooling1
    x = Conv1D(filters=4, kernel_size=10, activation=activation_function, padding='same')(input_trace)
    x = BatchNormalization()(x)  # 使用BN防止过拟合并加速训练
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # 编码器模块 2
    # 对应表2中的 Conv2 和 MaxPooling2
    x = Conv1D(filters=8, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4, padding='same')(x)

    # 编码器模块 3
    # 对应表2中的 Conv3 和 MaxPooling3
    x = Conv1D(filters=16, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling1D(pool_size=8, padding='same')(x)

    # --- 解码器 (Decoder) ---
    # 解码器使用卷积和上采样层来重建输入

    # 解码器模块 1
    # 对应表2中的 Conv4 和 UpSampling1
    x = Conv1D(filters=16, kernel_size=10, activation=activation_function, padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=8)(x)

    # 解码器模块 2
    # 对应表2中的 Conv5 和 UpSampling2
    x = Conv1D(filters=8, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=4)(x)

    # 解码器模块 3
    # 对应表2中的 Conv6 和 UpSampling3
    x = Conv1D(filters=4, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    # 输出层
    # 最后的卷积层 (Conv7) 将能量迹重建为原始尺寸
    # 使用大小为1的卷积核来生成最终的输出通道
    # 线性激活函数 (linear) 适合用于重建信号
    x = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')(x)

    decoded = Cropping1D(cropping=(2, 2))(x)  # 新的输出 (None, 700, 1)

    # --- 构建并编译模型 ---
    autoencoder = Model(input_trace, decoded)

    optimizer = tf.keras.optimizers.Adam()
    loss_function = 'mean_squared_error'

    autoencoder.compile(optimizer=optimizer, loss=loss_function)

    return autoencoder


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 800
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

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    model_denoise = build_denoising_autoencoder()

    model_denoise_root = 'E:/logit_backup/ensemble_model/datasetASCAD_leakage_modelHW_epoch100_batch_size128_outputalldenoise_ascad.h5'
    model_denoise.load_weights(model_denoise_root)

    X_profiling, X_attack = model_denoise(X_profiling), model_denoise(X_attack)

    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000], Y_profiling))
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
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    log = open('F:/result/cnn/a/logit_curve_3000.txt', mode='a',
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

    y_pred_valid_metric_int = tf.argmax(predictions, axis=-1)
    y_true_valid_metric_int = tf.argmax(Y_attack[:5000, :9], axis=-1)

    confusion_matrix = tf.math.confusion_matrix(y_true_valid_metric_int, y_pred_valid_metric_int)
    print("Confusion Matrix:")
    print(confusion_matrix)

    cm = confusion_matrix.numpy()
    plt.figure(figsize=(10, 8), dpi=120)
    ax = plt.subplot(111)

    sns.set(font_scale=1.5)

    # 使用更柔和的色板，添加边框线
    sns.heatmap(cm,
                cmap='Reds',  # 关键修改：从白到黑的渐变
                annot=True,
                fmt='d',
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
    title = "./" + experiment_time_end + ".png"
    plt.savefig(title, bbox_inches='tight', dpi=140, pad_inches=0.1)
    # plt.show()

    # def generate_image(text="程序已跑完", output_path="completed_matplotlib.png"):
    #     # 设置图像参数
    #     fig = plt.figure(figsize=(6, 3), dpi=280)  # 图像尺寸（宽600px，高300px）
    #     ax = plt.axes([0, 0, 1, 1], frameon=False)  # 全屏显示，无边框
    #     plt.axis('off')  # 隐藏坐标轴
    #
    #     # 设置背景颜色（白色）
    #     ax.set_facecolor('white')
    #
    #     # 设置中文字体（需要指定字体路径）
    #     # 中文字体路径示例（根据系统调整）：
    #     # Windows: 'C:/Windows/Fonts/simhei.ttf'
    #     # Linux: '/usr/share/fonts/truetype/arphic/uming.ttc'
    #     # macOS: '/System/Library/Fonts/Supplemental/Song.ttf'
    #     font_path = 'simhei.ttf'  # 如果使用默认字体可删除此行和下面的 FontProperties
    #
    #     # 自定义字体（中文字体需要指定路径）
    #     font_properties = {
    #         'family': 'SimHei',  # 中文字体名称（需系统支持）
    #         'size': 24,  # 字体大小
    #         'color': 'black',  # 字体颜色
    #         'weight': 'bold'  # 粗体
    #     }
    #
    #     # 如果需要指定字体路径，使用FontProperties
    #     # from matplotlib.font_manager import FontProperties
    #     # prop = FontProperties(fname=font_path)
    #     # font_properties['fontproperties'] = prop
    #
    #     # 添加文本（居中显示）
    #     ax.text(
    #         0.5,  # x坐标（0-1比例）
    #         0.5,  # y坐标（0-1比例）
    #         text,
    #         ha='center',  # 水平居中
    #         va='center',  # 垂直居中
    #         **font_properties
    #     )
    #
    #     plt.show()  # 显示图形
    #
    #
    # generate_image()
