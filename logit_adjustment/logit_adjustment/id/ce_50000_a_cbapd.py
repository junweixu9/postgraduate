from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
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
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, AveragePooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def create_cbapd_model(input_shape=(700, 1)):
    trace_input = Input(shape=input_shape, name='trace_input')

    x = Conv1D(filters=64, kernel_size=11,padding='same', activation=None)(trace_input)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    # Block 2
    # Subsequent convolutional layers double the number of filters.
    x = Conv1D(filters=128, kernel_size=11, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    # Block 3
    x = Conv1D(filters=256, kernel_size=11, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    # Block 4
    x = Conv1D(filters=512, kernel_size=11, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(filters=512, kernel_size=11, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    output = Dense(256, name='output')(x)

    # Create and compile the model
    model = Model(inputs=trace_input, outputs=output, name='CBAPD_Model')

    # The paper specifies using the RMSprop optimizer with a learning rate of 0.00001
    # and the categorical crossentropy loss function.
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
    model.compile(optimizer=optimizer,
                  loss=adjustment_loss,
                  metrics=None)

    return model


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

            Y_profiling = Y_profiling[:50000, :256]
            y_pred_train_metric = self.model.predict(X_profiling)
            y_pred_train_metric = tf.nn.softmax(y_pred_train_metric, 1)
            train_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_profiling, y_pred_train_metric))
            print('train Accuracy:', train_accuracy)

            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            valid_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_attack_valid, y_pred_valid_metric))
            print('valid_accuracy:', valid_accuracy)


            epoch_count = epoch_count + 1

            # if epoch_count % 20 == 0:

            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
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
                    if count == 95:
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

    return adjustments


def compute_adjustment_ldl(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling_part = np.sum(Y_profiling[:,:256], 0)
    Y_profiling_all = np.sum(Y_profiling_part)
    label_freq_array = Y_profiling_part / Y_profiling_all
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    return adjustments


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss


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
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'ID'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 1000
    epoch_count = 0
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 100
    learning_rate = 0.00001
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    rank_logs = []
    batch_size = 200
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

    adjustments = compute_adjustment_ldl(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    """创建神经网络模型"""
    """ select the output_metric function """

    model = create_cbapd_model(input_shape=(700, 1))

    model.summary()

    lr_manager = OneCycleLR(len(X_attack[:5000]), 200, learning_rate, end_percentage=0.05, scale_percentage=0.11,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        # lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000], Y_profiling))
         all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000], Y_profiling))
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
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :256], batch_size=200, verbose=2,
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
    y_true_valid_metric_int = tf.argmax(Y_attack[:5000, :256], axis=-1)

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
    title = "./"+experiment_time_end+".png"
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
