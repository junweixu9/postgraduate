import time

from data_load_ascad_50000_50 import read_data
from SCA_util import perform_attacks
from datetime import datetime
from clr import OneCycleLR
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import requests

def build_nmax_50_model(length):
    INIT = 'he_uniform'
    ACT = 'selu'
    PAD = 'same'
    """
    复现表格第二列: NN architecture when Nmax = 50 (8 weights layers)
    """
    img_input = Input(shape=(length, 1))

    # --- Inception Block ---
    b1 = Conv1D(filters=4, kernel_size=1, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)
    b2 = Conv1D(filters=4, kernel_size=7, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)
    b3 = Conv1D(filters=4, kernel_size=11, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)
    x = Concatenate()([b1, b2, b3])

    # --- Layers ---
    # conv1-8, selu
    x = Conv1D(filters=8, kernel_size=1, kernel_initializer=INIT, activation=ACT, padding=PAD)(x)
    # BN
    x = BatchNormalization()(x)
    # average pooling, 2 by 2
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    # conv25-16, selu
    x = Conv1D(filters=16, kernel_size=25, kernel_initializer=INIT, activation=ACT, padding=PAD)(x)
    # BN
    x = BatchNormalization()(x)
    # average pooling, 25 by 25
    x = AveragePooling1D(pool_size=25, strides=25)(x)

    # conv3-32, selu
    x = Conv1D(filters=32, kernel_size=3, kernel_initializer=INIT, activation=ACT, padding=PAD)(x)
    # BN
    x = BatchNormalization()(x)
    # average pooling, 4 by 4
    x = AveragePooling1D(pool_size=4, strides=4)(x)

    # flatten
    x = Flatten()(x)

    # 3 layers of FC-15, selu
    x = Dense(15, kernel_initializer=INIT, activation=ACT)(x)
    x = Dense(15, kernel_initializer=INIT, activation=ACT)(x)
    x = Dense(15, kernel_initializer=INIT, activation=ACT)(x)

    # FC-8, sigmoid
    output_layer = Dense(8, activation='sigmoid', name='output_nmax_50')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_50')
    return model


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
            avg_rank_current, avg_attack_traces= perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    "HW", dataset,
                                                                                    num_traces_attacks)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                # print("corr:", avg_corr_current)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                # print("corr:", avg_corr_current)


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    attack_model = 'CNN'  # MLP/CNN
    num_traces_attacks = 5000
    batch_size = 256
    epoch_count = 1
    # Select leakage model
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
        "HW",
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        0, 0)

    """创建神经网络模型"""
    """ select the output_metric function """

    model = build_nmax_50_model(X_profiling.shape[1])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
    )

    model.summary()
    lr_manager = OneCycleLR(len(X_profiling[45000:]), 256, 5e-3, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None,
                            minimum_momentum=None, verbose=True)

    callback = [ lr_manager]
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, "HW",
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    start_time = time.perf_counter()
    history = model.fit(x=X_profiling[:45000], y=Y_profiling[:45000, :9], batch_size=batch_size, verbose=2,
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
    predictions = model.predict(X_attack[:10000])
    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")
    # predictions = predictions - 0.25 * adjustments
    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack[:10000], predictions, "attack_traces",
                                    "HW", dataset, 10000)

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