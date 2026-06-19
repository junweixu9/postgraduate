from tensorflow.keras.optimizers import Adam
from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
from datetime import datetime
from clr import OneCycleLR
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import requests
import time
import tensorflow.keras as tk

def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""
    N = Y_profiling.shape[0]
    Y_profiling = np.sum(Y_profiling, 0)
    label_freq_array = Y_profiling / N
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    return adjustments

def build_inception_sca_model_ches(length):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    # FC-8, sigmoid (Output)
    output_layer = Dense(8, name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')

    return model

def build_inception_sca_model_rand(length):
    img_input = Input(shape=(length, 1))
    x = Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)

    # FC-8, sigmoid (Output)
    output_layer = Dense(8, name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')


    return model

def adjustment_loss(y_true, y_pred):

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return loss


if __name__ == '__main__':

    """变量配置"""
    dataset = 'CHES_CTF'  # ASCAD/ASCAD_rand/CHES_CTF
    attack_model = 'CNN'  # MLP/CNN
    num_traces_attacks = 2000
    batch_size = 128
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
    adjust_flag = True

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

    tro = 0

    adjustments = compute_adjustment(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    model = build_inception_sca_model_ches(X_profiling.shape[1])
    optimizer = Adam(learning_rate=5e-3)  # Or any optimizer of your choice
    model.compile(optimizer=optimizer, loss=adjustment_loss)


    X_profiling = tf.cast(X_profiling, dtype=tf.double)
    Y_profiling = tf.cast(Y_profiling, dtype=tf.double)


    model.summary()
    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [lr_manager
                ]
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
    history = model.fit(x=X_profiling, y=Y_profiling, batch_size=batch_size, verbose=2,
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
    import time

    # 记录开始时间
    start_time = time.perf_counter()
    predictions = model.predict(X_attack[:5000])

    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")



    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack[:5000], predictions, "attack_traces",
                                    "HW", dataset, num_traces_attacks)

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

    current_time_end = datetime.now()
    day_end = current_time_end.day
    hour_end = current_time_end.hour
    minute_end = current_time_end.minute
    experiment_time_end = 'time_is{}_{}_{}'.format(int(day_end),
                                                   int(hour_end),
                                                   int(minute_end)
                                                   )
    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')