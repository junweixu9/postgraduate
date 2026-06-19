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

def compute_adjustment_ldl(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling_part = np.sum(Y_profiling[:, :9], 0)
    Y_profiling_all = np.sum(Y_profiling_part)
    label_freq_array = Y_profiling_part / Y_profiling_all
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    return adjustments

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
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 1  # sigma for the HW leakage model
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
    tro = 1.25
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

    adjustments = compute_adjustment_ldl(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    """创建神经网络模型"""
    """ select the output_metric function """

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       kl_adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)

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
    log = open('C:\paper_final_revise\ldl-la.txt', mode='a',
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
    print("learning_rate_", kl_adjustment_loss,
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