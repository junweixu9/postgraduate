import time

import requests
from data_load_ascad_50000 import read_data
from SCA_util_knl_new import perform_attacks
import DL_model
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR
import numpy as np
from tensorflow.keras import backend as K

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

            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)

            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            # avg_attack_traces = perform_attacks(all_valid_plt_attack_metric,y_pred_valid_metric,
            #                                                                         'attack_traces',
            #                                                                         leakage_model, dataset,
            #                                                                         num_traces_attacks)
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
                    if count == 8:
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
    adjustments = np.log(label_freq_array + 1e-12)

    return adjustments

def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)

    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss


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

    """创建神经网络模型"""
    """ select the output_metric function """

    #
    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       adjustment_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    # callback = [
    #     lr_manager
    # ]
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
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :9], batch_size=128, verbose=2,
                        epochs=epoch,
                        )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time} 秒")

    history.history.keys()
    loss = history.history['loss']

    start_time = time.perf_counter()
    predictions = model.predict(X_attack[5000:])
    predictions = tf.nn.softmax(predictions)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")

    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"密钥恢复: {execution_time} 秒")


    print("\n############### Predictions Done #################\n")

    print("\n############### Starting Attack on Test Set #################\n")
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))

    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')



