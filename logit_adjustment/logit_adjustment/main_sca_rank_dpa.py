import scipy.stats as ss
from data_load_dpa import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from datetime import datetime

from sklearn.metrics import classification_report


class rank(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(rank, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(rank, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            logs['rank_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_rank_current, avg_attack_traces = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
                                                                  'rank',
                                                                  leakage_model, dataset, num_traces_attacks)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败",)
                print("GE:", avg_attack_traces[-1, correct_key])
                print("攻击失败", file=log)
                print("GE:", avg_attack_traces[-1, correct_key], file=log)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("攻击成功", file=log)
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1), file=log)
            logs['rank_val'] = avg_rank_current
            all_rank_logs.append(avg_rank_current)

            return logs['rank_val']


class corr_max(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(corr_max, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(corr_max, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            logs['corr_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            # y_pred_valid_metric = y_pred_valid_metric - 0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_corr_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
                                               'corr',
                                               leakage_model, dataset, num_traces_attacks)
            all_corr_logs.append(avg_corr_current)

            logs['corr_val'] = avg_corr_current

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
                    if count == 6:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)
            return logs['corr_val']


class acc(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(acc, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(acc, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            X_valid_metric, Y_valid_metric = self.validation[0], self.validation[2]
            y_pred_valid_metric = self.model.predict(X_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            l_pred = tf.argmax(y_pred_valid_metric, 1)
            l_pred = l_pred.numpy()

            l_true = Y_valid_metric

            target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',
                            'class 8']
            print(classification_report(l_true, l_pred, target_names=target_names))


def kl_loss_with_dpa_var_loss(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred, 1)

    # dpa
    dpa_loss = dpa_ldl_loss(y_true, y_pred)

    # var
    _, var_true = tf.nn.moments(y_true, axes=[1])
    # _, var_pred = tf.nn.moments(y_pred, axes=[1])
    var_pred_non_sum = tf.square(y_pred - 1 / 9)
    var_pred = tf.reduce_mean(var_pred_non_sum, axis=1)
    diff_square = tf.square(var_true - var_pred)
    diff_square_sum = tf.reduce_sum(diff_square, axis=0)

    # kl
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    y_true = K.clip(y_true, K.epsilon(), 1)
    kl_loss = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    return kl_loss - dpa_lamb * dpa_loss + sigma_square * diff_square_sum + 0.01 * tf.reduce_sum(model.losses)
    # return kl_loss


def dpa_ldl_loss(y_true, y_pred):
    y_true_index = tf.argmax(y_true, 1)
    index_ascending_container = tf.gather(index_ascending_container_representation, y_true_index)
    dpa_socore = tf.reduce_sum(tf.multiply(y_pred, index_ascending_container)) / 9

    return dpa_socore


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD_rand'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 2  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 5000
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 15
    learning_rate = 0.0005
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
    dpa_lamb = 0.001
    sigma_square = 10
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling_Q, Y_attack), (
        plt_profiling,
        plt_attack), correct_key, attack_byte, num_profiling_traces, index_ascending_container_representation = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    index_ascending_container_representation = tf.cast(index_ascending_container_representation, dtype=tf.float32)
    """创建神经网络模型"""
    log = open('C:/Users/Administrator/OneDrive/Desktop/result/experiment_dpa.txt', mode='a', encoding='utf-8')
    """ select the output_metric function """
    callback = [
                corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                ]

    # callback = [rank(validation=(X_attack[5000:], plt_attack[5000:], Y_attack[5000:])),
    #             # corr_max(validation=(X_attack[5000:], plt_attack[5000:], Y_attack[5000:]))
    #             ]

    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       kl_loss_with_dpa_var_loss, learning_rate, model=attack_model,
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
    history = model.fit(x=X_profiling, y=Y_profiling_Q, batch_size=batch_size, verbose=2, epochs=epoch,
                        callbacks=callback)
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    predictions = model.predict(X_attack[:5000])
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[:5000], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    if attack_traces[-1, correct_key] > 0:

        print(file=log)
        print(experiment_time, file=log)
        print("learning_rate_", learning_rate,
              "___architecture_", attack_model,
              "___dataset_", dataset,
              "___epochs_", epoch,
              "___num_profiling_traces_", num_profiling_traces,
              "___num_attack_traces_", num_traces_attacks,
              "___sigma_hw_", sigma_hw,
              "___sigma_id_", sigma_id,
              "___sigma_square_", sigma_square,
              "___dpa_lamb_", dpa_lamb, file=log)
        print("攻击失败", file=log)
        print("GE:", attack_traces[-1, correct_key], file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch], file=log)
        log.close()

    else:
        print(file=log)
        print(experiment_time, file=log)
        print("learning_rate_", learning_rate,
              "___architecture_", attack_model,
              "___dataset_", dataset,
              "___epochs_", epoch,
              "___num_profiling_traces_", num_profiling_traces,
              "___num_attack_traces_", num_traces_attacks,
              "___sigma_hw_", sigma_hw,
              "___sigma_id_", sigma_id,
              "___sigma_square_", sigma_square,
              "___dpa_lamb_", dpa_lamb, file=log)
        print("攻击成功", file=log)
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch], file=log)
            # print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch],
            #       "__corr_", all_corr_logs[epoch], file=log)
        log.close()
