from data_load_ascad_50000 import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import wandb
from scipy.special import comb


def CER(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 3
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = tf.keras.losses.categorical_crossentropy(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / average_fake_k_loss

    return loss_cer


class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(all, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(all, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
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
                print("攻击失败", file=log)
                print("corr:", avg_corr_current, file=log)
                print("GE:", avg_attack_traces[-1, correct_key], file=log)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("corr:", avg_corr_current)
                print("攻击成功", file=log)
                print("corr:", avg_corr_current, file=log)
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1), file=log)

def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_true_index = tf.argmax(y_true, 1)
    new_tensor_list = []

    for i in range(9):
        res = tf.reduce_sum(tf.cast(tf.equal(y_true_index, i), tf.int32))
        new_tensor_list.append(res)
    label_freq_array = new_tensor_list / tf.reduce_sum(new_tensor_list)
    adjustments = tf.math.log(label_freq_array ** tro + 1e-12)
    adjustments = tf.cast(adjustments, dtype=tf.float32)

    y_true = K.clip(y_true, K.epsilon(), 1)

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments
    #
    y_pred = tf.nn.softmax(y_pred, 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 10000
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 50
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
    tro = 0.275
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

    log = open('F:/result/ascad_rand_logit/tro0275/dynamic.txt', mode='a',
               encoding='utf-8')
    print(file=log)
    print(file=log)
    """创建神经网络模型"""

    if sigma_hw == 0 and sigma_id == 0:

        """ select the output_metric function """

        callback = [
            all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
        ]
        #
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           CER, learning_rate, model=attack_model,
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
        history = model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=128, verbose=2, epochs=11,
                            callbacks=callback
                            )
        history.history.keys()
        loss = history.history['loss']

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
              "___adjust_flag_", adjust_flag,
              "___adjust_tro_", tro, file=log)
        print("best_corr", np.max(all_corr_logs), file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "__train_loss_", loss[epoch], file=log)
        log.close()

    else:

        """ select the output_metric function """
        callback = [
            all(validation=(X_attack[:10000], plt_attack[:10000], Y_attack[:10000]))
        ]

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           kl_adjustment_loss, learning_rate, model=attack_model,
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
        history = model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
                            callbacks=callback)
        history.history.keys()
        loss = history.history['loss']

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
              "___adjust_flag_", adjust_flag,
              "___adjust_tro_", tro, file=log)
        print("best_corr", np.max(all_corr_logs), file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "__train_loss_", loss[epoch], file=log)
        log.close()

        # Attack
        # print('======Attack======')
        # predictions = model.predict(X_attack[:5000])
        # predictions = tf.nn.softmax(predictions)
        # attack_traces = perform_attacks(plt_attack[:5000], predictions, "attack_traces",
        #                                 leakage_model, dataset, num_traces_attacks)
        # if attack_traces[-1, correct_key] > 0:
        #
        #     print(file=log)
        #     print(experiment_time, file=log)
        #     print("learning_rate_", learning_rate,
        #           "___architecture_", attack_model,
        #           "___dataset_", dataset,
        #           "___epochs_", epoch,
        #           "___num_profiling_traces_", num_profiling_traces,
        #           "___num_attack_traces_", num_traces_attacks,
        #           "___sigma_hw_", sigma_hw,
        #           "___sigma_id_", sigma_id,
        #           "___adjust_flag_", adjust_flag, file=log)
        #     print("攻击失败", file=log)
        #     print("GE:", attack_traces[-1, correct_key], file=log)
        #     for epoch in range(len(loss)):
        #         print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch], "___corr_",
        #               all_corr_logs[epoch], file=log)
        #     log.close()
        # else:
        #     print(file=log)
        #     print(experiment_time, file=log)
        #     print("learning_rate_", learning_rate,
        #           "___architecture_", attack_model,
        #           "___dataset_", dataset,
        #           "___epochs_", epoch,
        #           "___num_profiling_traces_", num_profiling_traces,
        #           "___num_attack_traces_", num_traces_attacks,
        #           "___sigma_hw_", sigma_hw,
        #           "___sigma_id_", sigma_id,
        #           "___adjust_flag_", adjust_flag, file=log)
        #     print("攻击成功", file=log)
        #     print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)
        #     for epoch in range(len(loss)):
        #         print("轮数", epoch, "___train_loss_", loss[epoch], "___rank_", all_rank_logs[epoch], "___corr_",
        #               all_corr_logs[epoch], file=log)
        #     log.close()
