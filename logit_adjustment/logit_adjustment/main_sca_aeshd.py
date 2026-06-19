from data_load import read_data
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

            l_true = tf.argmax(Y_valid_metric[:, :9], 1)
            l_true = l_true.numpy()

            target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',
                            'class 8']
            print(classification_report(l_true, l_pred, target_names=target_names))


class rank_min(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(rank_min, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(rank_min, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            logs['rank_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_rank_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric, 'rank',
                                               leakage_model, dataset, num_traces_attacks)
            logs['rank_val'] = avg_rank_current
            if not rank_logs:
                rank_logs.append(avg_rank_current)
                best_weights = self.model.get_weights()
            else:
                if rank_logs[-1] > avg_rank_current:
                    rank_logs.append(avg_rank_current)
                    best_weights = self.model.get_weights()
                    count = 0
                else:
                    count = count + 1
                    print(count)
                    if avg_rank_current == 0:
                        best_weights = self.model.get_weights()
                    if count == 10:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)
            return logs['rank_val']


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
            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_rank_current, avg_attack_traces = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
                                                                  'rank',
                                                                  leakage_model, dataset, num_traces_attacks)
            if avg_attack_traces[-1, correct_key] > 0:
                print("验证集攻击失败")
                print("验证集GE:", avg_attack_traces[-1, correct_key])
            else:
                print("验证集攻击成功")
                print("验证集TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
            logs['rank_val'] = avg_rank_current
            all_rank_logs.append(avg_rank_current)

            predictions = self.model.predict(X_attack[5000:])
            predictions = tf.nn.softmax(predictions)
            attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                            leakage_model, dataset, num_traces_attacks)
            if attack_traces[-1, correct_key] > 0:
                print("测试集攻击失败")
                print("测试集GE:", attack_traces[-1, correct_key])
            else:
                print("测试集攻击成功")
                print("测试集TGE0:", np.argmax(attack_traces[:, correct_key] < 1))

            return logs['rank_val']

class corr(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(corr, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(corr, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            logs['corr_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_corr_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
                                               'corr',
                                               leakage_model, dataset, num_traces_attacks)

            logs['corr_val'] = avg_corr_current
            return logs['corr_val']


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
                    if count == 5:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)
                        print()
            return logs['corr_val']


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

    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred) + tf.reduce_sum(model.losses)
    return loss


def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    y_true = K.clip(y_true, K.epsilon(), 1)
    if adjust_flag:
        y_pred = y_pred + 1 * adjustments
    y_pred = tf.nn.softmax(y_pred, 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


# def kl_loss_model(y_true, y_pred):
#     y_pred = tf.nn.softmax(y_pred, 1)
#     f = K.clip(binomial_distribution, K.epsilon(), 1)
#     f = tf.cast(f, tf.double)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
#     y_pred = tf.cast(y_pred, tf.double)
#     return tf.keras.losses.KLDivergence()(f, y_pred)


class adjustment_loss_metric(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(adjustment_loss_metric, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(adjustment_loss_metric, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            logs['loss_val'] = float('inf')
            X_attack_valid_metric, all_valid_true = self.validation[0], self.validation[2]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = tf.cast(y_pred_valid_metric, dtype=tf.double)
            y_pred_valid_metric = y_pred_valid_metric + 1 * adjustments_valid
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            loss = tk.backend.categorical_crossentropy(all_valid_true[:, :9], y_pred_valid_metric)

            loss_logs.append(tf.reduce_mean(loss))
            # 加了正则化损失
            # loss = loss + tf.reduce_sum(model.losses)

            logs['loss_val'] = loss
            return logs['loss_val']


# class kl_loss_valid(tf.keras.callbacks.Callback):
#     def __init__(self, validation=None):
#         super(kl_loss_valid, self).__init__()
#         self.validation = validation
#
#     def set_params(self, params):
#         super(kl_loss_valid, self).set_params(params)
#
#     def on_epoch_end(self, epoch, logs=None):
#         if self.validation:
#             logs['kl_loss_model_val'] = float('inf')
#             X_attack_valid_metric, all_valid_true = self.validation[0], self.validation[2]
#             y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
#             y_pred_valid_metric = tf.cast(y_pred_valid_metric, dtype=tf.double)
#             y_pred_valid_metric = y_pred_valid_metric + 1 * adjustments_valid
#             y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
#             loss = kl_loss_model(all_valid_true[:, :9], y_pred_valid_metric)
#
#             kl_loss_logs.append(tf.reduce_mean(loss))
#             # 加了正则化损失
#             # loss = loss + tf.reduce_sum(model.losses)
#
#             logs['kl_loss_model_val'] = loss
#             return logs['kl_loss_model_val']


if __name__ == '__main__':

    """变量配置"""
    dataset = 'AESHD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
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
    if sigma_hw == 0 and sigma_id == 0:

        """ select the output_metric function """

        callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    # corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    # acc(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                    # adjustment_loss_metric(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    # kl_loss_valid(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                    ]

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
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
        history = model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=15,
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
        if attack_traces[-1, correct_key] > 0:
            log = open('C:/Users/Administrator/OneDrive/Desktop/experiment4.txt', mode='a', encoding='utf-8')
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
                  "___adjust_flag_", adjust_flag, file=log)
            print("攻击失败", file=log)
            print("GE:", attack_traces[-1, correct_key], file=log)
            for epoch in range(len(loss)):
                print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch],
                      "__corr_", all_corr_logs[epoch], file=log)
            log.close()
        else:
            log = open('C:/Users/Administrator/OneDrive/Desktop/experiment4.txt', mode='a', encoding='utf-8')
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
                  "___adjust_flag_", adjust_flag, file=log)
            print("攻击成功", file=log)
            print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)
            for epoch in range(len(loss)):
                print("轮数", epoch, "___train_loss_", loss[epoch], "___rank_", all_rank_logs[epoch],
                      "___corr_", all_corr_logs[epoch], file=log)
            log.close()

        # wandb.finish()
    else:

        """ select the output_metric function """
        callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
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

        # Attack
        print('======Attack======')
        predictions = model.predict(X_attack[5000:])
        # predictions = predictions - 0.25 * adjustments
        predictions = tf.nn.softmax(predictions)
        attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                        leakage_model, dataset, num_traces_attacks)
        if attack_traces[-1, correct_key] > 0:

            log = open('C:/Users/Administrator/OneDrive/Desktop/experiment4.txt', mode='a', encoding='utf-8')
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
                  "___adjust_flag_", adjust_flag, file=log)
            print("攻击失败", file=log)
            print("GE:", attack_traces[-1, correct_key], file=log)
            for epoch in range(len(loss)):
                print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch],
                      "__corr_", all_corr_logs[epoch], file=log)
            log.close()
        else:
            log = open('C:/Users/Administrator/OneDrive/Desktop/experiment4.txt', mode='a', encoding='utf-8')
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
                  "___adjust_flag_", adjust_flag, file=log)
            print("攻击成功", file=log)
            print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)
            for epoch in range(len(loss)):
                print("轮数", epoch, "___train_loss_", loss[epoch], "___rank_", all_rank_logs[epoch],
                      "___corr_", all_corr_logs[epoch], file=log)
            log.close()
