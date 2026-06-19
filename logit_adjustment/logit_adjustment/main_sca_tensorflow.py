from data_load_tensorflow import read_data
from Util.SCA_util_tensorflow import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K


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
                                               leakage_model, dataset, nb_traces_attacks)
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
            avg_rank_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric, 'rank',
                                               leakage_model, dataset, nb_traces_attacks)
            logs['rank_val'] = avg_rank_current
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
                                               leakage_model, dataset, nb_traces_attacks)

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
                                               leakage_model, dataset, nb_traces_attacks)

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
            return logs['corr_val']


def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""

    # 使用 TensorFlow 操作代替 NumPy 操作
    Y_profiling = tf.cast(tf.argmax(Y_profiling[:, :9], axis=1), tf.int32)
    label_freq = tf.math.bincount(Y_profiling, minlength=9, maxlength=9, dtype=tf.int32)
    label_freq_array = label_freq / tf.reduce_sum(label_freq)
    adjustments = tf.cast(tf.math.log(label_freq_array ** tro + 1e-12), tf.float32)

    return adjustments


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    if adjust_flag:
        y_pred = y_pred + 0.125 * adjustments
    y_pred = tf.nn.softmax(y_pred, 1)
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


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
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            loss = tk.backend.categorical_crossentropy(all_valid_true[:, :9], y_pred_valid_metric)

            # 加了正则化损失
            # loss = loss + tf.reduce_sum(model.losses)

            logs['loss_val'] = loss
            return logs['loss_val']


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    nb_traces_attacks = 5000
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 10
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all
    model_size = 64  # the size of the profiling model
    rank_logs = []
    corr_logs = []
    loss_logs = []
    count = 0
    best_weights = None
    tro = 1
    adjust_flag = True

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key, attack_byte = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    adjustments = compute_adjustment(Y_profiling, tro)

    """创建神经网络模型"""
    if sigma_hw == 0 and sigma_id == 0:

        """ select the output_metric function """

        callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    adjustment_loss_metric(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, model=attack_model,
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
        model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
                  callbacks=callback)
        # Attack
        print('======Attack======')
        predictions = model.predict(X_attack[5000:])
        predictions = tf.nn.softmax(predictions)
        attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                        leakage_model, dataset, nb_traces_attacks)
        if attack_traces[-1, correct_key] > 0:
            print("攻击失败")
            print("GE:", attack_traces[-1, correct_key])

        else:
            print("攻击成功")
            print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
    else:

        """ select the output_metric function """
        callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                    corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                    ]

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           kl_adjustment_loss, model=attack_model,
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
        model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
                  callbacks=callback)
        # Attack
        print('======Attack======')
        predictions = model.predict(X_attack[5000:])
        # predictions = predictions - 0.25 * adjustments
        predictions = tf.nn.softmax(predictions)
        attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                        leakage_model, dataset, nb_traces_attacks)
        if attack_traces[-1, correct_key] > 0:
            print("攻击失败")
            print("GE:", attack_traces[-1, correct_key])

        else:
            print("攻击成功")
            print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
