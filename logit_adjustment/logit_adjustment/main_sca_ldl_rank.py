from data_load_rank import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
from datetime import datetime


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
            avg_rank_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric, 'rank',
                                               leakage_model, dataset, num_traces_attacks)
            logs['rank_val'] = avg_rank_current
            all_rank_logs.append(avg_rank_current)
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
            return logs['corr_val']


def kl_loss_with_rank_loss(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred, 1)
    rank_loss = rank_ldl_loss(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1)
    y_true = K.clip(y_true, K.epsilon(), 1)
    kl_loss = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    return kl_loss - 0.5 * rank_loss + 0.01 * tf.reduce_sum(model.losses)


def kl_loss_with_rank_loss_metrix(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred, 1)
    rank_loss = rank_ldl_loss_metrix(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1)
    y_true = K.clip(y_true, K.epsilon(), 1)
    kl_loss = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    return kl_loss - rank_lamb * rank_loss


def rank_ldl_loss_metrix(y_true, y_pred):
    #  给每一个样本赋上puv矩阵和（du-dv）^2矩阵

    y_true_index = tf.argmax(y_true, 1)

    square_diff_puv_container = tf.gather(square_diff_puv_container_representation, y_true_index)
    rank_puv_container = tf.gather(rank_puv_container_representation, y_true_index)

    # 给每一个样本计算p^uv矩阵：首先创建ijk矩阵和i_j_k矩阵
    ijk = tf.repeat(tf.expand_dims(y_pred, 1), repeats=9, axis=1)

    i_j_k = tf.transpose(ijk, perm=[0, 2, 1])

    predict_puv_container = tf.cast(tf.math.reciprocal(1 + tf.math.exp(sigma_square * (i_j_k - ijk))), dtype=tf.float32)

    # 根据rank_loss计算总值：二项损失*平方差
    binary_puv_predict_uv = tf.multiply(rank_puv_container, tf.math.log(predict_puv_container)) + tf.multiply(
        1 - rank_puv_container, tf.math.log(1 - predict_puv_container))

    individual_9_9_rank_loss = tf.multiply(binary_puv_predict_uv,
                                           square_diff_puv_container)

    mean_all_9_9_rank_loss = tf.reduce_mean(individual_9_9_rank_loss)

    return mean_all_9_9_rank_loss


def rank_ldl_loss(y_true, y_pred):
    store_loss = tf.constant(0, dtype=tf.float32)

    y_true = tf.argmax(y_true, 1)
    for index in range(len(y_pred)):
        ijk = tf.repeat(tf.expand_dims(y_pred[index], 0), repeats=9, axis=0)
        i_j_k = tf.transpose(ijk)
        predict_puv_container = tf.cast(tf.math.reciprocal(1 + tf.math.exp(-20 * (i_j_k - ijk))), dtype=tf.float32)
        # for ordinate in range(classes):
        #     for abscissa in range(classes):
        # 计算每个样本的p^uv
        # predict_puv_container = tf.tensor_scatter_nd_update(predict_puv_container, [[abscissa, ordinate]],
        #                                                     [1 / 1 + tf.exp(-100 * (
        #                                                             y_pred[index, abscissa] - y_pred[
        #                                                         index, ordinate]))])

        # 计算每个样本的rank loss = 二项式损失*（du-dv）^2
        label = tf.squeeze(tf.gather(y_true, indices=[index]))
        binary_puv_predict_uv = tf.multiply(rank_puv_container_representation[label, :, :],
                                            tf.math.log(predict_puv_container)) + tf.multiply(
            1 - rank_puv_container_representation[label, :, :],
            tf.math.log(1 - predict_puv_container))

        individual_9_9_rank_loss = tf.multiply(binary_puv_predict_uv,
                                               square_diff_puv_container_representation[label, :, :])

        # 每一个样本计算 mean_rank loss
        mean_individual_9_9_rank_loss = tf.reduce_mean(individual_9_9_rank_loss)

        # 将所有样本的mean_rank loss储存下来
        store_loss = store_loss + mean_individual_9_9_rank_loss

    # 将所有样本的mean_rank loss做平均
    maen_store_loss = store_loss / tf.cast(len(y_pred), dtype=tf.float32)

    return maen_store_loss


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
    epoch = 13
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
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    rank_lamb = 0.5
    sigma_square = -15
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling,
        plt_attack), correct_key, attack_byte, num_profiling_traces, rank_puv_container_representation, square_diff_puv_container_representation = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    """创建神经网络模型"""

    rank_puv_container_representation = tf.cast(rank_puv_container_representation, dtype=tf.float32)
    square_diff_puv_container_representation = tf.cast(square_diff_puv_container_representation, dtype=tf.float32)

    """ select the output_metric function """
    callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                ]

    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       kl_loss_with_rank_loss_metrix, learning_rate, model=attack_model,
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
    history = model.fit(x=X_profiling, y=Y_profiling, batch_size=batch_size, verbose=2, epochs=epoch,
                        callbacks=callback)
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    predictions = model.predict(X_attack[5000:])
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    if attack_traces[-1, correct_key] > 0:
        log = open('C:/Users/Administrator/OneDrive/Desktop/experiment5.txt', mode='a', encoding='utf-8')
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
              "___rank_lamb_", rank_lamb, file=log)
        print("攻击失败", file=log)
        print("GE:", attack_traces[-1, correct_key], file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "__train_loss_", loss[epoch], "__rank_", all_rank_logs[epoch],
                  "__corr_", all_corr_logs[epoch], file=log)
        log.close()

    else:
        log = open('C:/Users/Administrator/OneDrive/Desktop/experiment5.txt', mode='a', encoding='utf-8')
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
              "___rank_lamb_", rank_lamb, file=log)
        print("攻击成功", file=log)
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)
        for epoch in range(len(loss)):
            print("轮数", epoch, "___train_loss_", loss[epoch], "___rank_", all_rank_logs[epoch],
                  "___corr_", all_corr_logs[epoch], file=log)
        log.close()