from data_load_ascad_1000 import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
from datetime import datetime


class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(all, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(all, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            logs['all_val'] = float('inf')
            global best_weights
            global count
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
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

def kl_cb_focal_factor_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_pred = tf.nn.softmax(y_pred, 1)

    focal_factor = tf.pow(1 - y_pred, gamma)

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)

    return K.sum(tf.multiply(y_true * K.log(y_true / y_pred), focal_factor), axis=-1)


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
    epoch = 11
    gamma = 1.25  # 0.5,1.0,2.0
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
    log = open('F:/result/focal/ar/1000.txt', mode='a', encoding='utf-8')
    best_weights = None
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

    """ select the output_metric function """
    callback = [all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                ]

    model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                       companion_metric,
                                                       kl_cb_focal_factor_loss, learning_rate, model=attack_model,
                                                       model_size=model_size)
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = './ASCAD_data/ASCAD_trained_models/' + test_info + '.h5'

    """开始训练"""
    history = model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
                        callbacks=callback)
    # model.save(model_root)
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
          "gamma", gamma, file=log)
    print("best_corr", np.max(all_corr_logs), file=log)
    for epoch in range(len(loss)):
        print("轮数", epoch, "__train_loss_", loss[epoch], file=log)
    log.close()
