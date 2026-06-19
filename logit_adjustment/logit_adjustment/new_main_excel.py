from data_load import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
from datetime import datetime
import xlsxwriter as xw


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

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                print("corr:", avg_corr_current)


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


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 1  # sigma for the HW leakage model
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
    tro = 0.75
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
    bool_metric = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1]])

    Y_attack_index_prue = np.argmax(Y_attack[:5000, :9], 1)
    Y_attack_w_compute_index = bool_metric[Y_attack_index_prue]

    Y_attack_index_expand = np.expand_dims(np.argmax(Y_attack[:5000, :9], 1), axis=0)
    ziranshu_5000 = np.expand_dims(np.arange(5000), axis=0)
    Y_attack_index = np.concatenate((ziranshu_5000, Y_attack_index_expand), axis=0)

    Y_attack_index = Y_attack_index.T
    print()
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
    model.load_weights(
        r'./ASCAD_data/ASCAD_trained_models/datasetASCAD_leakage_modelHW_epoch11_batch_size32_outputall_7000.h5')

    validation_logits = model.predict(X_attack[:5000])
    y_pred = tf.nn.softmax(validation_logits, 1)

    y_pred_prob_w_sum = tf.reduce_sum(tf.multiply(y_pred, Y_attack_w_compute_index), 1)
    y_pred_prob_max = tf.reduce_max(y_pred, 1)
    y_pred_prob_max_index = tf.argmax(y_pred, 1)

    y_true_in_y_pred_prob = tf.gather_nd(y_pred, Y_attack_index)
    y_pred_prob_in_y_pred_w_sum = y_true_in_y_pred_prob/y_pred_prob_w_sum


    fileName = "F:/result/ascad_excel/7000.xlsx"
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()

    worksheet1.write_column('A1', y_pred[:, 0])
    worksheet1.write_column('B1', y_pred[:, 1])
    worksheet1.write_column('C1', y_pred[:, 2])
    worksheet1.write_column('D1', y_pred[:, 3])
    worksheet1.write_column('E1', y_pred[:, 4])
    worksheet1.write_column('F1', y_pred[:, 5])
    worksheet1.write_column('G1', y_pred[:, 6])
    worksheet1.write_column('H1', y_pred[:, 7])
    worksheet1.write_column('I1', y_pred[:, 8])
    worksheet1.write_column('J1', y_pred_prob_max)
    worksheet1.write_column('K1', y_pred_prob_max_index)
    worksheet1.write_column('L1', y_true_in_y_pred_prob)
    worksheet1.write_column('M1', Y_attack_index_prue)
    worksheet1.write_column('N1', y_pred_prob_w_sum)
    worksheet1.write_column('O1', y_pred_prob_in_y_pred_w_sum)

    workbook.close()
