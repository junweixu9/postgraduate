from data_load import read_data
from Util.SCA_util import perform_attacks
import Util.DL_model as DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
import xlsxwriter as xw
from datetime import datetime
from tensorflow.keras.models import Model


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    # if adjust_flag:
    #     y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss


def kl_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    y_true = K.clip(y_true, K.epsilon(), 1)
    # if adjust_flag:
    #     y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


if __name__ == '__main__':
    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    # Select leakage model
    classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 11
    learning_rate = 0.0005
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    best_weights = None

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)
    Y_attack_index = np.expand_dims(np.argmax(Y_attack[:5000, :9], 1), axis=0)
    ziranshu_5000 = np.expand_dims(np.arange(5000), axis=0)
    Y_attack_index = np.concatenate((ziranshu_5000, Y_attack_index), axis=0)
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
        r'./ASCAD_data/ASCAD_trained_models/datasetASCAD_leakage_modelHW_epoch11_batch_size32_outputall_3000.h5')

    validation_logits = model.predict(X_attack[:5000])
    y_pred = tf.nn.softmax(validation_logits, 1)
    y_pred_prob_max = tf.reduce_max(y_pred, 1)
    y_pred_prob_max_index = tf.argmax(y_pred, 1)

    y_true_prob = tf.gather_nd(y_pred, Y_attack_index)  # 1
    print()

    fileName = "C:/Users/hp/Desktop/7000.xlsx"
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
    worksheet1.write_column('L1', y_true_prob)
    worksheet1.write_column('M1', Y_attack_index)

    workbook.close()










