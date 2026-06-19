from data_load_ascad_50000 import read_data
from SCA_util_e import perform_attacks

import scipy.stats as ss
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
import time

class Utility:
    def AES_Sbox():
        return np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

    def hw():
        return np.array([bin(x).count("1") for x in range(256)])

    def labelize(plaintexts, keys):
        return Utility.AES_Sbox()[plaintexts ^ keys]

    def calculate_HW(data):
        if isinstance(data, int):
            print('Input must be an array')
            sys.exit(-1)
        if data.ndim == 1:
            return Utility.hw()[data]
        else:
            return np.reshape([Utility.hw()[data.ravel()]], np.shape(data))

    ''' rk_key_all_traces函数的作用是对输入的rank_array中的每一行进行密钥排名。具体来说，它对每一行中的元素按降序排名，然后将排名结果存储在一个新的数组中。'''

    def rk_key_all_traces(rank_array):
        container = np.empty(rank_array.shape, dtype=int)
        for k, row in enumerate(rank_array):
            container[k] = ss.rankdata(-row, method='dense') - 1
        return container

    def compute_label_distribution(labels, leakage_model, sigma_hw, sigma_id):
        if leakage_model == 'HW':
            sigma = sigma_hw
            classes = 9
        else:
            sigma = sigma_id
            classes = 256
        container = np.zeros((len(labels), classes), dtype=np.float32)
        # Label Distribution Learning
        for idx, label in enumerate(labels):
            container[idx] = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - label) ** 2 / (2 * sigma ** 2)) for
                              bins in range(classes)]
        return container


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


def ascad_f_hw_cnn_rs(length, metric, loss, learning_rate):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 800
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


    """创建神经网络模型"""
    """ select the output_metric function """
    #

    # 执行你的代码
    number_of_models = 300
    number_of_best_models = 300
    all_son_predictions_softmax = []
    all_son_attack_traces = []
    np.random.seed(2075)
    shuffled_indices_list = [np.random.permutation(np.arange(5000))[:800] for _ in range(50)]
    start_time = time.time()
    for i in range(1, number_of_models + 1):
        model = ascad_f_hw_cnn_rs(X_profiling.shape[1], None, adjustment_loss, 5e-3)
        """最佳模型的存储地址"""
        test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                     epoch,
                                                                                     128,
                                                                                     output_metric,
                                                                                     )

        """开始训练"""
        i_30 = i % 30
        model_root = 'C:/workspace/code/logit_adjustment/logit_adjustment/ensemble_model/' + test_info + str(i_30) + '.h5'
        # model_root = 'E:/logit_backup/ensemble_model/' + test_info + 'false' + str(i) + '.h5'
        model.load_weights(model_root)
        each_son_predictions = model.predict(X_attack[5000:])
        each_son_predictions_softmax = tf.nn.softmax(each_son_predictions)
        each_son_attack_traces = perform_attacks(plt_attack[5000:], each_son_predictions_softmax, "one_attack_traces",
                                                 leakage_model, dataset, num_traces_attacks, shuffled_indices_list)

        if each_son_attack_traces[-1, correct_key] > 0:
            # print("攻击失败")
            # print("GE:", each_son_attack_traces[-1, correct_key])
            all_son_attack_traces.append(num_traces_attacks + each_son_attack_traces[-1, correct_key])

        else:
            # print("攻击成功")
            # print("TGE0:", np.argmax(each_son_attack_traces[:, correct_key] < 1))
            all_son_attack_traces.append(np.argmax(each_son_attack_traces[:, correct_key] < 1))
        # print("攻击成功")
        # print("TGE0:", np.argmax(each_son_attack_traces[:, correct_key] < 1))
        all_son_predictions_softmax.append(each_son_predictions_softmax)

    sorted_indices = np.argsort(all_son_attack_traces).tolist()

    all_son_attack_traces_ensemble, all_son_attack_traces_best_ensemble = perform_attacks(plt_attack[5000:],
                                                                                          all_son_predictions_softmax,
                                                                                          "all_attack_traces",
                                                                                          leakage_model, dataset,
                                                                                          num_traces_attacks,
                                                                                          shuffled_indices_list,
                                                                                          sorted_indices=sorted_indices,
                                                                                          number_of_best_models=number_of_best_models)
    if all_son_attack_traces_best_ensemble[-1, correct_key] > 0:
        print("攻击失败")
        print("所有模型集成得到的GE:{},".format( all_son_attack_traces_ensemble[-1, correct_key]))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time} 秒")
        print("最好的几个模型集成得到的GE:{},".format(
            all_son_attack_traces_best_ensemble[-1, correct_key]))


    else:
        print("攻击成功")

        print("所有模型集成得到的TGE0:{},".format(np.argmax(all_son_attack_traces_ensemble[:, correct_key] < 1)))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间：{execution_time} 秒")
        print("最好的几个模型集成得到的TGE0:{},".format(
            np.argmax(all_son_attack_traces_best_ensemble[:, correct_key] < 1)))



    # sorted_models = sorted(all_son_attack_traces, key=lambda l: l[:])
    # list_of_best_models = []
    # for model_index in range(number_of_models):
    #     list_of_best_models.append(sorted_models[model_index][1] - 1)
    # kr_ensemble = np.zeros(num_traces_attacks)
    # # kr_ensemble_other = np.zeros(kr_nt)
    # krs_ensemble = np.zeros((20, num_traces_attacks))
    # kr_ensemble_best_models = np.zeros(num_traces_attacks)
    # krs_ensemble_best_models = np.zeros((20, num_traces_attacks))
    #
    # for run in range(20):
    #
    #     key_p_ensemble = np.zeros(256)
    #     key_p_ensemble_best_models = np.zeros(256)
    #
    #     for index in range(num_traces_attacks):
    #         for model_index in range(number_of_models):
    #             key_p_ensemble += k_ps_all[list_of_best_models[model_index]][run][index]
    #         for model_index in range(number_of_best_models):
    #             key_p_ensemble_best_models += k_ps_all[list_of_best_models[model_index]][run][index]
    #
    #         key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
    #         # key_p_ensemble_sorted_other = ss.rankdata(-key_p_ensemble_best_models, method='dense') - 1
    #         # kr_position_other = key_p_ensemble_sorted_other[correct_key]
    #         key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]
    #
    #         kr_position = list(key_p_ensemble_sorted).index(correct_key)
    #         kr_ensemble[index] += kr_position
    #         # kr_ensemble_other[index] += kr_position_other
    #
    #         kr_position = list(key_p_ensemble_best_models_sorted).index(correct_key)
    #         kr_ensemble_best_models[index] += kr_position
    #
    #     print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, number_of_models,
    #                                                                    int(kr_ensemble[num_traces_attacks - 1] / (run + 1)),
    #                                                                    number_of_best_models,
    #                                                                    int(kr_ensemble_best_models[num_traces_attacks - 1] / (
    #                                                                            run + 1))))
    #
    # ge_ensemble = kr_ensemble / 20
    # # ge_ensemble_other = kr_ensemble_other / 20
    # ge_ensemble_best_models = kr_ensemble_best_models / 20
    # print("ge_ensemble", np.argmax(ge_ensemble < 1))
    # print("ge_ensemble_best_models", np.argmax(ge_ensemble_best_models < 1))
