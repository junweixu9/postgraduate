from data_load_ascad_10000 import read_data
import numpy as np
import DL_model as DL_model
import tensorflow.keras as tk
import tensorflow as tf
from keras.utils.np_utils import *

def brier_compute(leakage_model,
                  data_arguementation,
                  data_arguementation_level,
                  attack_model, dataset):
    def adjustment_loss(y_true, y_pred):
        y_true = y_true[:, :classes]
        y_pred = tf.nn.softmax(y_pred, 1)
        loss = tk.backend.categorical_crossentropy(y_true, y_pred)
        return loss

    if dataset == "ASCAD":
        (X_profiling, X_attack), (Y_profiling, Y_attack), (
            plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
            leakage_model,
            data_arguementation,
            data_arguementation_level,
            attack_model, dataset,
            1, 0)
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_tao0_sigma_hw1_nump10000.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_01 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.2_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_02 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.75_sigma_hw1_nump10000.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.4_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_04 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.5_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_05 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.6_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_06 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.7_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_07 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.8_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_08 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_1 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_11 = model.predict(X_attack[:5000])
    elif dataset == "ASCAD_rand":
        (X_profiling, X_attack), (Y_profiling, Y_attack), (
            plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
            leakage_model,
            data_arguementation,
            data_arguementation_level,
            attack_model, dataset,
            1, 0)
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_tao0_sigma_hw1_nump10000.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_01 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.2_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_02 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.2_sigma_hw1_nump10000.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.4_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_04 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.5_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_05 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.6_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_06 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.7_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_07 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.8_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_08 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_1 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_11 = model.predict(X_attack[:5000])

    else:
        (X_profiling, X_attack), (Y_profiling, Y_attack), (
            plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
            leakage_model,
            data_arguementation,
            data_arguementation_level,
            attack_model, dataset,
            1, 0)
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_tao0_sigma_hw1_nump10000.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_01 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.2_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_02 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.2_sigma_hw1_nump10000.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.4_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_04 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.5_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_05 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.6_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_06 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.7_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_07 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao0.8_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_08 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_1 = model.predict(X_attack[:5000])
        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_tao1.1_sigma_hw1_nump10000.h5')
        y_pred_ce_logit_11 = model.predict(X_attack[:5000])

    y_true = Y_attack[:5000, :9]
    y_true = np.argmax(y_true, axis=1)
    y_true = to_categorical(y_true)
    brier_score_ce = np.mean(np.square(y_pred_ce - y_true))
    brier_score_ce_logit = np.mean(np.square(y_pred_ce_logit - y_true))
    brier_score_ce_logit_01 = np.mean(np.square(y_pred_ce_logit_01 - y_true))
    brier_score_ce_logit_02 = np.mean(np.square(y_pred_ce_logit_02 - y_true))
    brier_score_ce_logit_04 = np.mean(np.square(y_pred_ce_logit_04 - y_true))
    brier_score_ce_logit_05 = np.mean(np.square(y_pred_ce_logit_05 - y_true))
    brier_score_ce_logit_06 = np.mean(np.square(y_pred_ce_logit_06 - y_true))
    brier_score_ce_logit_07 = np.mean(np.square(y_pred_ce_logit_07 - y_true))
    brier_score_ce_logit_08 = np.mean(np.square(y_pred_ce_logit_08 - y_true))
    brier_score_ce_logit_1 = np.mean(np.square(y_pred_ce_logit_1 - y_true))
    brier_score_ce_logit_11 = np.mean(np.square(y_pred_ce_logit_11 - y_true))

    return (
    brier_score_ce, brier_score_ce_logit, brier_score_ce_logit_01, brier_score_ce_logit_02, brier_score_ce_logit_04,
    brier_score_ce_logit_05, brier_score_ce_logit_06, brier_score_ce_logit_07, brier_score_ce_logit_08,
    brier_score_ce_logit_1, brier_score_ce_logit_11)


if __name__ == '__main__':
    """变量配置"""
    dataset_ar = 'ASCAD_rand'  # ASCAD/ASCAD_rand/CHES_CTF
    dataset_a = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    dataset_c = 'CHES_CTF'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 5000
    # Select leakage model
    classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 15
    learning_rate = 0.0005
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model

    # brier_score_ce_a, brier_score_ce_a_logit, brier_score_ce_a_logit_01, brier_score_ce_a_logit_02, brier_score_ce_a_logit_04, brier_score_ce_a_logit_05, brier_score_ce_a_logit_06, brier_score_ce_a_logit_07, brier_score_ce_a_logit_08, brier_score_ce_a_logit_1, brier_score_ce_a_logit_11 = brier_compute(
    #     leakage_model, data_arguementation,
    #     data_arguementation_level, attack_model, dataset_a,
    #     )
    # print("a")
    # print(brier_score_ce_a)
    # print(brier_score_ce_a_logit_01)
    # print(brier_score_ce_a_logit_02)
    # print(brier_score_ce_a_logit_04)
    # print(brier_score_ce_a_logit_05)
    # print(brier_score_ce_a_logit_06)
    # print(brier_score_ce_a_logit_07)
    # print(brier_score_ce_a_logit)
    # print(brier_score_ce_a_logit_08)
    # print(brier_score_ce_a_logit_1)
    # print(brier_score_ce_a_logit_11)
    # print()
    # brier_score_ce_ar, brier_score_ce_ar_logit, brier_score_ce_ar_logit_01, brier_score_ce_ar_logit_02, brier_score_ce_ar_logit_04, brier_score_ce_ar_logit_05, brier_score_ce_ar_logit_06, brier_score_ce_ar_logit_07, brier_score_ce_ar_logit_08, brier_score_ce_ar_logit_1, brier_score_ce_ar_logit_11 = brier_compute(
    #     leakage_model, data_arguementation,
    #     data_arguementation_level, attack_model, dataset_ar,
    #     )
    #
    # print("ar")
    # print(brier_score_ce_ar)
    # print(brier_score_ce_ar_logit_01)
    # print(brier_score_ce_ar_logit_02)
    # print(brier_score_ce_ar_logit)
    # print(brier_score_ce_ar_logit_04)
    # print(brier_score_ce_ar_logit_05)
    # print(brier_score_ce_ar_logit_06)
    # print(brier_score_ce_ar_logit_07)
    # print(brier_score_ce_ar_logit_08)
    # print(brier_score_ce_ar_logit_1)
    # print(brier_score_ce_ar_logit_11)
    # print()
    brier_score_ce_c, brier_score_ce_c_logit, brier_score_ce_c_logit_01, brier_score_ce_c_logit_02, brier_score_ce_c_logit_04, brier_score_ce_c_logit_05, brier_score_ce_c_logit_06, brier_score_ce_c_logit_07, brier_score_ce_c_logit_08, brier_score_ce_c_logit_1, brier_score_ce_c_logit_11 = brier_compute(
        leakage_model, data_arguementation,
        data_arguementation_level, attack_model, dataset_c,
        )


    print("c")
    print(brier_score_ce_c)
    print(brier_score_ce_c_logit_01)
    print(brier_score_ce_c_logit_02)
    print(brier_score_ce_c_logit)
    print(brier_score_ce_c_logit_04)
    print(brier_score_ce_c_logit_05)
    print(brier_score_ce_c_logit_06)
    print(brier_score_ce_c_logit_07)
    print(brier_score_ce_c_logit_08)
    print(brier_score_ce_c_logit_1)
    print(brier_score_ce_c_logit_11)
