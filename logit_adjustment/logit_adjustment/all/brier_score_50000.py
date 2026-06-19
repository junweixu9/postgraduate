from data_load_ascad_50000 import read_data
import numpy as np
import DL_model as DL_model
import tensorflow.keras as tk
import tensorflow as tf


def brier_compute(leakage_model,
                  data_arguementation,
                  data_arguementation_level,
                  attack_model, dataset,
                  sigma_hw, sigma_id):
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    def adjustment_loss(y_true, y_pred):
        y_true = y_true[:, :classes]
        y_pred = tf.nn.softmax(y_pred, 1)
        loss = tk.backend.categorical_crossentropy(y_true, y_pred)
        return loss

    if dataset == "ASCAD":
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_sigma_hw0.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_sigma_hw0.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])
    elif dataset == "ASCAD_rand":
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_sigma_hw0.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetASCAD_rand_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_sigma_hw0.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])
    else:
        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagFalse_sigma_hw0.h5')
        y_pred_ce = model.predict(X_attack[:5000])

        model, batch_size, epoch_sota = DL_model.pick_SOAT(dataset, leakage_model, X_profiling.shape[1],
                                                           companion_metric,
                                                           adjustment_loss, learning_rate, model=attack_model,
                                                           model_size=model_size)

        model.load_weights(
            r'./model_save/datasetCHES_CTF_leakage_modelHW_epoch15_batch_size32_adjust_flagTrue_sigma_hw0.h5')
        y_pred_ce_logit = model.predict(X_attack[:5000])

    y_true = Y_attack[:5000, :9]

    brier_score_ce = np.linalg.norm(y_pred_ce - y_true, ord=2)
    brier_score_ce_logit = np.linalg.norm(y_pred_ce_logit - y_true, ord=2)

    return brier_score_ce, brier_score_ce_logit


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

    a_brier_score_ce, a_brier_score_ce_logit = brier_compute(leakage_model, data_arguementation,
                                                             data_arguementation_level, attack_model, dataset_a,
                                                             sigma_hw, sigma_id)
    ar_brier_score_ce, ar_brier_score_ce_logit = brier_compute(leakage_model, data_arguementation,
                                                               data_arguementation_level, attack_model, dataset_ar,
                                                               sigma_hw, sigma_id)
    c_brier_score_ce, c_brier_score_ce_logit = brier_compute(leakage_model, data_arguementation,
                                                             data_arguementation_level, attack_model, dataset_c,
                                                             sigma_hw, sigma_id)
    print("a_brier_score_ce:", a_brier_score_ce)
    print("a_brier_score_ce_logit:", a_brier_score_ce_logit)
    print("ar_brier_score_ce:", ar_brier_score_ce)
    print("ar_brier_score_ce_logit:", ar_brier_score_ce_logit)
    print("c_brier_score_ce:", c_brier_score_ce)
    print("c_brier_score_ce_logit:", c_brier_score_ce_logit)