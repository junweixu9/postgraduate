import tensorflow as tf
import Util.SCA_dataset as SCA_dataset
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
import Util.SCA_util as SCA_util


def read_data(leakage_model, data_arguementation, data_arguementation_level,
              attack_model, dataset, sigma_hw, sigma_id):
    """数据导入"""

    if dataset == 'ASCAD':
        correct_key = 224  # 正确密钥
        attack_byte = 2  # 攻击s盒的字节位置
        num_profiling_traces = 5000
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), (
            key_profiling, key_attack) = SCA_dataset.load_ascad(
            "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/ASCAD_data/ASCAD_databases/ASCAD.h5",
            leakage_model=leakage_model,
            profiling_traces=num_profiling_traces,
            key_info=True)
    elif dataset == 'ASCAD_rand':
        data_root = 'C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/ASCAD_data/ASCAD_databases/ascad-variable.h5'
        correct_key = 34
        attack_byte = 2
        num_profiling_traces = 5000
        num_attack_traces = 10000
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), (
            key_profiling, key_attack) = SCA_dataset.load_ascad_rand(data_root,
                                                                     leakage_model=leakage_model,
                                                                     profiling_traces=num_profiling_traces,
                                                                     attack_trace=num_attack_traces,
                                                                     key_info=True)
    elif dataset == 'CHES_CTF':
        data_root = 'C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/ASCAD_data/ASCAD_databases/ches_ctf.h5'
        correct_key = 46
        attack_byte = 0
        num_profiling_traces = 5000
        num_attack_traces = 10000
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), (
            key_profiling, key_attack) = SCA_dataset.load_chesctf(data_root, leakage_model=leakage_model,
                                                                  profiling_traces=num_profiling_traces,
                                                                  key_info=True)
    elif dataset == 'AESHD':
        data_root = 'C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/ASCAD_data/ASCAD_databases/aes_hd.h5'
        correct_key = 200
        nb_traces_attacks = 5000
        (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (plt_profiling, plt_attack) = SCA_dataset.load_aeshd(data_root, leakage_model=leakage_model,
                                                                  profiling_traces=45000)
    else:
        print('No dataset defined!')
        sys.exit(-1)

    """数据处理"""

    # 能量轨迹归一化
    scaler = StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    # Performing data arguementation
    if data_arguementation:
        X_profiling, Y_profiling, plt_profiling = SCA_dataset.data_augmentation_gaussian_noise(X_profiling,
                                                                                               Y_profiling,
                                                                                               plt_profiling,
                                                                                               arg_level=data_arguementation_level)
    if attack_model == 'CNN':
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9

    # Prepare the label: label+identifier+pleintext (for metric calculation)
    if sigma_hw == 0 and sigma_id == 0:
        print('noLD_{}_{}'.format(sigma_hw, sigma_id))
        Y_profiling = np.concatenate(
            (to_categorical(Y_profiling, num_classes=classes), np.zeros((len(plt_profiling), 1)), plt_profiling),
            axis=1)
        Y_attack = np.concatenate(
            (to_categorical(Y_attack, num_classes=classes), np.ones((len(plt_attack), 1)), plt_attack), axis=1)
    else:
        print('LD_{}_{}'.format(sigma_hw, sigma_id))
        if leakage_model == 'HW':
            Y_profiling = np.concatenate((
                SCA_util.Utility.compute_label_distribution(Y_profiling, leakage_model, sigma_hw,
                                                            sigma_id),
                np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
            Y_attack = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_attack, leakage_model, sigma_hw,
                                                                                   sigma_id),
                                       np.ones((len(plt_attack), 1)), plt_attack), axis=1)
        else:
            Y_profiling = np.concatenate((
                SCA_util.Utility.compute_label_distribution(Y_profiling, leakage_model, sigma_hw,
                                                            sigma_id),
                np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
            Y_attack = np.concatenate((SCA_util.Utility.compute_label_distribution(Y_attack, leakage_model, sigma_hw,
                                                                                   sigma_id),
                                       np.ones((len(plt_attack), 1)), plt_attack), axis=1)

    return (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces
