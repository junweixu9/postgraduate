import numpy as np
import scipy.stats as ss
from tensorflow.keras import backend as K
from scipy import stats
import sys
import random


class Utility:
    def AES_Sbox_inv():
        return np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
                             0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
                             0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
                             0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
                             0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
                             0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
                             0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
                             0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
                             0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
                             0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
                             0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
                             0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
                             0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
                             0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
                             0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
                             0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
                             0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
                             0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
                             0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
                             0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
                             0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
                             0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
                             0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
                             0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
                             0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
                             0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
                             0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
                             0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
                             0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
                             0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
                             0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
                             0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
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


def perform_attacks(all_valid_plt_attack, y_pred, output_metric, leakage_model, dataset, nb_traces_attacks,
                    shuffle=True):
    if dataset == "ASCAD":
        k_c = 224
        attack_byte = 2
    elif dataset == "ASCAD_rand":
        k_c = 34
        attack_byte = 2
    elif dataset == "CHES_CTF":
        k_c = 46
        attack_byte = 0
    else:
        k_c = 200
        attack_byte = 11

    num_of_attacks = 20

    if output_metric == "rank":
        all_rank_evol = np.zeros((num_of_attacks, 1))
        all_key_attack_traces = np.zeros((num_of_attacks, nb_traces_attacks, 256))
    elif output_metric == "corr":
        all_corr_evol = np.zeros((num_of_attacks, 1))
    elif output_metric == "attack_traces":
        all_key_attack_traces = np.zeros((num_of_attacks, nb_traces_attacks, 256))
    elif output_metric == "all":
        all_rank_evol = np.zeros((num_of_attacks, 1))
        all_key_attack_traces = np.zeros((num_of_attacks, nb_traces_attacks, 256))
        # all_corr_evol = np.zeros((num_of_attacks, 1))
    #  打乱攻击数据集
    for i in range(num_of_attacks):
        if shuffle:
            l = list(zip(y_pred, all_valid_plt_attack))
            random.shuffle(l)
            s1, s2 = list(zip(*l))

            yibufen_attack_pred = s1[:nb_traces_attacks]
            yibufen_attack_pred = np.array(yibufen_attack_pred)

            yibufen_attack_plt = s2[:nb_traces_attacks]
            yibufen_attack_plt = np.array(yibufen_attack_plt)

        '''得到打乱之后的数据集，进行一次rank or corr的计算'''
        results = rank_compute(yibufen_attack_pred, yibufen_attack_plt, attack_byte, k_c, output_metric, leakage_model,
                               nb_traces_attacks)
        #  把进行一次计算的rank or corr存储到新的向量中，以便做num_of_attacks的平均值
        if output_metric == 'corr':
            all_corr_evol[i] = results
        elif output_metric == 'rank':
            #  计算一次指标需进行 nb_attacks 次数据的计算
            all_rank_evol[i] = results[nb_traces_attacks - 1, k_c]
            all_key_attack_traces[i] = results
        elif output_metric == "attack_traces":
            all_key_attack_traces[i] = results
        elif output_metric == "all":
            all_rank_evol[i] = results[nb_traces_attacks - 1, k_c]
            all_key_attack_traces[i] = results
            # all_corr_evol[i] = results[1]

    """进行num_of_attacks次打乱的攻击数据集后，进行平均值计算"""
    if output_metric == 'corr':
        return np.mean(all_corr_evol, axis=0)
    elif output_metric == 'rank':
        #  将进行 nb_attacks 次计算得到的矩阵（该矩阵的第i行是使用i条攻击能量轨迹数得到的猜测向量），计算平均值，消除随机性
        return (np.mean(all_rank_evol, axis=0)), np.mean(all_key_attack_traces, axis=0)
    elif output_metric == "all":
        return (np.mean(all_rank_evol, axis=0)), np.mean(all_key_attack_traces, axis=0)
    elif output_metric == 'attack_traces':
        #  将进行 nb_attacks 次计算得到的矩阵（该矩阵的第i行是使用i条攻击能量轨迹数得到的猜测向量），计算平均值，消除随机性
        return np.mean(all_key_attack_traces, axis=0)


def KD_distribution(yibufen_plt_attack, attack_byte, k_c):
    num_plt_attack = len(yibufen_plt_attack)
    k = np.zeros((num_plt_attack, 256))
    variance = np.zeros((256, 1))
    for i in range(num_plt_attack):
        # (num_plt_attack,256)
        k[i] = Utility.calculate_HW(
            Utility.AES_Sbox()[np.bitwise_xor(range(256), int(yibufen_plt_attack[i, attack_byte]))])
    k = k.T  # (256,num_plt_attack)
    for i in range(256):
        variance[i] = np.sum(abs(np.power(k[k_c] - k[i], 2)))
    KD_rank = ss.rankdata(variance)
    return KD_rank


def compute_values(predicted_output, result, g):
    return predicted_output / result[g]


def convert_to_multilabel(byte_values):
    """Converts a 1D array of byte values (0-255) to a 2D numpy array of shape (N, 8)."""
    # Ensure input is a numpy array of type uint8 for unpackbits
    byte_values = np.array(byte_values, dtype=np.uint8)
    # np.unpackbits converts each byte into an array of 8 bits (MSB first)
    return np.unpackbits(byte_values[:, np.newaxis], axis=1)


def rank_compute(yibufen_attack_pred, yibufen_attack_plt, attack_byte, k_c, output_metric, leakage_model,
                 nb_traces_attacks) \
        :
    (nb_traces, nb_hyp) = yibufen_attack_pred.shape

    key_log_prob_accu = np.zeros(256)
    key_log_prob_evol = np.zeros((nb_traces, 256))

    #  将得到的概率分布转换为log分数
    epsilon = 1e-36
    yibufen_attack_pred = np.clip(yibufen_attack_pred, epsilon, 1 - epsilon)

    log_prob_bit_1 = np.log(yibufen_attack_pred)
    log_prob_bit_0 = np.log(1 - yibufen_attack_pred)

    for i in range(nb_traces):
        #   从log的概率分布取出所有密钥根据汉明重量计算出相应标签的log得分

        c_log_prob_bit_1 = log_prob_bit_1[i]
        c_log_prob_bit_0 = log_prob_bit_0[i]

        f = Utility.AES_Sbox_inv()[np.bitwise_xor(range(256), int(yibufen_attack_plt[i, 11]))]
        g = np.bitwise_xor(f,int(yibufen_attack_plt[i, 7]))

        g_bits = convert_to_multilabel(g)

        t_likelihoods = np.where(g_bits == 1, c_log_prob_bit_1, c_log_prob_bit_0)
        #  并进行对应密钥得分的相乘

        key_scores_for_this_trace = np.sum(t_likelihoods, axis=1)

        key_log_prob_accu += key_scores_for_this_trace
        #  用i条攻击轨迹得到的未经过排名的猜测向量 guessing entropy ==  key_log_prob_evol[i]
        key_log_prob_evol[i] = key_log_prob_accu  # (nb_traces,256)

    if output_metric == 'corr':
        '''rk_key_all_traces 函数的作用是对输入的未经过排名的猜测向量 key_log_prob_evol 中的每一行进行密钥排名。具体来说，它对每一行中的元素按降序排名，然后将排名结果存储在一个新的数组中。'''
        rank = Utility.rk_key_all_traces(key_log_prob_evol)
        corr, _ = stats.spearmanr(rank[nb_traces_attacks - 1], KD_rank)
        return corr
    elif output_metric == 'rank':
        rank = Utility.rk_key_all_traces(key_log_prob_evol)
        return rank
    elif output_metric == 'attack_traces':
        attack = Utility.rk_key_all_traces(key_log_prob_evol)
        return attack
    elif output_metric == "all":
        rank = Utility.rk_key_all_traces(key_log_prob_evol)
        return rank
