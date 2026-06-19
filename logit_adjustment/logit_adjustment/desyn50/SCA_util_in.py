import numpy as np
import scipy.stats as ss
from tensorflow.keras import backend as K
from scipy import stats
import sys
import random
from scipy.special import softmax


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


def perform_attacks(all_valid_plt_attack, y_pred, output_metric, leakage_model, dataset, nb_traces_attacks,
                    Y_attack_valid, shuffle=True):
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
        all_corr_evol = np.zeros((num_of_attacks, 1))
    #  打乱攻击数据集
    for i in range(num_of_attacks):
        if shuffle:
            l = list(zip(y_pred, all_valid_plt_attack, Y_attack_valid))
            random.shuffle(l)
            s1, s2, s3 = list(zip(*l))

            yibufen_attack_pred = s1[:nb_traces_attacks]
            yibufen_attack_pred = np.array(yibufen_attack_pred)

            yibufen_attack_plt = s2[:nb_traces_attacks]
            yibufen_attack_plt = np.array(yibufen_attack_plt)

            yibufen_attack_label = s3[:nb_traces_attacks]
            yibufen_attack_label = np.array(yibufen_attack_label)

        '''得到打乱之后的数据集，进行一次rank or corr的计算'''
        results = rank_compute(yibufen_attack_pred, yibufen_attack_plt, attack_byte, k_c, output_metric, leakage_model,
                               nb_traces_attacks, yibufen_attack_label)
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
            all_rank_evol[i] = results[0][nb_traces_attacks - 1, k_c]
            all_key_attack_traces[i] = results[0]
            all_corr_evol[i] = results[1]

    """进行num_of_attacks次打乱的攻击数据集后，进行平均值计算"""
    if output_metric == 'corr':
        return np.mean(all_corr_evol, axis=0)
    elif output_metric == 'rank':
        #  将进行 nb_attacks 次计算得到的矩阵（该矩阵的第i行是使用i条攻击能量轨迹数得到的猜测向量），计算平均值，消除随机性
        return (np.mean(all_rank_evol, axis=0)), np.mean(all_key_attack_traces, axis=0)
    elif output_metric == "all":
        return (np.mean(all_rank_evol, axis=0)), np.mean(all_key_attack_traces, axis=0), np.mean(all_corr_evol, axis=0)
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


def rank_compute(yibufen_attack_pred, yibufen_attack_plt, attack_byte, k_c, output_metric, leakage_model,
                 nb_traces_attacks, yibufen_attack_label) \
        :
    (nb_traces, nb_hyp) = yibufen_attack_pred.shape

    if output_metric == "corr" or "all":
        # 得到密钥分布的排名
        KD_rank = KD_distribution(yibufen_attack_plt, attack_byte, k_c)

    key_log_prob_accu = np.zeros(256)
    key_log_prob_evol = np.zeros((nb_traces, 256))

    yibufen_attack_label = np.argmax(yibufen_attack_label[:, :9], 1)
    label_freq = {}
    for key in yibufen_attack_label:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array + 1e-12)

    yibufen_attack_pred = yibufen_attack_pred - adjustments
    yibufen_attack_pred = softmax(yibufen_attack_pred, axis=1)

    #  将得到的概率分布转换为log分数
    yibufen_attack_pred = np.log(np.where(yibufen_attack_pred <= K.epsilon(), K.epsilon(), yibufen_attack_pred))

    for i in range(nb_traces):
        #   从log的概率分布取出所有密钥根据汉明重量计算出相应标签的log得分
        g = Utility.calculate_HW(
            Utility.AES_Sbox()[np.bitwise_xor(range(256), int(yibufen_attack_plt[i, attack_byte]))])
        predicted_output = yibufen_attack_pred[i, g]
        #  并进行对应密钥得分的相乘
        key_log_prob_accu += predicted_output
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
        corr, _ = stats.spearmanr(rank[nb_traces_attacks - 1], KD_rank)
        return rank, corr
