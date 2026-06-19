from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from Util.SCA_util import perform_attacks
from data_load import read_data
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk


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
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            avg_rank_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric, 'rank',
                                               leakage_model, dataset, num_traces_attacks)
            logs['rank_val'] = avg_rank_current
            all_rank_logs.append(avg_rank_current)
            return logs['rank_val']


# class NormedLinear(nn.Module):
#
#     def __init__(self, in_features, out_features):
#         super(NormedLinear, self).__init__()
#         self.weight = Parameter(torch.Tensor(in_features, out_features))
#         self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#
#     def forward(self, x):
#         cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
#         return cosine
class CosineLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CosineLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    def build(self, input_shape):
        # input_shape = [batch_size,feature_vector_output]
        super(CosineLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], 9),
                                      # initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                      initializer='glorot_normal',
                                      trainable=True
                                      )

        # self.kernel.assign(tf.math.l2_normalize(self.kernel, axis=0) * 1e5)

    @tf.function
    def call(self, inputs, training=None):
        x = tf.math.l2_normalize(inputs, axis=1, name='normalized_x')
        w = tf.math.l2_normalize(self.kernel, axis=0, name='normalized_w')
        # output = tf.matmul(inputs, self.kernel)
        cos_t = tf.matmul(x, w, name='cos_t')
        return cos_t


def feature_extractor_layer():
    img_input = Input(shape=(length,))
    # x = Dense(200, activation='elu')(img_input)
    # x = Dense(200, activation='elu')(x)
    # x = Dense(304, activation='elu')(x)
    # x = Dense(832, activation='elu')(x)
    # x = Dense(176, activation='elu')(x)
    # x = Dense(872, activation='elu')(x)
    # x = Dense(608, activation='elu')(x)
    # x = Dense(512, activation='elu')(x)

    x = Dense(496, activation='relu')(img_input)
    x = Dense(496, activation='relu')(x)
    x = Dense(136, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    x = Dense(552, activation='relu')(x)
    x = Dense(408, activation='relu')(x)
    x = Dense(232, activation='relu')(x)
    x = Dense(856, activation='relu')(x)

    cos = CosineLayer()

    x = cos(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)  # 0.0005
    model.compile(loss=adaptive_adjustment_loss, optimizer=optimizer, metrics=None)

    return model


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


def adaptive_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        cos_adjustment = tf.multiply(y_true, (1 - y_pred) / 2)
        cos_adjustment = tf.multiply(cos_adjustment, adjustments)
        y_pred = y_pred - cos_adjustment

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


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
    adjustments_norm = - adjustments / np.min(adjustments)
    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments_norm


def compute_adjustment_adaptive(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling = np.argmax(Y_profiling[:, :9], 1)
    label_freq = {}
    for key in Y_profiling:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array_divide = label_freq_array / np.min(label_freq_array)
    adjustments = 1 / np.log(label_freq_array_divide + 1)

    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments


# def compute_adjustment_m_list(Y_profiling):
#     """compute the base probabilities"""
#
#     Y_profiling = np.argmax(Y_profiling[:, :9], 1)
#     label_freq = {}
#     for key in Y_profiling:
#         label_freq[key] = label_freq.get(key, 0) + 1
#     label_freq = dict(sorted(label_freq.items()))
#     label_freq_array = np.array(list(label_freq.values()))
#
#     # min_freq = np.min(label_freq_array)
#     #
#     # QF = 1 / np.log((label_freq_array / min_freq) + 1)
#     #
#     # QF = tf.cast(QF, dtype=tf.float32)
#
#     m_list = np.log(label_freq_array)
#
#     m_list = np.max(m_list) - m_list
#
#     m_list = m_list / np.max(m_list)
#
#     return m_list


# def GCL_loss(y_true, y_pred):
#     m_list_tf = tf.cast(m_list, tf.float32)
#
#     zhengtai_0 = tf.zeros_like(y_pred)
#
#     noise = tf.random.normal(shape=[len(zhengtai_0), 1], mean=0, stddev=1 / 3)
#
#     noise_clip = tf.clip_by_value(noise, -1, 1)
#
#     y_pred = 30 * (y_pred - noise_mul * tf.multiply(tf.abs(noise_clip), m_list_tf))
#
#     # y_pred = tf.where(y_true, y_pred - m, y_pred)
#
#     y_true = y_true[:, :9]
#
#     y_pred = tf.nn.softmax(y_pred, 1)
#
#     loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#
#     return loss


if __name__ == '__main__':
    """变量配置"""
    dataset = 'ASCAD_rand'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 5000
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 20
    learning_rate = 0.0005
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all
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
    noise_mul = 0.5
    m = 0
    batch_size = 32
    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    length = X_profiling.shape[1]

    adjustments = compute_adjustment_adaptive(Y_profiling, tro)

    # m_list = compute_adjustment_m_list(Y_profiling)

    model = feature_extractor_layer()
    model.summary()
    callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                corr(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                ]
    model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
              callbacks=callback)
    print('======Attack======')
    predictions = model.predict(X_attack[5000:])
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
