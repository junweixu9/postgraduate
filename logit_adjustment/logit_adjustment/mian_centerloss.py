from scipy.stats import alpha
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from Util.SCA_util import perform_attacks
from data_load import read_data
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import layers


class corr_max(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(corr_max, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(corr_max, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            logs['corr_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)[1]
            # y_pred_valid_metric = y_pred_valid_metric - 0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_corr_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric,
                                               'corr',
                                               leakage_model, dataset, num_traces_attacks)
            all_corr_logs.append(avg_corr_current)

            logs['corr_val'] = avg_corr_current

            if not corr_logs:
                corr_logs.append(avg_corr_current)
                best_weights = self.model.get_weights()
            else:
                if corr_logs[-1] < avg_corr_current:
                    corr_logs.append(avg_corr_current)
                    best_weights = self.model.get_weights()
                    count = 0
                else:
                    count = count + 1
                    print(count)
                    if count == 5:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)
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
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)[1]
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            avg_rank_current = perform_attacks(all_valid_plt_attack_metric, y_pred_valid_metric, 'rank',
                                               leakage_model, dataset, num_traces_attacks)
            logs['rank_val'] = avg_rank_current
            all_rank_logs.append(avg_rank_current)
            return logs['rank_val']


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
        # x = tf.math.l2_normalize(inputs, axis=1, name='normalized_x')
        # w = tf.math.l2_normalize(self.kernel, axis=0, name='normalized_w')
        output = tf.matmul(inputs, self.kernel)
        # cos_t = tf.matmul(x, w, name='cos_t')
        return output


def feature_extractor_layer():
    img_input = Input(shape=(length,), name="x")
    x = Dense(512, activation='relu')(img_input)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    feature = Dense(128, activation='relu', name="feature")(x)
    # x = Dense(200, activation='elu')(img_input)
    # x = Dense(200, activation='elu')(x)
    # x = Dense(304, activation='elu')(x)
    # x = Dense(832, activation='elu')(x)
    # x = Dense(176, activation='elu')(x)
    # x = Dense(872, activation='elu')(x)
    # x = Dense(608, activation='elu')(x)
    # feature = Dense(512, activation='elu', name="feature")(x)
    # x = Dense(496, activation='relu')(img_input)
    # x = Dense(496, activation='relu')(x)
    # x = Dense(136, activation='relu')(x)
    # x = Dense(288, activation='relu')(x)
    # x = Dense(552, activation='relu')(x)
    # x = Dense(408, activation='relu')(x)
    # x = Dense(232, activation='relu')(x)
    # x = Dense(856, activation='relu')(x)

    # cos = CosineLayer()

    # x = cos(x)
    x = Dense(9, name="class_feature")(feature)
    model = Model(img_input, outputs=[feature, x])
    optimizer = RMSprop(lr=0.0005)  # 0.0005
    model.compile(loss={"feature": centerloss, "class_feature": adjustment_loss}, optimizer=optimizer,
                  loss_weights=[0.8, 1])

    return model


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

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

    return adjustments


class Center_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.centers = tf.Variable(tf.zeros([9, n_features]), name="centers",
                                   trainable=False)

    def call(self, y_true, y_pred):
        # labels_batch = tf.reshape(labels_batch, (-1,))
        labels_batch = tf.math.argmax(y_true[:, :9], axis=1)
        x_batch = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
        centers_batch = tf.gather(self.centers, labels_batch)
        # print("centers_batch", centers_batch)
        # the reduction of batch dimension will be done by the parent class
        center_loss = tf.keras.losses.mean_squared_error(x_batch, centers_batch)

        # update center
        labels_batch_expand = tf.expand_dims(labels_batch, 0)
        # print("labels_batch_expand", labels_batch_expand)
        # print("centers", self.centers)
        # print("x_batch", x_batch)
        x_correspond_centers_batch = tf.gather(self.centers, labels_batch)
        # print("x_correspond_centers_batch", x_correspond_centers_batch)
        diff = (x_correspond_centers_batch - x_batch)
        # print("diff", diff)
        one = tf.ones_like(labels_batch_expand)  # 生成与a大小一致的值全部为1的矩阵
        zero = tf.zeros_like(labels_batch_expand)
        for i in range(classes):
            if i == 0:
                stack_label_0_1 = tf.where(labels_batch_expand == i, x=one, y=zero)
            else:
                stack_label_0_1 = tf.concat([stack_label_0_1, tf.where(labels_batch_expand == i, x=one, y=zero)],
                                            axis=0)
        # print("stack_label", stack_label_0_1)
        stack_label_0_1 = tf.cast(stack_label_0_1, dtype=tf.float32)
        stack_label_0_1_count = tf.reduce_sum(stack_label_0_1, axis=1)
        stack_label_0_1_count = tf.expand_dims(stack_label_0_1_count, axis=1)
        # print("stack_label_0_1_count", stack_label_0_1_count)
        update = tf.matmul(stack_label_0_1, diff)
        # print("update", update)
        res = tf.ones(shape=(9, n_features))
        stack_label_0_1_count_same_metrix = tf.multiply(res, stack_label_0_1_count + 1)
        # print("stack_label_0_1_count_same_metrix", stack_label_0_1_count_same_metrix)
        update_norm = tf.math.divide(update, stack_label_0_1_count_same_metrix)
        # print("update_norm", update_norm)
        center_update = 0.1 * update_norm
        # print("center_update", center_update)
        self.centers.assign_sub(center_update)  # this will do the following: self.center = self.center - updates
        # print("self.centers after ", self.centers)

        return center_loss


# class Center_Loss(layers.Layer):
#     def __init__(self, name="center_loss", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.centers = tf.Variable(tf.zeros([9, n_features]), name="centers",
#                                    trainable=False)
#
#     def call(self, inputs, **kwargs):
#         # labels_batch = tf.reshape(labels_batch, (-1,))
#         x_batch, labels_batch = inputs
#         labels_batch = tf.math.argmax(labels_batch, axis=1)
#         x_batch = tf.reshape(x_batch, (tf.shape(x_batch)[0], -1))
#         centers_batch = tf.gather(self.centers, labels_batch)
#         # print("centers_batch", centers_batch)
#         # the reduction of batch dimension will be done by the parent class
#         center_loss = tf.keras.losses.mean_squared_error(x_batch, centers_batch)
#
#         # update center
#         labels_batch_expand = tf.expand_dims(labels_batch, 0)
#         print("labels_batch_expand", labels_batch_expand)
#         print("centers", self.centers)
#         print("x_batch", x_batch)
#         x_correspond_centers_batch = tf.gather(self.centers, labels_batch)
#         print("x_correspond_centers_batch", x_correspond_centers_batch)
#         diff = (x_correspond_centers_batch - x_batch)
#         print("diff", diff)
#         one = tf.ones_like(labels_batch_expand)  # 生成与a大小一致的值全部为1的矩阵
#         zero = tf.zeros_like(labels_batch_expand)
#         for i in range(classes):
#             if i == 0:
#                 stack_label_0_1 = tf.where(labels_batch_expand == i, x=one, y=zero)
#             else:
#                 stack_label_0_1 = tf.concat([stack_label_0_1, tf.where(labels_batch_expand == i, x=one, y=zero)],
#                                             axis=0)
#         print("stack_label", stack_label_0_1)
#         stack_label_0_1 = tf.cast(stack_label_0_1, dtype=tf.float32)
#         stack_label_0_1_count = tf.reduce_sum(stack_label_0_1, axis=1)
#         stack_label_0_1_count = tf.expand_dims(stack_label_0_1_count, axis=1)
#         print("stack_label_0_1_count", stack_label_0_1_count)
#         update = tf.matmul(stack_label_0_1, diff)
#         print("update", update)
#         res = tf.ones(shape=(9, n_features))
#         stack_label_0_1_count_same_metrix = tf.multiply(res, stack_label_0_1_count + 1)
#         print("stack_label_0_1_count_same_metrix", stack_label_0_1_count_same_metrix)
#         update_norm = tf.math.divide(update, stack_label_0_1_count_same_metrix)
#         print("update_norm", update_norm)
#         center_update = alpha * update_norm
#         print("center_update", center_update)
#         self.centers.assign_sub(center_update)  # this will do the following: self.center = self.center - updates
#         print("self.centers after ", self.centers)
#
#         return center_loss


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
    alpha = 0.1
    length = X_profiling.shape[1]
    n_features = 128
    adjustments = compute_adjustment(Y_profiling, tro)
    centerloss = Center_Loss()
    # m_list = compute_adjustment_m_list(Y_profiling)

    model = feature_extractor_layer()
    model.summary()
    callback = [rank(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000])),
                corr_max(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
                ]
    model.fit(x={"x": X_profiling}, y={"feature": Y_profiling[:, :9], "class_feature": Y_profiling[:, :9]},
              batch_size=batch_size, verbose=2, epochs=epoch,
              callbacks=callback)
    print('======Attack======')
    predictions = model.predict(X_attack[5000:])[1]
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
