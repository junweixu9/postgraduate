import tensorflow as tf
from tensorflow.keras import layers, Model
from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import he_uniform
# import tensorflow_probability as tfp


class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(all, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(all, self).set_params(params)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation:
            global best_weights
            global count
            global epoch_count
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric, Y_attack_valid = self.validation[0], self.validation[1], \
                self.validation[2]
            y_pred_valid_metric_all = self.model.predict(X_attack_valid_metric)

            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric_all,
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
                    if count == 14:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)


class NormedLinear(Layer):
    def __init__(self, units, **kwargs):
        super(NormedLinear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='weight',
            shape=(input_shape[-1], self.units),
            initializer=he_uniform(),
            trainable=True
        )
        self.built = True

    def call(self, inputs):
        # L2 归一化输入特征和权重
        x_norm = tf.math.l2_normalize(inputs, axis=1)
        w_norm = tf.math.l2_normalize(self.w, axis=0)

        # 计算余弦相似度
        return tf.matmul(x_norm, w_norm)


class GCLLoss(tf.keras.losses.Loss):
    def __init__(self, cls_num_list, m=0.5, s=30, train_cls=False,
                 noise_mul=1., gamma=0., name="gcl_loss"):
        super().__init__(name=name)

        # 直接赋值属性
        self.cls_num_list = tf.constant(cls_num_list, dtype=tf.float32)
        m_list = tf.math.log(self.cls_num_list)
        self.m_list = tf.reduce_max(m_list) - m_list
        self.m = m
        self.s = s
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def call(self, y_true, cosine):

        y_true = tf.cast(y_true, tf.int32)

        # 确保y_true是一维的
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)

        # 创建目标类索引矩阵
        index = tf.one_hot(y_true, depth=tf.shape(cosine)[1])

        # 生成高斯噪声并截断 - 使用纯TensorFlow实现
        noise = tf.random.normal(
            shape=tf.shape(cosine),
            mean=0.0,
            stddev=1 / 3,  # 标准差1/3
            dtype=tf.float32
        )
        noise = tf.clip_by_value(noise, -1.0, 1.0)

        # 计算类相关噪声幅度
        max_m = tf.reduce_max(self.m_list)
        noise_scale = self.noise_mul * tf.abs(noise) * (self.m_list / max_m)

        # 应用高斯云化调整
        adjusted_cosine = cosine - noise_scale

        # 为目标类添加额外间隔
        output = tf.where(
            index > 0,
            adjusted_cosine - self.m,
            adjusted_cosine
        )

        # 缩放输出
        scaled_output = self.s * output

        # 计算交叉熵损失
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=scaled_output
        )

        # 根据阶段选择损失函数
        if self.train_cls:
            p = tf.exp(-ce_loss)
            focal_loss = tf.pow(1 - p, self.gamma) * ce_loss
            return tf.reduce_mean(focal_loss)
        else:
            return tf.reduce_mean(ce_loss)


def create_model(length, num_classes=9, cls_num_list=None):
    # 输入层
    img_input = Input(shape=(length, 1))

    # 特征提取层
    x = Conv1D(2, 25, kernel_initializer=he_uniform(), activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer=he_uniform(), activation='selu')(x)
    x = Dense(10, kernel_initializer=he_uniform(), activation='selu')(x)
    x = Dense(4, kernel_initializer=he_uniform(), activation='selu')(x)

    # 使用 NormedLinear 作为分类层
    cosine_output = NormedLinear(num_classes, name='normed_linear')(x)

    # 创建模型
    model = Model(img_input, cosine_output)

    return model


ajust_flag = False
rank_logs = []
all_rank_logs = []
corr_logs = []
all_corr_logs = []
loss_logs = []
kl_loss_logs = []
dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
leakage_model = 'HW'
attack_model = 'CNN'  # MLP/CNN
sigma_hw = 0  # sigma for the HW leakage model
sigma_id = 0  # sigma for the ID leakage model
num_traces_attacks = 3000
epoch_count = 0
data_arguementation = False  # enable/disbale data arguementation
data_arguementation_level = 0.25  # data arguementation level
tro = 1
best_weights = None
count = 0
(X_profiling, X_attack), (Y_profiling, Y_attack), (
    plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
    leakage_model,
    data_arguementation,
    data_arguementation_level,
    attack_model, dataset,
    sigma_hw, sigma_id)
model = create_model(X_profiling.shape[1], 9)
Y_profiling_int = np.argmax(Y_profiling[:, :9], 1)
label_freq = {}
for key in Y_profiling_int:
    label_freq[key] = label_freq.get(key, 0) + 1
cls_num_list = dict(sorted(label_freq.items()))
cls_num_list = np.array(list(cls_num_list.values()))
prior = cls_num_list / cls_num_list.sum()
adjustments = np.log(prior + 1e-12)
adjustments_tf = adjustments * tro

gcl_feature_loss = GCLLoss(
    cls_num_list=cls_num_list,
    m=0.5,
    s=30,
    train_cls=False,
    noise_mul=0.5,
    gamma=0.,
    name='gcl_feature_loss'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss=gcl_feature_loss
)
lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                        maximum_momentum=None, minimum_momentum=None, verbose=True)

history = model.fit(
    x=X_profiling[:50000], y= np.argmax(Y_profiling[:50000, :9], 1),
    batch_size=128,
    epochs=50,
    callbacks=[lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]
)

for layer in model.layers:
        if layer.name != 'normed_linear':  # 只保留分类层可训练
            layer.trainable = False

gcl_classifier_loss = GCLLoss(
    cls_num_list=cls_num_list,
    m=0.1,
    s=20,
    train_cls=True,
    noise_mul=0.3,
    gamma=1.0,
    name='gcl_classifier_loss'
)

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=gcl_classifier_loss,
    )


history = model.fit(
    x=X_profiling[:50000], y=np.argmax(Y_profiling[:50000, :9], 1),
    batch_size=128,
    epochs=50,
    callbacks=[lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]
)


print("**************************************************************************************************")
all_predictions = model.predict(X_attack[5000:])

attack_traces = perform_attacks(plt_attack[5000:], all_predictions, "attack_traces",
                                leakage_model, dataset, num_traces_attacks)

if attack_traces[-1, correct_key] > 0:
    print("攻击失败")
    print("GE:", attack_traces[-1, correct_key])

else:
    print("攻击成功")
    print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
