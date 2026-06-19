from tensorflow.keras.optimizers import Adam
from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
from datetime import datetime
from clr import OneCycleLR
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import requests
import time
import tensorflow.keras as tk

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, AveragePooling1D, Dense,
    LayerNormalization, MultiHeadAttention, Add,
    GlobalAveragePooling1D, AlphaDropout
)
from tensorflow.keras.models import Model

def ascad_f_hw_cnn_rs_selu_attn_stable(length):
    img_input = Input(shape=(length, 1), name="trace")

    # 1) 升通道（让 attention 有空间），并改 initializer
    x = Conv1D(16, 25, kernel_initializer='lecun_normal',
               activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)   # (T/4, 16)

    # 2) 极轻 self-attention（先从最保守配置开始）
    h = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(num_heads=1, key_dim=16, dropout=0.0)(h, h, h)
    x = Add()([x, attn])

    # 3) 用 GAP 替代 Flatten，降低过拟合
    x = GlobalAveragePooling1D()(x)

    # 原 Dense 栈（改 initializer + 用 AlphaDropout）
    x = Dense(15, kernel_initializer='lecun_normal', activation='selu')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(10, kernel_initializer='lecun_normal', activation='selu')(x)
    x = AlphaDropout(0.1)(x)
    x = Dense(4, kernel_initializer='lecun_normal', activation='selu')(x)

    output_layer = Dense(8, name='output_nmax_0')(x)  # logits
    return Model(img_input, output_layer, name="Model_Nmax_0_SelfAttn_Stable")


def ascad_f_hw_cnn_rs_self_attention(
    length,
    num_heads=2,
    key_dim=16,
    attn_dropout=0.1,
    ffn_units=32,
    ffn_dropout=0.1,
):
    img_input = Input(shape=(length, 1), name="trace")

    # ===== 原始 CNN（保持不变）=====
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)  # 现在 shape: (T', C) 其中 C=2

    # ===== Self-Attention Block（Transformer encoder 子块）=====
    # 1) Pre-LN + Multi-Head Self-Attention + Residual
    h = LayerNormalization(epsilon=1e-6, name="attn_ln")(x)
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attn_dropout,
        name="mha_self"
    )(h, h, h)  # self-attention: query=key=value
    attn_out = Dropout(attn_dropout, name="attn_dropout")(attn_out)
    x = Add(name="attn_residual")([x, attn_out])

    # 2) Pre-LN + FFN + Residual（轻量前馈网络）
    h2 = LayerNormalization(epsilon=1e-6, name="ffn_ln")(x)
    ffn = Dense(ffn_units, activation='selu', kernel_initializer='he_uniform', name="ffn_dense1")(h2)
    ffn = Dropout(ffn_dropout, name="ffn_dropout")(ffn)
    ffn = Dense(x.shape[-1], activation=None, kernel_initializer='he_uniform', name="ffn_dense2")(ffn)
    x = Add(name="ffn_residual")([x, ffn])

    # ===== 回到你原来的 Flatten + MLP（保持不变）=====
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)

    output_layer = Dense(8, name='output_nmax_0')(x)  # 8-bit logits（与原模型一致）
    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0_SelfAttn')

    return model

def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""

    pi = np.mean(Y_profiling, axis=0)  # (8,)
    adj = tro * np.log((pi + 1e-12) / (1 - pi + 1e-12))  # log-odds
    return adj.astype(np.float32)

def build_inception_sca_model_ches(length):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    # FC-8, sigmoid (Output)
    output_layer = Dense(8, name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')

    return model

def build_inception_sca_model_rand(length):
    img_input = Input(shape=(length, 1))
    x = Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)

    # FC-8, sigmoid (Output)
    output_layer = Dense(8, name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')


    return model


class BitCrossAttention(Layer):
    """
    为每个比特建立与其他比特的交叉注意力
    适用于明确建模8个比特之间的相互依赖
    """

    def __init__(self, num_bits=8, **kwargs):
        super(BitCrossAttention, self).__init__(**kwargs)
        self.num_bits = num_bits

    def build(self, input_shape):
        # 为每个比特创建query向量
        self.bit_queries = [
            self.add_weight(
                name=f'query_bit_{i}',
                shape=(input_shape[-1], 1),
                initializer='glorot_uniform',
                trainable=True
            ) for i in range(self.num_bits)
        ]

        # 共享的key和value矩阵
        self.W_k = self.add_weight(
            name='W_k',
            shape=(input_shape[-1], self.num_bits),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='W_v',
            shape=(input_shape[-1], self.num_bits),
            initializer='glorot_uniform',
            trainable=True
        )

        super(BitCrossAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, features)
        K = tf.matmul(x, self.W_k)  # (batch, num_bits)
        V = tf.matmul(x, self.W_v)  # (batch, num_bits)

        outputs = []
        for i, query_weight in enumerate(self.bit_queries):
            # 每个比特的query
            Q_i = tf.matmul(x, query_weight)  # (batch, 1)

            # 计算与所有比特的注意力
            attention_scores = Q_i * K  # (batch, num_bits)
            attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.num_bits, tf.float32))
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)

            # 加权求和
            context = tf.reduce_sum(attention_weights * V, axis=-1, keepdims=True)
            outputs.append(context)

        # 拼接所有比特的上下文
        output = tf.concat(outputs, axis=-1)  # (batch, num_bits)

        return output

    def get_config(self):
        config = super(BitCrossAttention, self).get_config()
        config.update({'num_bits': self.num_bits})
        return config

def ascad_f_hw_cnn_rs_classifier_chain(length):
    """
    在原 ascad_f_hw_cnn_rs 基础上加入 Classifier Chain：
    - 预测 bit_0
    - 预测 bit_1 时拼接 bit_0 的预测
    - ...
    - 预测 bit_7 时拼接 bit_0..bit_6 的预测
    输出：shape (B, 8) 的 logits（建议用 BCE(from_logits=True) 训练）
    """

    img_input = Input(shape=(length, 1), name="trace")

    # ===== 原始特征提取（保持不变） =====
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)

    x = BitCrossAttention(num_bits=8, name='bit_correlation_attention')(x)

    # 输出层（8个比特）
    output_layer = Dense(8, activation='sigmoid', name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0_with_Attention')

    return model


def ascad_f_hw_cnn_rs(length):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    # 4.1 投影 (Projection)
    # 目标是将特征变换为 (Batch_Size, 8, embed_dim) 的形状。
    # 8 代表我们有 8 个比特要预测 (Multilabel)。
    # embed_dim 是每个比特拥有的特征维度，这里设为 16 或 32。
    num_bits = 8
    embed_dim = 16

    # 全连接层将维度扩展到 8 * 16 = 128
    x = Dense(num_bits * embed_dim, kernel_initializer='he_uniform', activation='selu')(x)

    # Reshape 成 (Batch, Sequence=8, Channels=16) 以适应 MultiHeadAttention
    x_reshaped = Reshape((num_bits, embed_dim))(x)

    # 4.2 多头自注意力 (Multi-Head Attention)
    # query=x, value=x, key=x (因为是 Self-Attention)
    # num_heads=2 或 4，让模型从不同子空间捕捉比特间的依赖
    attn_out = MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x_reshaped, x_reshaped)

    # 4.3 残差连接与归一化 (Residual + Norm)
    # 这是 Transformer 模块的标准操作，防止梯度消失并稳定训练
    x = Add()([x_reshaped, attn_out])
    x = LayerNormalization()(x)

    # --- 5. 输出层 ---
    # 目前 x 的形状是 (Batch, 8, 16)
    # 我们需要对这 8 个向量分别做一个线性变换，得到 1 个 Logit

    # 这一步会将形状变为 (Batch, 8, 1)
    x = Dense(1, name='logit_projection')(x)

    # 展平为 (Batch, 8)
    # 注意：这里不加激活函数 (activation=None)，输出的是 Logits
    # 这样可以直接配合你之前写的 adjustment_loss 使用
    output_layer = Flatten(name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')


    return model

def build_inception_sca_model(length):

    # 定义模型输入
    INIT = 'he_uniform'
    ACT = 'selu'
    PAD = 'same'

    """
    复现表格第一列: NN architecture when Nmax = 0 (5 weights layers)
    """
    img_input = Input(shape=(length, 1))

    # --- Inception Block (Parallel Convs) ---
    # conv1-4, selu
    b1 = Conv1D(filters=4, kernel_size=1, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)
    # conv7-4, selu
    b2 = Conv1D(filters=4, kernel_size=7, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)
    # conv11-4, selu
    b3 = Conv1D(filters=4, kernel_size=11, kernel_initializer=INIT, activation=ACT, padding=PAD)(img_input)

    # concatenate
    x = Concatenate()([b1, b2, b3])

    # --- Subsequent Layers ---
    # conv1-4, selu
    x = Conv1D(filters=4, kernel_size=1, kernel_initializer=INIT, activation=ACT, padding=PAD)(x)
    # BN
    x = BatchNormalization()(x)

    # average pooling, 2 by 2 (Pool=2, Stride=2)
    # 注意：这里表格将pooling画在左侧跨越了多行，但在Nmax=0列它似乎直接接在BN后并在Flatten前
    x = AveragePooling1D(pool_size=2, strides=2)(x)

    # flatten
    x = Flatten()(x)

    # FC-10, selu
    x = Dense(10, kernel_initializer=INIT, activation=ACT)(x)
    # FC-10, selu
    x = Dense(10, kernel_initializer=INIT, activation=ACT)(x)

    # FC-8, sigmoid (Output)
    output_layer = Dense(8, name='output_nmax_0')(x)

    model = Model(inputs=img_input, outputs=output_layer, name='Model_Nmax_0')


    return model

def adjustment_loss(y_true, y_pred):

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return tf.reduce_mean(loss, axis=-1)


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    attack_model = 'CNN'  # MLP/CNN
    num_traces_attacks = 2000
    batch_size = 128
    epoch_count = 1
    # Select leakage model
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 100
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
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )
    adjust_flag = False

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        "HW",
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        0, 0)

    """创建神经网络模型"""
    """ select the output_metric function """

    tro = 0

    adjustments = compute_adjustment(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    model = ascad_f_hw_cnn_rs_selu_attn_stable(X_profiling.shape[1])
    optimizer = Adam(learning_rate=5e-3)  # Or any optimizer of your choice
    model.compile(optimizer=optimizer, loss=adjustment_loss)


    X_profiling = tf.cast(X_profiling, dtype=tf.double)
    Y_profiling = tf.cast(Y_profiling, dtype=tf.double)


    model.summary()
    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [lr_manager
                ]
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, "HW",
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    start_time = time.perf_counter()
    history = model.fit(x=X_profiling, y=Y_profiling, batch_size=batch_size, verbose=2,
                        epochs=epoch,
                        callbacks=callback
                        )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time} 秒")
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    import time

    # 记录开始时间
    start_time = time.perf_counter()
    predictions = model.predict(X_attack[5000:])
    predictions = 1.0 / (1.0 + np.exp(-predictions))

    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")



    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    "HW", dataset, num_traces_attacks)

    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"密钥恢复: {execution_time} 秒")


    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))

    current_time_end = datetime.now()
    day_end = current_time_end.day
    hour_end = current_time_end.hour
    minute_end = current_time_end.minute
    experiment_time_end = 'time_is{}_{}_{}'.format(int(day_end),
                                                   int(hour_end),
                                                   int(minute_end)
                                                   )
    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')