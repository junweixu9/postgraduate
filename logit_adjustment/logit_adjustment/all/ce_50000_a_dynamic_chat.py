
from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import DL_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
from datetime import datetime
from clr import OneCycleLR

import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense
from tensorflow.keras.models import Model

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
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
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
                    if count == 10:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)


class DLLALoss(tf.keras.losses.Loss):
    def __init__(self, model, num_classes=9, tau=2.0, lambda_etf=0.3, name="dlla_loss"):
        super().__init__(name=name)
        self.model = model
        self.num_classes = num_classes
        self.tau = tau
        self.lambda_etf = lambda_etf

    def compute_rank_metric(self, features, classifier_weights, y_true):
        # 使用TensorFlow操作处理所有计算
        class_indices = tf.cast(tf.argmax(y_true, axis=1), dtype=tf.int32)

        # 转置分类器权重为[hidden_dim, num_classes]
        class_prototypes = tf.transpose(classifier_weights)

        # 使用TensorFlow向量化操作代替循环
        masks = tf.one_hot(class_indices, depth=self.num_classes, dtype=tf.bool)
        batch_ranks = tf.TensorArray(tf.float32, size=self.num_classes)

        for c in tf.range(self.num_classes):
            # 获取当前类别的样本特征
            class_features = tf.boolean_mask(features, masks[:, c])

            # 添加类原型
            prototype = tf.expand_dims(class_prototypes[:, c], 0)
            augmented_features = tf.concat([class_features, prototype], axis=0)

            # 计算秩指标
            normalized = tf.nn.l2_normalize(augmented_features, axis=1)
            similarity = tf.matmul(normalized, normalized, transpose_b=True)
            s_matrix = tf.clip_by_value(0.5 * (similarity + 1.0), 0.0, 1.0)
            rank = tf.linalg.norm(s_matrix, ord='nuc', axis=(-2, -1))

            batch_ranks = batch_ranks.write(c, rank)

        all_ranks = batch_ranks.stack()
        total_rank = tf.reduce_sum(all_ranks) + 1e-8
        return all_ranks / total_rank

    def call(self, y_true, y_pred):
        # 从模型获取分类器权重（关键修改）
        classifier_layer = self.model.get_layer('logits')
        classifier_weights = classifier_layer.kernel

        # 分解模型输出
        features = y_pred  # 模型直接输出特征

        # 调整特征维度
        features = tf.reshape(features, [tf.shape(features)[0], -1])

        # 计算动态调整项
        pi_y = self.compute_rank_metric(features, classifier_weights, y_true)
        adjustments = self.tau * tf.math.log(pi_y + 1e-8)

        # 应用logit调整
        adjusted_logits = classifier_layer(features) + adjustments

        # 计算交叉熵损失
        ce_loss = tf.keras.losses.categorical_crossentropy(
            y_true,
            tf.nn.softmax(adjusted_logits),
            from_logits=False
        )

        # ETF正则化项
        etf_constraint = tf.reduce_sum(tf.square(tf.reduce_sum(classifier_weights, axis=1)))

        return ce_loss + self.lambda_etf * etf_constraint


def ascad_f_hw_cnn_rs(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    # 示例卷积层，可根据用户原结构进行调整。
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    # 特征向量提取层
    features = tf.keras.layers.Dense(256, activation='relu')(x)
    # 最终分类层，使用无偏置 (bias) 便于ETF约束实现
    logits = tf.keras.layers.Dense(num_classes, use_bias=False, name='classifier')(features)
    # 模型同时输出logits和features
    model = tf.keras.Model(inputs=inputs, outputs=[logits, features])
    return model

if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 3000
    epoch_count = 0
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

    # adjustments = compute_adjustment(Y_profiling, tro)
    # adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    """创建神经网络模型"""
    """ select the output_metric function """
    model = ascad_f_hw_cnn_rs(input_shape, num_classes)
    # 构建模型
    model = build_model_with_features(X_profiling.shape[1:])

    # 创建损失函数并绑定模型
    dlla_loss = DLLALoss(model=model, num_classes=9, tau=2.0, lambda_etf=0.3)
    # 编译模型时使用自定义损失
    model.compile(
        loss=dlla_loss,
        optimizer='adam',
        metrics=None
    )

    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
        # all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
    ]
    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 128,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :9], batch_size=128, verbose=2,
                        epochs=epoch,
                        callbacks=callback
                        )
    history.history.keys()
    loss = history.history['loss']

    # Attack
    print('======Attack======')
    predictions = model.predict(X_attack[5000:])
    # predictions = predictions - 0.25 * adjustments
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
                                    leakage_model, dataset, num_traces_attacks)
    log = open('F:/result/cnn/a/dynamic.txt', mode='a',
               encoding='utf-8')

    print(file=log)
    print(file=log)

    current_time_mid = datetime.now()
    day_mid = current_time_mid.day
    hour_mid = current_time_mid.hour
    minute_mid = current_time_mid.minute
    experiment_time_mid = 'time_is{}_{}_{}'.format(int(day_mid),
                                                   int(hour_mid),
                                                   int(minute_mid)
                                                   )


    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key], file=log)

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)

    current_time_end = datetime.now()
    day_end = current_time_end.day
    hour_end = current_time_end.hour
    minute_end = current_time_end.minute
    experiment_time_end = 'time_is{}_{}_{}'.format(int(day_end),
                                                   int(hour_end),
                                                   int(minute_end)
                                                   )
    print(experiment_time, file=log)
    print(experiment_time_mid, file=log)
    print(experiment_time_end, file=log)


    log.close()





