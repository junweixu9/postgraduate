from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import tensorflow as tf
from datetime import datetime
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *



# -------- 1. LRMCSLoss 自定义损失类 --------
class LRMCSLoss(tf.keras.losses.Loss):
    def __init__(self, final_classifier_layer, num_classes, tau, gamma, name="lrmcs_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.final_classifier_layer = final_classifier_layer
        self.num_classes = num_classes
        self.tau = tau
        self.gamma = gamma

    def call(self, y_true_original, y_pred_and_features):
        logits = y_pred_and_features[0]
        features_batch = y_pred_and_features[1]

        # --- 调试和形状检查 ---
        # tf.print("LRMCSLoss: Original logits shape:", tf.shape(logits))
        # tf.print("LRMCSLoss: Original features_batch shape:", tf.shape(features_batch))

        perform_main_logic = True  # 默认执行主损失逻辑

        current_rank = tf.rank(features_batch)
        if current_rank == 1:
            tf.print("LRMCSLoss WARNING: features_batch is 1D. Shape:", tf.shape(features_batch),
                     ". This might indicate an issue with model output or data pipeline.")
            expected_feature_dim = tf.shape(self.final_classifier_layer.kernel)[0]
            if tf.shape(features_batch)[0] == expected_feature_dim:
                features_batch = tf.reshape(features_batch, [1, -1])  # 尝试修正形状
                tf.print("LRMCSLoss INFO: Reshaped 1D features_batch to:", tf.shape(features_batch))
                # 修正后，再次检查秩是否变为2D，以防万一 (虽然reshape到[1,-1]应该是2D)
                if tf.rank(features_batch) < 2:
                    tf.print("LRMCSLoss ERROR: Reshaped features_batch is still not 2D. Shape:",
                             tf.shape(features_batch))
                    perform_main_logic = False  # 修正失败，标记为回退
            else:  # 1D 但长度不匹配，无法修正
                tf.print("LRMCSLoss ERROR: features_batch is 1D and its length does not match expected feature_dim.",
                         "Shape:", tf.shape(features_batch), "Expected feature_dim:", expected_feature_dim)
                perform_main_logic = False  # 标记为回退
        elif current_rank < 2:  # 例如，0D（标量）或其他不支持的秩
            tf.print("LRMCSLoss ERROR: features_batch rank is < 1 (e.g. 0D). Shape:", tf.shape(features_batch))
            perform_main_logic = False  # 标记为回退

        # 如果标记为回退，则计算并返回简单损失
        if not perform_main_logic:
            tf.print("LRMCSLoss INFO: Falling back to simple categorical crossentropy.")
            simple_loss = tf.keras.losses.categorical_crossentropy(y_true_original, logits, from_logits=True)
            return tf.reduce_mean(simple_loss)

        # --- 如果执行到这里，说明 features_batch 被认为是可用的 (至少是二维的) ---
        features_batch_normalized, _ = tf.linalg.normalize(features_batch, axis=1)

        classifier_weights = self.final_classifier_layer.kernel
        classifier_weights_normalized, _ = tf.linalg.normalize(classifier_weights, axis=0)

        y_true_indices = tf.argmax(y_true_original, axis=1)
        all_class_ranks = []

        for c in range(self.num_classes):
            class_c_mask = tf.equal(y_true_indices, c)
            class_c_features_in_batch = tf.boolean_mask(features_batch_normalized, class_c_mask)

            prototype_c = tf.reshape(classifier_weights_normalized[:, c], [1, -1])

            if tf.shape(class_c_features_in_batch)[0] == 0:
                augmented_features_c = prototype_c
            else:
                augmented_features_c = tf.concat([class_c_features_in_batch, prototype_c], axis=0)

            cos_sim_matrix = tf.matmul(augmented_features_c, augmented_features_c, transpose_b=True)
            cos_sim_matrix = tf.clip_by_value(cos_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7)
            s_ij_matrix = tf.minimum(1.0, (1.0 / self.gamma) * cos_sim_matrix)

            current_num_vectors = tf.shape(augmented_features_c)[0]
            if current_num_vectors == 0:
                rank_S_c = tf.constant(0.0, dtype=tf.float32)
            else:
                rank_S_c = tf.cast(tf.linalg.matrix_rank(s_ij_matrix, tol=1e-5), dtype=tf.float32)

            all_class_ranks.append(rank_S_c)

        sum_all_ranks = tf.reduce_sum(all_class_ranks) + 1e-12
        pi_tilde_array = tf.stack(all_class_ranks) / sum_all_ranks

        adjustments = self.tau * tf.math.log(pi_tilde_array + 1e-12)
        adjustments = tf.cast(adjustments, dtype=tf.float32)

        adjusted_logits = logits + adjustments

        loss = tf.keras.losses.categorical_crossentropy(y_true_original, adjusted_logits, from_logits=True)

        return tf.reduce_mean(loss)  # 主逻辑的最终返回


# -------- 2. 修改后的模型定义函数 --------
def ascad_f_hw_cnn_rs_modified(length, num_classes, feature_dim=4):
    """
    修改后的CNN模型，用于ASCAD数据集上的HW泄露模型。
    输出 logits 和分类器前的特征。
    返回模型和最终分类器层。
    """
    img_input = Input(shape=(length, 1), name="input_traces")
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    # 'feature_layer' 输出的是论文中的 'z'
    features_z = Dense(feature_dim, kernel_initializer='he_uniform', activation='selu', name='feature_layer')(x)

    # 最终分类器层
    final_classifier_dense_layer = Dense(num_classes, name='final_classifier')  # 使用 num_classes
    logits = final_classifier_dense_layer(features_z)

    # 创建一个同时输出 logits 和 features_z 的模型
    model = Model(inputs=img_input, outputs=[logits, features_z])

    # 编译步骤移到主函数中
    # model.summary() # 可以在主函数中调用 model.summary()
    return model, final_classifier_dense_layer


# -------- 3. 更新后的回调和主函数 --------
class all(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(all, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(all, self).set_params(params)

    def on_epoch_end(self, epoch_num, logs=None):  # 参数名改为 epoch_num 避免与全局 epoch 冲突
        if self.validation:
            global best_weights
            global count
            global epoch_count  # 注意：这个全局变量与函数的epoch参数重名了，已修改函数参数名
            logs['all_val'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]

            # 模型现在输出 [logits, features]，我们只需要 logits 进行评估
            predictions_list = self.model.predict(X_attack_valid_metric)
            y_pred_logits_valid_metric = predictions_list[0]  # 获取 logits

            y_pred_valid_metric_softmax = tf.nn.softmax(y_pred_logits_valid_metric)  # 对 logits 应用 softmax

            epoch_count = epoch_count + 1
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric_softmax,
                                                                                    # 使用 softmax后的结果
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
                    print("Early stopping count:", count)  # 明确打印信息
                    if count == 10:  # 早停阈值
                        print("Early stopping triggered.")
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)

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
    tro = 1.0  # 这是 LRMCSLoss 中的 tau (τ) 参数
    gamma_val = 0.99  # LRMCSLoss 中的 gamma (γ) 参数, 论文中常用的值
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
    model_feature_dim = 4
    # adjustments = compute_adjustment(Y_profiling, tro)
    # adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    """创建神经网络模型"""
    """ select the output_metric function """

    #
    model, final_classifier_layer = ascad_f_hw_cnn_rs_modified(
        length=X_profiling.shape[1],
        num_classes=classes,
        feature_dim=model_feature_dim  # 确保这个维度与模型定义一致
    )

    lrmcs_loss_instance = LRMCSLoss(
        final_classifier_layer=final_classifier_layer,
        num_classes=classes,
        tau=tro,
        gamma=gamma_val
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lrmcs_loss_instance, metrics=None)
    model.summary()  # 打印模型结构
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
    predictions_list_attack  = model.predict(X_attack[5000:])
    logits_for_attack = predictions_list_attack[0]
    # predictions = predictions - 0.25 * adjustments
    predictions = tf.nn.softmax(logits_for_attack)
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





