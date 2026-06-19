from data_load_ascad_50000 import read_data
import tensorflow as tf
import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from SCA_util import perform_attacks


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
            y_pred_valid_metric_0, y_pred_valid_metric_1, y_pred_valid_metric_2 = \
                y_pred_valid_metric_all[0], y_pred_valid_metric_all[1], y_pred_valid_metric_all[2]

            y_pred_valid_metric_0 = tf.nn.softmax(y_pred_valid_metric_0, 1)
            y_pred_valid_metric_1 = tf.nn.softmax(y_pred_valid_metric_1, 1)
            y_pred_valid_metric_2 = tf.nn.softmax(y_pred_valid_metric_2, 1)

            # print("0:",y_pred_valid_metric_0)
            # print("1:",y_pred_valid_metric_1)
            # print("2:",y_pred_valid_metric_2)
            # print("3:",y_pred_valid_metric_3)
            y_pred_valid_metric = (
                                              y_pred_valid_metric_0 + y_pred_valid_metric_1 + y_pred_valid_metric_2 ) / 4

            # y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)
            y_pred_valid = tf.argmax(y_pred_valid_metric, 1)
            comparisons = tf.equal(y_pred_valid, y_pred_valid[0])

            # 检查所有的比较结果是否都是True
            all_equal = tf.reduce_all(comparisons)

            print(all_equal.numpy())

            unique_values, _, counts = tf.unique_with_counts(y_pred_valid)

            # 计算每个值的频率
            total_count = tf.size(y_pred_valid)
            frequencies = tf.cast(counts, tf.float32) / tf.cast(total_count, tf.float32)

            # 输出结果
            for value, count, freq in zip(unique_values.numpy(), counts.numpy(), frequencies.numpy()):
                print(f"Value: {value}, Count: {count}, Frequency: {freq:.2f}")

            Y_attack_valid = Y_attack_valid[:, :9]
            Y_attack_valid_int = tf.argmax(Y_attack_valid, 1)
            correct = tf.equal(y_pred_valid, Y_attack_valid_int)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Validation Accuracy:', accuracy)
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


def get_region_points(num_experts,
                      cls_num_list):  # Divide data sets equally according to the number of samples # 确保可以访问 num_experts
    region_num = sum(cls_num_list) // (num_experts - 1)
    sort_list = np.sort(cls_num_list)
    region_points = []
    now_sum = 0
    for i in range(len(sort_list)):
        now_sum += sort_list[i]
        if now_sum > region_num:
            region_points.append(sort_list[i])
            now_sum = 0
    region_points = list(reversed(region_points))

    region_left_right = []

    for i in range(len(region_points)):
        if i == 0:
            region_left_right.append([region_points[i], max(cls_num_list)])
        else:
            region_left_right.append([region_points[i], region_points[i - 1]])
    region_left_right.append([0, region_points[len(region_points) - 1]])

    return region_left_right


def adjustment_local0_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits_with_adjustment = expert_logits
    expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
    loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)

    return loss


def adjustment_local1_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits_with_adjustment = expert_logits + adjustments_tf_0
    expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
    loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)

    return loss


# def adjustment_local2_expert_loss(y_true, y_pred):
#     y_true = y_true[:, :9]
#     y_true = tf.cast(y_true, dtype=tf.double)
#     expert_logits = tf.cast(y_pred, dtype=tf.double)
#     expert_logits_with_adjustment = expert_logits + adjustments_stack2_tf
#     expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
#     loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)
#
#     return loss


def adjustment_global_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits = expert_logits + adjustments_tf_0 - adjustments_inverse
    expert_logits = tf.nn.softmax(expert_logits, 1)
    loss = tk.backend.categorical_crossentropy(y_true, expert_logits)
    return loss


def ascad_f_hw_cnn_rs(metric):
    img_input = Input(shape=(X_profiling.shape[1], 1))
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)

    x0 = Dense(15, kernel_initializer='he_uniform', activation='selu', name='expert0_D0')(x)
    x0 = Dense(10, kernel_initializer='he_uniform', activation='selu', name='expert0_D1')(x0)
    x0 = Dense(4, kernel_initializer='he_uniform', activation='selu', name='expert0_D2')(x0)
    x0 = Dense(9, name='expert0_D3')(x0)
    #
    x1 = Dense(15, kernel_initializer='he_uniform', activation='selu', name='expert1_D0')(x)
    x1 = Dense(10, kernel_initializer='he_uniform', activation='selu', name='expert1_D1')(x1)
    x1 = Dense(4, kernel_initializer='he_uniform', activation='selu', name='expert1_D2')(x1)
    x1 = Dense(9, name='expert1_D3')(x1)

    x2 = Dense(15, kernel_initializer='he_uniform', activation='selu', name='expert2_D0')(x)
    x2 = Dense(10, kernel_initializer='he_uniform', activation='selu', name='expert2_D1')(x2)
    x2 = Dense(4, kernel_initializer='he_uniform', activation='selu', name='expert2_D2')(x2)
    x2 = Dense(9, name='expert2_D3')(x2)

    model = Model(inputs=img_input, outputs=[x0, x1, x2])
    # model = Model(inputs=img_input, outputs =  x2)
    model.compile(loss={'expert0_D3': adjustment_local0_expert_loss,
                        'expert1_D3': adjustment_local1_expert_loss,
                        'expert2_D3': adjustment_global_expert_loss
                        }
                  , optimizer='adam', metrics=metric)

    model.summary()
    return model


if __name__ == '__main__':

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
    tro = 2
    best_weights = None
    count = 0
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    Y_profiling_int = np.argmax(Y_profiling[:, :9], 1)
    label_freq = {}
    for key in Y_profiling_int:
        label_freq[key] = label_freq.get(key, 0) + 1
    cls_num_list = dict(sorted(label_freq.items()))
    cls_num_list = np.array(list(cls_num_list.values()))
    prior = cls_num_list / cls_num_list.sum()
    adjustments = np.log(prior + 1e-12)
    adjustments_tf_0 = tf.cast(adjustments, dtype=tf.double)
    # adjustments_tf_1 = tf.cast(adjustments * tro, dtype=tf.double)

    value, idx0 = tf.math.top_k(prior, k=tf.shape(prior)[0], sorted=True)

    # 2. 对原始索引进行二次排序，获取逆序索引
    _, idx1 = tf.math.top_k(idx0, k=tf.shape(idx0)[0], sorted=True)

    # 3. 反转索引顺序（降序排列）
    idx2 = tf.range(tf.shape(prior)[0] - 1, -1, -1)  # 生成[2,1,0]等逆序索引
    idx2 = tf.gather(idx2, idx1)  # 根据二次排序结果调整逆序索引

    # 4. 根据最终索引提取逆先验概率
    inverse_prior = tf.gather(value, idx2)

    adjustments_inverse = np.log(inverse_prior + 1e-12)
    adjustments_inverse = tf.cast(adjustments_inverse * tro, dtype=tf.double)


    s = 30
    tau = 30
    num_experts = 4
    learning_rate = 5e-3

    # mask_cls = [[False, False, False, False, False, False, False, False, False],
    #             [False, False, False, False, False, False, False, False, False],
    #             [False, False, False, False, False, False, False, False, False]]
    # region_points = get_region_points(num_experts, cls_num_list)
    # print(region_points)
    # for i in range(len(region_points)):
    #     lower, upper = region_points[i]
    #     mask = (cls_num_list > lower) & (cls_num_list <= upper)
    #     mask_cls.append(mask)

    # adjustments_stack0 = np.zeros_like(adjustments) + np.max(adjustments)
    # adjustments_stack0[mask_cls[0]] = adjustments[mask_cls[0]]
    # adjustments_stack0_tf = tf.cast(adjustments_stack0, dtype=tf.double)

    # adjustments_stack1 = np.zeros_like(adjustments) + np.max(adjustments)
    # adjustments_stack1[mask_cls[1]] = adjustments[mask_cls[1]]
    # adjustments_stack1_tf = tf.cast(adjustments_stack1, dtype=tf.double)

    # adjustments_stack2 = np.zeros_like(adjustments) + np.max(adjustments)
    # adjustments_stack2[mask_cls[2]] = adjustments[mask_cls[2]]
    # adjustments_stack2_tf = tf.cast(adjustments_stack2, dtype=tf.double)

    model = ascad_f_hw_cnn_rs(None)
    lr_manager = OneCycleLR(len(X_attack[:5000]), 128, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callback = [
        lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
        # all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
    ]
    history = model.fit(x=X_profiling[:50000], y=Y_profiling[:50000, :9], batch_size=128, verbose=2,
                        epochs=50,
                        callbacks=callback
                        )
