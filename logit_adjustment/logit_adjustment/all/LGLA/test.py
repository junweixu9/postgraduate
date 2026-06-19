from data_load_ascad_50000 import read_data
import tensorflow as tf
import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from SCA_util_new import perform_attacks
from tensorflow.keras.optimizers import Adam, RMSprop


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
            y_pred_valid_metric_all = self.model.predict(X_attack_valid_metric)
            # all_logits = [tf.nn.softmax(y_pred, 1) for y_pred in y_pred_valid_metric_all]
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric_all,
                                                                                    "all",
                                                                                    leakage_model,
                                                                                    dataset,
                                                                                    num_traces_attacks,
                                                                                    shuffled_indices_list
                                                                                    )

            epoch_count = epoch_count + 1

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
                    if count == 12:
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
    expert_logits_with_adjustment = expert_logits + adjustments_stack0_tf
    expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
    loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)

    return loss


def adjustment_local1_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits_with_adjustment = expert_logits + adjustments_stack1_tf
    expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
    loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)

    return loss


def adjustment_local2_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits_with_adjustment = expert_logits + adjustments_stack2_tf
    expert_logits_with_softmax_adjustment = tf.nn.softmax(expert_logits_with_adjustment, 1)
    loss = tk.backend.categorical_crossentropy(expert_logits_with_softmax_adjustment, y_true)

    return loss


def adjustment_global_expert_loss(y_true, y_pred):
    y_true = y_true[:, :9]
    y_true = tf.cast(y_true, dtype=tf.double)
    expert_logits = tf.cast(y_pred, dtype=tf.double)
    expert_logits = expert_logits + adjustments_tf
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

    x3 = Dense(15, kernel_initializer='he_uniform', activation='selu', name='expert3_D0')(x)
    x3 = Dense(10, kernel_initializer='he_uniform', activation='selu', name='expert3_D1')(x3)
    x3 = Dense(4, kernel_initializer='he_uniform', activation='selu', name='expert3_D2')(x3)
    x3 = Dense(9, name='expert3_D3')(x3)

    model = Model(inputs=img_input, outputs=[x0, x1, x2, x3])
    # model = Model(inputs=img_input, outputs =  x2)
    model.compile(loss={'expert0_D3': adjustment_local0_expert_loss,
                        'expert1_D3': adjustment_local1_expert_loss,
                        'expert2_D3': adjustment_local2_expert_loss,
                        'expert3_D3': adjustment_global_expert_loss
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
    np.random.seed(2075)
    shuffled_indices_list = [np.random.permutation(np.arange(num_traces_attacks)) for _ in range(20)]
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    tro = 30
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
    adjustments_tf = tf.cast(adjustments * tro, dtype=tf.double)

    s = 30
    tau = 30
    num_experts = 4
    learning_rate = 5e-3

    mask_cls = [[False, False, False, False, True, False, False, False, False],
                [False, False, True, True, False, True, True, False, False],
                [True, True, False, False, False, False, False, True, True]]
    # region_points = get_region_points(num_experts, cls_num_list)
    # print(region_points)
    # for i in range(len(region_points)):
    #     lower, upper = region_points[i]
    #     mask = (cls_num_list > lower) & (cls_num_list <= upper)
    #     mask_cls.append(mask)

    adjustments_stack0 = np.zeros_like(adjustments) + np.max(adjustments)
    adjustments_stack0[mask_cls[0]] = adjustments[mask_cls[0]]
    adjustments_stack0_tf = tf.cast(adjustments_stack0, dtype=tf.double)

    adjustments_stack1 = np.zeros_like(adjustments) + np.max(adjustments)
    adjustments_stack1[mask_cls[1]] = adjustments[mask_cls[1]]
    adjustments_stack1_tf = tf.cast(adjustments_stack1, dtype=tf.double)

    adjustments_stack2 = np.zeros_like(adjustments) + np.max(adjustments)
    adjustments_stack2[mask_cls[2]] = adjustments[mask_cls[2]]
    adjustments_stack2_tf = tf.cast(adjustments_stack2, dtype=tf.double)

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
