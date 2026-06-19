import tensorflow as tf
from tensorflow.keras import layers, Model
from data_load_ascad_50000 import read_data
from SCA_util_new import perform_attacks
import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
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
            # attack_traces = perform_attacks(plt_attack[:5000], y_pred_valid_metric_all,
            #                                 "all_attack_traces",
            #                                 leakage_model,
            #                                 dataset,
            #                                 num_traces_attacks,
            #                                 shuffled_indices_list
            #                                 )
            #
            # if attack_traces[-1, correct_key] > 0:
            #     print("攻击失败")
            #     print("GE:", attack_traces[-1, correct_key])
            #
            # else:
            #     print("攻击成功")
            #     print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(plt_attack[:5000],
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
                    if count == 11:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)


class OnlineDistillationModel(Model):
    def __init__(self, num_experts=6, num_classes=9, temperature=3, alpha=1):
        super(OnlineDistillationModel, self).__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.alpha = alpha

        # 创建多个专家模型
        self.experts = []
        for i in range(num_experts):
            # 每个专家可以是不同的架构

            # model = tf.keras.Sequential([
            #     layers.Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same',
            #                   input_shape=(1400, 1), name="expert" + str(i) + "cnn1D"),
            #     layers.AveragePooling1D(25, strides=25, name="expert" + str(i) + "averagePooling1D"),
            #     layers.Flatten(name="expert" + str(i) + "flatten"),
            #     layers.Dense(30, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn0"),
            #     layers.Dense(30, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn1"),
            #     layers.Dense(20, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn2"),
            #     layers.Dense(9, name="expert" + str(i) + "dnn3")
            # ])
            # if i == 0 or i % 2 == 0:
            model = tf.keras.Sequential([
                layers.Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same',
                              input_shape=(700, 1), name="expert" + str(i) + "cnn1D"),
                layers.AveragePooling1D(4, strides=4, name="expert" + str(i) + "averagePooling1D"),
                layers.Flatten(name="expert" + str(i) + "flatten"),
                layers.Dense(15, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn0"),
                layers.Dense(10, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn1"),
                layers.Dense(4, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn2"),
                layers.Dense(num_classes, name="expert" + str(i) + "dnn3")
        ])
            # elif i == 1 or i % 3 == 0:
            #     model = tf.keras.Sequential([layers.Conv1D(16, 100, kernel_initializer='he_uniform', activation='selu', padding='same',input_shape=(700, 1)),
            #         layers.AveragePooling1D(25, strides=25),
            #         layers.Flatten(name='flatten'),
            #         layers.Dense(15, kernel_initializer='he_uniform', activation='selu'),
            #         layers.Dense(4, kernel_initializer='he_uniform', activation='selu'),
            #         layers.Dense(4, kernel_initializer='he_uniform', activation='selu'),
            #         layers.Dense(9)])

            # elif i == 2:
            #     model = tf.keras.Sequential([
            #         layers.Dense(496, activation='relu',input_shape=(700,)),
            #         layers.Dense(496, activation='relu'),
            #         layers.Dense(136, activation='relu'),
            #         layers.Dense(288, activation='relu'),
            #         layers.Dense(552, activation='relu'),
            #         layers.Dense(408, activation='relu'),
            #         layers.Dense(232, activation='relu'),
            #         layers.Dense(856, activation='relu'),
            #         layers.Dense(9)])

            self.experts.append(model)

        # 指标跟踪
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_loss_tracker = tf.keras.metrics.Mean(name="ce_loss")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        # self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_loss_tracker,
            self.distill_loss_tracker,
        ]

    def call(self, inputs):
        # 在推理时，可以使用所有专家的平均预测
        all_logits = [tf.nn.softmax(expert(inputs), 1) for expert in self.experts]
        return all_logits

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            # 1. 前向传播：获取所有专家的预测
            all_logits = [tf.cast(expert(x, training=True), dtype=tf.double) for
                          expert in self.experts]

            # 2. 计算分类损失（交叉熵）
            ce_losses = []
            for logits in all_logits:
                ce_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y, logits + adjustments_tf,
                                                                    from_logits=True)
                )
                ce_losses.append(ce_loss)
            ce_loss = tf.reduce_mean(ce_losses)

            # 3. 计算蒸馏损失（KL散度）
            distill_loss = 0.0

            # 计算所有专家对的KL散度
            for i in range(self.num_experts):
                for j in range(self.num_experts):
                    if i == j:
                        continue

                    # 软化预测概率
                    teacher_logits = tf.nn.softmax(all_logits[i] + adjustments_tf, 1) / self.temperature
                    student_logits = tf.nn.softmax(all_logits[j] + adjustments_tf, 1) / self.temperature

                    # KL散度损失
                    kl_loss = tf.reduce_mean(
                        tf.keras.losses.kullback_leibler_divergence(
                            teacher_logits,
                            student_logits
                        )
                    )
                    distill_loss += kl_loss

            # 4. 总损失 = 分类损失 + 蒸馏损失
            total_loss = ce_loss + self.alpha * distill_loss / (self.num_experts * (self.num_experts - 1))
            # total_loss = ce_loss - self.alpha * distill_loss / (self.num_experts - 1)

        # 5. 计算梯度并更新所有专家
        all_trainable_vars = []
        for expert in self.experts:
            all_trainable_vars.extend(expert.trainable_variables)

        gradients = tape.gradient(total_loss, all_trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, all_trainable_vars))

        # 6. 更新指标
        self.loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.distill_loss_tracker.update_state(distill_loss)

        # 使用平均预测计算准确率
        # avg_logits = tf.reduce_mean(all_logits, axis=0)
        # self.acc_tracker.update_state(y, avg_logits)

        return {m.name: m.result() for m in self.metrics}


num_experts = 10
number_of_best_experts = 4
distillation_model = OnlineDistillationModel(
    num_experts=num_experts,
    num_classes=9,
    temperature=1,
    alpha=0.6
)

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
num_traces_attacks = 800
np.random.seed(2075)
shuffled_indices_list = [np.random.permutation(np.arange(num_traces_attacks)) for _ in range(20)]

epoch_count = 0
data_arguementation = False  # enable/disbale data arguementation
data_arguementation_level = 0.25  # data arguementation level
tro = 0.715
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

distillation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3)
)
lr_manager = OneCycleLR(len(X_attack[5000:]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                        maximum_momentum=None, minimum_momentum=None, verbose=True)

history = distillation_model.fit(
    x=X_profiling[:50000], y=np.argmax(Y_profiling[:50000, :9], 1),
    batch_size=128,
    epochs=50,
    callbacks=[lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]
)

print("**************************************************************************************************")
all_predictions = distillation_model.predict(X_attack[5000:])

attack_traces = perform_attacks(plt_attack[5000:], all_predictions,
                                "all_attack_traces",
                                leakage_model,
                                dataset,
                                num_traces_attacks,
                                shuffled_indices_list
                                )

if attack_traces[-1, correct_key] > 0:
    print("攻击失败")
    print("GE:", attack_traces[-1, correct_key])

else:
    print("攻击成功")
    print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
print()
# predictions = predictions - 0.25 * adjustments
# predictions = tf.nn.softmax(predictions)
# attack_traces = perform_attacks(plt_attack[5000:], predictions, "attack_traces",
#                                 leakage_model, dataset, num_traces_attacks)
#
# if attack_traces[-1, correct_key] > 0:
#     print("攻击失败")
#     print("GE:", attack_traces[-1, correct_key])
#
# else:
#     print("攻击成功")
#     print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
