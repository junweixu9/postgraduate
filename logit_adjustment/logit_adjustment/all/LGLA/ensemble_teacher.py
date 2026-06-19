import tensorflow as tf
from tensorflow.keras import layers, Model
from data_load_ascad_50000 import read_data
from SCA_util_new import perform_attacks
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.layers import *

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
            student_model = tf.keras.Sequential([
                layers.Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same',
                              input_shape=(700, 1), name="expert" + str(i) + "cnn1D"),
                layers.AveragePooling1D(4, strides=4, name="expert" + str(i) + "averagePooling1D"),
                layers.Flatten(name="expert" + str(i) + "flatten"),
                layers.Dense(15, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn0"),
                layers.Dense(10, kernel_initializer='he_uniform', activation='selu', name="expert" + str(i) + "dnn1"),
                layers.Dense(num_classes, name="expert" + str(i) + "dnn3")
            ])
            self.experts.append(student_model)

        for i in range(1, num_experts + 1):
            # 每个专家可以是不同的架构
            teacher_model = tf.keras.Sequential([
                layers.Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same',
                              input_shape=(700, 1), name="expert" + str(i + num_experts - 1) + "cnn1D"),
                layers.AveragePooling1D(4, strides=4, name="expert" + str(i + num_experts - 1) + "averagePooling1D"),
                layers.Flatten(name="expert" + str(i) + "flatten"),
                layers.Dense(15, kernel_initializer='he_uniform', activation='selu',
                             name="expert" + str(i + num_experts - 1) + "dnn0"),
                layers.Dense(10, kernel_initializer='he_uniform', activation='selu',
                             name="expert" + str(i + num_experts - 1) + "dnn1"),
                layers.Dense(4, kernel_initializer='he_uniform', activation='selu',
                             name="expert" + str(i + num_experts - 1) + "dnn2"),
                layers.Dense(num_classes, name="expert" + str(i + num_experts - 1) + "dnn3")
            ])
            """最佳模型的存储地址"""
            test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}false'.format(dataset, leakage_model,
                                                                                              epoch,
                                                                                              128,
                                                                                              output_metric,
                                                                                              )

            """开始训练"""

            model_root = 'E:/logit_backup/ensemble_model/' + test_info + str(i) + '.h5'
            teacher_model.load_weights(model_root)
            self.experts.append(teacher_model)

        # 指标跟踪
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_loss_student_sum_tracker = tf.keras.metrics.Mean(name="ce_loss_student")
        self.ce_loss_teacher_sum_tracker = tf.keras.metrics.Mean(name="ce_loss_teacher")
        self.kl_loss_teacher_sum_student_sum_tracker = tf.keras.metrics.Mean(name="kl_loss_teacher_student")
        self.mse_loss_teacher_sum_student_sum_tracker = tf.keras.metrics.Mean(name="mse_loss_teacher_student")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        self.mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")

        # self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_loss_student_sum_tracker,
            self.ce_loss_teacher_sum_tracker,
            self.kl_loss_teacher_sum_student_sum_tracker,
            self.mse_loss_teacher_sum_student_sum_tracker,
            self.distill_loss_tracker,
            self.mse_loss_tracker,
        ]

    def call(self, inputs):
        # 在推理时，可以使用所有专家的平均预测

        all_logits = [tf.nn.softmax(expert(inputs), 1) for expert in self.experts]

        return all_logits

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:

            all_logits = [tf.cast(expert(x, training=True), dtype=tf.double) for expert in self.experts]

            # CE_student_sum
            all_logits_student_sum = all_logits[0]
            for logits in all_logits[1:self.num_experts]:
                all_logits_student_sum = all_logits_student_sum + logits

            ce_loss_student_sum = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, all_logits_student_sum,
                                                                from_logits=True)
            )
            # CE_teacher_sum
            all_logits_teacher_sum = all_logits[self.num_experts]
            for logits in all_logits[self.num_experts + 1:]:
                all_logits_teacher_sum = all_logits_teacher_sum + logits

            ce_loss_teacher_sum = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, all_logits_teacher_sum,
                                                                from_logits=True)
            )

            # KL_teacher_sum_student_sum
            kl_loss_teacher_sum_student_sum = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(
                tf.nn.softmax(all_logits_student_sum, 1), tf.nn.softmax(all_logits_teacher_sum / self.temperature, 1)
            ))

            # MSE_teacher_sum_student_sum
            mse_loss_teacher_sum_student_sum = tf.reduce_mean(tf.keras.losses.MSE(all_logits_student_sum,
                                                                                  all_logits_teacher_sum
                                                                                  ))

            distill_loss = 0.0
            mse_loss = 0.0

            for i in range(self.num_experts):
                kl_loss_teacher_sum_student_sum = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(
                    tf.nn.softmax(all_logits[i], 1),
                    tf.nn.softmax(all_logits[i + self.num_experts] / self.temperature, 1)
                ))
                distill_loss = distill_loss + kl_loss_teacher_sum_student_sum

                mse_loss_teacher_sum_student_sum = tf.reduce_mean(tf.keras.losses.MSE(
                    all_logits[i], all_logits[i + self.num_experts]
                ))

                mse_loss = mse_loss + mse_loss_teacher_sum_student_sum

            total_loss = 0.5 * ce_loss_student_sum + 0.5 * ce_loss_teacher_sum + 0.6 * (
                        kl_loss_teacher_sum_student_sum + mse_loss_teacher_sum_student_sum
                        + distill_loss + mse_loss)

        # 5. 计算梯度并更新所有专家
        all_trainable_vars = []
        for expert in self.experts:
            all_trainable_vars.extend(expert.trainable_variables)

        gradients = tape.gradient(total_loss, all_trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, all_trainable_vars))

        # 6. 更新指标
        self.loss_tracker.update_state(total_loss)
        self.ce_loss_student_sum_tracker.update_state(ce_loss_student_sum)
        self.ce_loss_teacher_sum_tracker.update_state(ce_loss_teacher_sum)
        self.kl_loss_teacher_sum_student_sum_tracker.update_state(kl_loss_teacher_sum_student_sum)
        self.mse_loss_teacher_sum_student_sum_tracker.update_state(mse_loss_teacher_sum_student_sum)
        self.distill_loss_tracker.update_state(distill_loss)
        self.mse_loss_tracker.update_state(mse_loss)

        return {m.name: m.result() for m in self.metrics}


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
tro = 1
best_weights = None
count = 0
num_experts = 3
number_of_best_experts = 3
num_classes = 9
epoch = 50
output_metric = "all"
distillation_model = OnlineDistillationModel(
    num_experts=num_experts,
    num_classes=9,
    temperature=2,
    alpha=0.4
)

(X_profiling, X_attack), (Y_profiling, Y_attack), (
    plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
    leakage_model,
    data_arguementation,
    data_arguementation_level,
    attack_model, dataset,
    sigma_hw, sigma_id)

distillation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3)
)
lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                        maximum_momentum=None, minimum_momentum=None, verbose=True)

history = distillation_model.fit(
    x=X_profiling[:50000], y=np.argmax(Y_profiling[:50000, :9], 1),
    batch_size=128,
    epochs=50,
    callbacks=[lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]
)
print("**************************************************************************************************")
all_predictions = distillation_model.call(X_attack[:5000])

attack_traces = perform_attacks(plt_attack[:5000], all_predictions,
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