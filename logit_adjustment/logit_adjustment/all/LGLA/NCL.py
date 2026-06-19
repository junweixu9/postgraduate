import tensorflow as tf
from tensorflow.keras import layers, Model
from data_load_ascad_50000 import read_data
from SCA_util_e import perform_attacks as perform_attacks_e

import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *



def ensemble_ge_valid(all_predictions, num_experts):
    ge_all_validation = []
    k_ps_all = []
    for i in range(num_experts):
        attack_traces, kp_krs = perform_attacks_e(plt_attack[:5000], all_predictions[i], "attack_traceswithscore",
                                                  leakage_model, dataset, num_traces_attacks)

        ge_all_validation.append([np.argmax(attack_traces[:, correct_key] < 1), i])
        k_ps_all.append(kp_krs)

    sorted_models = sorted(ge_all_validation, key=lambda l: l[:])
    list_of_best_models = []
    for model_index in range(num_experts):
        list_of_best_models.append(sorted_models[model_index][1] - 1)
    kr_ensemble = np.zeros(num_traces_attacks)
    # kr_ensemble_other = np.zeros(kr_nt)
    # krs_ensemble = np.zeros((20, num_traces_attacks))
    # kr_ensemble_best_models = np.zeros(num_traces_attacks)
    # krs_ensemble_best_models = np.zeros((20, num_traces_attacks))

    for run in range(20):

        key_p_ensemble = np.zeros(256)
        key_p_ensemble_best_models = np.zeros(256)

        for index in range(num_traces_attacks):
            for model_index in range(num_experts):
                key_p_ensemble += k_ps_all[list_of_best_models[model_index]][run][index]

            key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]

            kr_position = list(key_p_ensemble_sorted).index(correct_key)
            kr_ensemble[index] += kr_position
            # kr_ensemble_other[index] += kr_position_other

    ge_ensemble = kr_ensemble / 20

    return ge_ensemble


def ensemble_ge_test(all_predictions, num_experts):
    ge_all_validation = []
    k_ps_all = []
    for i in range(num_experts):
        attack_traces, kp_krs = perform_attacks_e(plt_attack[5000:], all_predictions[i], "attack_traceswithscore",
                                                  leakage_model, dataset, num_traces_attacks)
        print(str(i) + '/' + str(num_experts))
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))
        ge_all_validation.append([np.argmax(attack_traces[:, correct_key] < 1), i])
        k_ps_all.append(kp_krs)

    sorted_models = sorted(ge_all_validation, key=lambda l: l[:])
    list_of_best_models = []
    for model_index in range(num_experts):
        list_of_best_models.append(sorted_models[model_index][1] - 1)
    kr_ensemble = np.zeros(num_traces_attacks)
    # kr_ensemble_other = np.zeros(kr_nt)
    # krs_ensemble = np.zeros((20, num_traces_attacks))
    kr_ensemble_best_models = np.zeros(num_traces_attacks)
    # krs_ensemble_best_models = np.zeros((20, num_traces_attacks))

    for run in range(20):

        key_p_ensemble = np.zeros(256)
        key_p_ensemble_best_models = np.zeros(256)

        for index in range(num_traces_attacks):
            for model_index in range(num_experts):
                key_p_ensemble += k_ps_all[list_of_best_models[model_index]][run][index]
            for model_index in range(number_of_best_experts):
                key_p_ensemble_best_models += k_ps_all[list_of_best_models[model_index]][run][index]

            key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]

            key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

            kr_position = list(key_p_ensemble_sorted).index(correct_key)
            kr_ensemble[index] += kr_position
            # kr_ensemble_other[index] += kr_position_other

            kr_position = list(key_p_ensemble_best_models_sorted).index(correct_key)
            kr_ensemble_best_models[index] += kr_position

        print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, num_experts,
                                                                       int(kr_ensemble[num_traces_attacks - 1] / (
                                                                               run + 1)),
                                                                       number_of_best_experts,
                                                                       int(kr_ensemble_best_models[
                                                                               num_traces_attacks - 1] / (
                                                                                   run + 1))))

    ge_ensemble = kr_ensemble / 20
    ge_ensemble_best_models = kr_ensemble_best_models / 20
    print("ge_ensemble", np.argmax(ge_ensemble < 1))
    print("ge_ensemble_best_models", np.argmax(ge_ensemble_best_models < 1))
    return ge_ensemble, ge_ensemble_best_models


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
            y_pred_valid_metric_all, _ = self.model.predict(X_attack_valid_metric)

            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            # y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)

            # y_pred_valid = tf.argmax(y_pred_valid_metric, 1)
            # comparisons = tf.equal(y_pred_valid, y_pred_valid[0])
            #
            # # 检查所有的比较结果是否都是True
            # all_equal = tf.reduce_all(comparisons)
            #
            # print(all_equal.numpy())
            #
            # unique_values, _, counts = tf.unique_with_counts(y_pred_valid)
            #
            # # 计算每个值的频率
            # total_count = tf.size(y_pred_valid)
            # frequencies = tf.cast(counts, tf.float32) / tf.cast(total_count, tf.float32)
            #
            # # 输出结果
            # for value, everycount, freq in zip(unique_values.numpy(), counts.numpy(), frequencies.numpy()):
            #     print(f"Value: {value}, Count: {everycount}, Frequency: {freq:.2f}")
            #
            # Y_attack_valid = Y_attack_valid[:, :9]
            # Y_attack_valid_int = tf.argmax(Y_attack_valid, 1)
            # correct = tf.equal(y_pred_valid, Y_attack_valid_int)
            # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # print('Validation Accuracy:', accuracy)
            ge_ensemble_valid = ensemble_ge_valid(y_pred_valid_metric_all, num_experts)
            epoch_count = epoch_count + 1

            if ge_ensemble_valid[num_traces_attacks - 1] > 0:
                print("攻击失败", )
                print("GE:", ge_ensemble_valid[num_traces_attacks - 1])
                avg_corr_current = num_traces_attacks - 1

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(ge_ensemble_valid < 1))
                avg_corr_current = np.argmax(ge_ensemble_valid < 1)

            if not corr_logs:
                corr_logs.append(avg_corr_current)
                best_weights = self.model.get_weights()
            else:
                if corr_logs[-1] > avg_corr_current:
                    corr_logs.append(avg_corr_current)
                    best_weights = self.model.get_weights()
                    count = 0
                else:
                    count = count + 1
                    print(count)
                    if count == 6:
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
        avg_logits = tf.reduce_mean(all_logits, axis=0)
        return all_logits, avg_logits

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

    def test_step(self, data):
        x, y = data

        # 测试时使用平均预测
        all_logits = [expert(x, training=False) for expert in self.experts]
        avg_logits = tf.reduce_mean(all_logits, axis=0)

        # 计算损失
        ce_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, avg_logits, from_logits=True)
        )

        # 更新指标
        self.loss_tracker.update_state(ce_loss)
        # self.acc_tracker.update_state(y, avg_logits)

        return {"loss": self.loss_tracker.result()}


num_experts = 6
number_of_best_experts = 6
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
lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                        maximum_momentum=None, minimum_momentum=None, verbose=True)

history = distillation_model.fit(
    x=X_profiling[:50000], y=np.argmax(Y_profiling[:50000, :9], 1),
    batch_size=128,
    epochs=50,
    callbacks=[lr_manager, all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))]
)

print("**************************************************************************************************")
all_predictions, predictions = distillation_model.predict(X_attack[5000:])
ge_ensemble_valid, _ = ensemble_ge_test(all_predictions, num_experts)
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
