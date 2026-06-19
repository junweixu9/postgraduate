import time

import tensorflow as tf
from tensorflow.keras import layers, Model
from data_load_ascad_50 import read_data
from SCA_util_standard import perform_attacks
import tensorflow.keras as tk
from clr import OneCycleLR
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from scipy.special import softmax
from tensorflow.keras import backend as K

class Custom_Model(tf.keras.Model):
    def __init__(self, loss_parameters, *args, **kwargs):
        super(Custom_Model, self).__init__(*args, **kwargs)
        self.loss_type = loss_parameters[0]
        if self.loss_type == "soft_nn" or self.loss_type == "flr_soft_nn":
            self.temperature = loss_parameters[1]
            self.loss_obj = Soft_nearest_neighbour(temperature=self.temperature, distance_fn="euclidean",
                                                   name="soft_nearest_neighbour")
        elif self.loss_type == "center_loss" or self.loss_type == "flr_center_loss":
            self.alpha = loss_parameters[1]
            self.n_classes = loss_parameters[2]
            self.n_features = loss_parameters[3]
            self.loss_obj = Center_Loss(self.n_classes, self.n_features, alpha=self.alpha, update_center=True,
                                        name="center_loss")
        elif self.loss_type == "categorical_crossentropy" or self.loss_type == "flr":
            self.loss_obj = None
        print(self.loss_obj)


class Center_Loss(tf.keras.losses.Loss):
    def __init__(self, n_classes, n_features, alpha=0.1, update_center=True, name="center_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.update_centers = update_center
        self.centers = tf.Variable(tf.zeros([n_classes, n_features]), name="centers",
                                   trainable=False, )

        # in a distributed strategy, we want updates to this variable to be summed.
        #aggregation=tf.VariableAggregation.SUM)
        print("Alpha:", self.alpha)
        print("n_features:", self.n_features)

    def call(self, x_batch, labels_batch):
        # labels_batch = tf.reshape(labels_batch, (-1,))
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        x_batch = tf.reshape(x_batch, (tf.shape(x_batch)[0], -1))
        centers_batch = tf.gather(self.centers, labels_batch)
        # print("centers_batch", centers_batch)
        # the reduction of batch dimension will be done by the parent class
        center_loss = tf.keras.losses.mean_squared_error(x_batch, centers_batch)

        ### THIS IS TO CHECK ####
        # print("center loss", center_loss)
        # batch_size = tf.shape(x_batch)[0]
        # batch_size_total = tf.cast(batch_size, dtype=x_batch.dtype)
        # print("batch_size_total: ", batch_size_total)
        # center_loss_2 = tf.reduce_sum(center_loss)
        # center_loss_2 = tf.math.divide(center_loss_2, batch_size_total)
        # print("center_loss_2: ", center_loss_2)
        # print("self.centers : ", self.centers)
        return center_loss

    def update_center(self, x_batch, labels_batch):  #,  update_center = True):
        # self.update_centers = update_center
        # print("Updating")
        # print("self.centers before ", self.centers)
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        centers_batch = tf.gather(self.centers, labels_batch)
        diff = (centers_batch - x_batch)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels_batch)
        # print("unique_label", unique_label)
        # print("unique_idx", unique_idx)
        # print("unique_count", unique_count)
        appear_times = tf.gather(unique_count, unique_idx)
        # print("appear_times 1", appear_times)
        appear_times = tf.reshape(appear_times, [-1, 1])
        # print("appear_times 2", appear_times)
        batch_size = tf.shape(x_batch)[0]
        # find all same class neighbour
        pos_mask, _ = self.build_masks(labels_batch, labels_batch, batch_size, remove_diagonal=False)
        # print("diff: ", diff)
        # print("pos_mask: ", pos_mask)
        pos_mask = tf.cast(pos_mask, dtype=x_batch.dtype)
        sacn_diff = tf.tensordot(pos_mask, diff, axes=1)
        # print("sacn_diff", sacn_diff)

        sacn_diff = tf.math.divide(sacn_diff, tf.cast((1 + appear_times), tf.float32))
        sacn_diff = self.alpha * sacn_diff
        updates = tf.scatter_nd(indices=labels_batch[:, None], updates=sacn_diff,
                                shape=self.centers.shape)  # This will cause: update = [1,2,3] -> [0,1,0,0,2,3], with indices = [1,4,5] and shape = (6,)
        # print("updates", updates)
        # using assign_sub will make sure updates are added during distributed
        # training
        self.centers.assign_sub(updates)  # this will do the following: self.center = self.center - updates
        # print("self.centers after ", self.centers)

    def build_masks(self, query_labels, key_labels, batch_size, remove_diagonal=True):
        """Build masks that allows to select only the positive or negatives
        embeddings.
        Args:
            query_labels: 1D int `Tensor` that contains the query class ids.
            key_labels: 1D int `Tensor` that contains the key class ids.
            batch_size: size of the batch.
            remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask
        Returns:
            Tuple of Tensors containing the positive_mask and negative_mask
        """

        query_labels = tf.reshape(query_labels, (-1, 1))

        key_labels = tf.reshape(key_labels, (-1, 1))
        # print("query_labels inside build_mask 2", query_labels)
        # print("key_labels inside build_mask 2", key_labels)
        # same class mask
        positive_mask = tf.math.equal(query_labels, tf.transpose(key_labels))

        # not the same class
        negative_mask = tf.math.logical_not(positive_mask)

        if remove_diagonal:
            # It is optional to remove diagonal from positive mask.
            # Diagonal is often removed if queries and keys are identical.
            positive_mask = tf.linalg.set_diag(positive_mask, tf.zeros(batch_size, dtype=tf.bool))
        return positive_mask, negative_mask


class Soft_nearest_neighbour(tf.keras.losses.Loss):
    def __init__(self, temperature=100, distance_fn="euclidean", name="soft_nearest_neighbour", **kwargs):
        super().__init__(name=name, **kwargs)
        self.t_placeholder = tf.Variable(1., dtype=tf.float32, trainable=False, name="temp")
        # self.initial_temperature = 1
        self.temperature = temperature  #Controls relative importance given to the pair of points.
        self.distance_fn = distance_fn  #distance function to compute the pairwise
        print("Temperature:", self.temperature)

    def call(self, x_batch, labels_batch):
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        x_batch = tf.reshape(x_batch, (tf.shape(x_batch)[0], -1))
        # print(x_batch)
        if self.distance_fn == 'euclidean':  #This is square euclidean
            x_squared_norm = tf.math.square(x_batch)
            x_squared_norm = tf.math.reduce_sum(x_squared_norm, axis=1, keepdims=True)
            distances = 2.0 * tf.linalg.matmul(x_batch, x_batch, transpose_b=True)
            distances = x_squared_norm - distances + tf.transpose(
                x_squared_norm)  #this is the expanded form of sum (p-q)^2
            # Avoid NaN and inf gradients when back propagating through the sqrt.
            # values smaller than 1e-18 produce inf for the gradient, and 0.0
            # produces NaN. All values smaller than 1e-13 should produce a gradient
            # of 1.0.
            distances = tf.math.maximum(distances, 1e-18)
        # print(distances)
        batch_size = tf.shape(x_batch)[0]
        eps = tf.cast(1e-9, dtype=x_batch.dtype)
        distances = distances / self.temperature
        # print("distances: ",distances)
        negexpd = tf.math.exp(-distances)
        negexpd = tf.math.maximum(negexpd,
                                  1e-18)  #This line makes the soft nn loss stable. If not will cause NaN as stated above.
        # print("negexpd: ",negexpd)
        # Mask out diagonal entries
        diag = tf.linalg.diag(tf.ones(batch_size, dtype=tf.bool))
        diag_mask = tf.cast(tf.logical_not(diag), dtype=x_batch.dtype)
        # print(diag_mask)
        negexpd = tf.math.multiply(negexpd, diag_mask)
        # creating mask to sample same class neighboorhood (note: remove the diagonal.)
        pos_mask, _ = self.build_masks(
            labels_batch,
            labels_batch,
            batch_size=batch_size,
            remove_diagonal=True,
        )
        # print(pos_mask)
        pos_mask = tf.cast(pos_mask, dtype=x_batch.dtype)
        # all class neighborhood
        alcn = tf.reduce_sum(negexpd, axis=1)

        # print("alcn: ", alcn)
        # same class neighborhood
        sacn = tf.reduce_sum(tf.math.multiply(negexpd, pos_mask), axis=1)
        # print("sacn: ", sacn)
        softnn_loss = tf.math.divide(sacn, alcn)
        # print("sacn/alcn: ", softnn_loss)

        softnn_loss = tf.math.log(eps + softnn_loss)
        # print("log(sacn/alcn): ", softnn_loss)
        softnn_loss = tf.math.multiply(softnn_loss, -1)
        ### THIS IS TO CHECK ####
        # softnn_loss_2 = tf.reduce_sum(softnn_loss)
        # # print("sum log(sacn/alcn): ", softnn_loss_2)
        # batch_size_total = tf.cast(batch_size, dtype=x_batch.dtype)
        # # print("batch_size_total: ", batch_size_total)
        # softnn_loss_2 = tf.math.divide(softnn_loss_2, batch_size_total)
        # print("1/b sum log(sacn/alcn) final: ", softnn_loss_2)

        return softnn_loss

    def build_masks(self, query_labels, key_labels, batch_size, remove_diagonal=True):
        """Build masks that allows to select only the positive or negatives
        embeddings.
        Args:
            query_labels: 1D int `Tensor` that contains the query class ids.
            key_labels: 1D int `Tensor` that contains the key class ids.
            batch_size: size of the batch.
            remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask
        Returns:
            Tuple of Tensors containing the positive_mask and negative_mask
        """
        if tf.rank(query_labels) == 1:
            # if len(query_labels.shape) == 1:
            query_labels = tf.reshape(query_labels, (-1, 1))

        if tf.rank(key_labels) == 1:
            # if len(query_labels.shape) == 1:
            key_labels = tf.reshape(key_labels, (-1, 1))
        # print("query_labels inside build_mask 2", query_labels)
        # print("key_labels inside build_mask 2", key_labels)
        # same class mask
        positive_mask = tf.math.equal(query_labels, tf.transpose(key_labels))

        # not the same class
        negative_mask = tf.math.logical_not(positive_mask)

        if remove_diagonal:
            # It is optional to remove diagonal from positive mask.
            # Diagonal is often removed if queries and keys are identical.
            positive_mask = tf.linalg.set_diag(positive_mask, tf.zeros(batch_size, dtype=tf.bool))
        return positive_mask, negative_mask


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
            X_attack_valid_metric, all_valid_plt_attack_metric,Y_attack_valid = self.validation[0], self.validation[1], self.validation[2]
            y_pred_valid_metric_all = self.model.predict(X_attack_valid_metric)

            # y_pred_valid = tf.argmax(y_pred_valid_metric_all, 1)
            # Y_attack_valid = Y_attack_valid[:, :9]
            # Y_attack_valid_int = tf.argmax(Y_attack_valid, 1)
            # np.set_printoptions(threshold=np.inf)

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
                    if count == 10:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)

            # print(y_pred_valid.numpy())
            # print(y_pred_valid_metric_all.numpy())
            # print(Y_attack_valid_int.numpy())
            #
            # correct = tf.equal(y_pred_valid, Y_attack_valid_int)
            # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # print('Validation Accuracy:', accuracy)
            # print()


def ascad_f_hw_cnn_rs_50(extra_loss_param, length, metric, loss, learning_rate):
    input_shape = (length, 1)
    intermediate_output = []
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(
        img_input)
    intermediate_output.append(x)
    x = BatchNormalization()(x)
    intermediate_output.append(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    intermediate_output.append(x)

    # 2nd convolutional block
    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    intermediate_output.append(x)
    x = BatchNormalization()(x)
    intermediate_output.append(x)
    x = AveragePooling1D(25, strides=25, name='block2_pool')(x)
    intermediate_output.append(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    intermediate_output.append(x)
    x = BatchNormalization()(x)
    intermediate_output.append(x)
    x = AveragePooling1D(4, strides=4, name='block3_pool')(x)
    intermediate_output.append(x)
    x = Flatten(name='flatten')(x)
    intermediate_output.append(x)
    # Classification part
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    intermediate_output.append(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    intermediate_output.append(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    intermediate_output.append(x)
    x = Dense(9)(x)
    intermediate_output.append(x)
    if extra_loss_param[0] == "center_loss":
        extra_loss_param.append(9)
        extra_loss_param.append(intermediate_output[-2].shape[1])
    # 只取 intermediate_output 的最后一个元素（即 Dense(9) 的输出）作为模型输出
    model = Custom_Model(extra_loss_param, img_input, intermediate_output[-1])
    model.compile(optimizer='adam', metrics=metric,loss= loss)
    model.summary()

    return model

def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    # print("y_pred.shape: ", y_pred.shape)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    alpha = np.array(0.25, dtype=np.float32)
    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)
    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, 2) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))


def flr_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :9]

    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 10
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = categorical_focal_loss_fixed(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / (average_fake_k_loss + 1e-40)
    return loss_cer


class OnlineDistillationModel(Model):
    def __init__(self):
        super(OnlineDistillationModel, self).__init__()

        img_input = Input(shape=(700, 1))
        intermediate_output = []
        x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
        intermediate_output.append(x)
        x = AveragePooling1D(4, strides=4)(x)
        intermediate_output.append(x)
        x = Flatten(name='flatten')(x)
        intermediate_output.append(x)
        x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
        intermediate_output.append(x)
        x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
        intermediate_output.append(x)
        x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
        intermediate_output.append(x)
        x = Dense(9)(x)
        intermediate_output.append(x)
        if extra_loss_param[0] == "center_loss":
            extra_loss_param.append(9)
            extra_loss_param.append(intermediate_output[-2].shape[1])
        self.model = Custom_Model(extra_loss_param, img_input, intermediate_output)
        self.model.compile(optimizer='adam', metrics=None)
        self.model.summary()

        # 指标跟踪
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pred_loss_tracker = tf.keras.metrics.Mean(name="ce_loss")
        self.other_loss_loss_tracker = tf.keras.metrics.Mean(name="other_loss")
        # self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.pred_loss_tracker,
            self.other_loss_loss_tracker,
        ]

    def call(self, inputs):
        # 在推理时，可以使用所有专家的平均预测

        predictions_with_intermediate_layer = self.model(inputs, training=True)
        predictions_without_softmax = predictions_with_intermediate_layer[-1]
        logits = tf.nn.softmax(predictions_without_softmax, 1)
        print("call")

        return logits

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            # 1. 前向传播：获取所有专家的预测
            predictions_with_intermediate_layer = self.model(x, training=True)
            predictions_without_softmax = predictions_with_intermediate_layer[-1]
            intermediate_layers_output = predictions_with_intermediate_layer[:-1]

            other_loss = 0

            if self.model.loss_type == "soft_nn":
                # print("soft_nn being activated")
                for one_layer_output in intermediate_layers_output:
                    soft_nn_loss = self.model.loss_obj(one_layer_output, y)
                    # print("train soft_nn_loss outside", soft_nn_loss)
                    other_loss += soft_nn_loss
            elif self.model.loss_type == "center_loss":
                # print("center_loss being activated")
                center_loss = self.model.loss_obj(intermediate_layers_output[-1], y)
                # print("center_loss outside", center_loss)
                other_loss += center_loss

            if ajust_flag:
                predictions_without_softmax = predictions_without_softmax

            predictions_softmax = tf.nn.softmax(predictions_without_softmax, 1)

            pred_loss = tk.backend.categorical_crossentropy(y, predictions_softmax)
            # y = K.clip(y, K.epsilon(), 1)
            # predictions_softmax = K.clip(predictions_softmax, K.epsilon(), 1)
            # pred_loss = K.sum(y * K.log(y / predictions_softmax), axis=-1)

            # print("cer",pred_loss)
            total_loss = pred_loss
            total_loss += lamb * other_loss
            # if len(self.model.losses) > 0:
            #     regularization_loss = tf.math.add_n(self.model.losses)
            #     total_loss = total_loss + regularization_loss

        self.gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.model.trainable_variables))

        # 6. 更新指标
        self.loss_tracker.update_state(total_loss)
        self.pred_loss_tracker.update_state(pred_loss)
        self.other_loss_loss_tracker.update_state(other_loss)
        if self.model.loss_type == "center_loss":
            self.model.loss_obj.update_center(intermediate_layers_output[-1], y)
        return {m.name: m.result() for m in self.metrics}


loss_type = "center_loss"  # “center_loss”  "soft_nn"
extra_loss_param = [loss_type, 0.005]

lamb = 0.6
ajust_flag = True
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
num_traces_attacks = 5000
epoch_count = 0
data_arguementation = False  # enable/disbale data arguementation
data_arguementation_level = 0.25  # data arguementation level
best_weights = None
count = 0
(X_profiling, X_attack), (Y_profiling, Y_attack), (
    plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
    leakage_model,
    data_arguementation,
    data_arguementation_level,
    attack_model, dataset,
    sigma_hw, sigma_id)
distillation_model = ascad_f_hw_cnn_rs_50(extra_loss_param,X_profiling.shape[1],None,flr_adjustment_loss,1e-5)
Y_profiling_int = np.argmax(Y_profiling[:, :9], 1)
label_freq = {}
for key in Y_profiling_int:
    label_freq[key] = label_freq.get(key, 0) + 1
cls_num_list = dict(sorted(label_freq.items()))
cls_num_list = np.array(list(cls_num_list.values()))
prior = cls_num_list / cls_num_list.sum()
# adjustments = np.log(prior + 1e-12)
# adjustments_tf = adjustments * tro

# Y_profiling_part = np.sum(Y_profiling[:, :9], 0)
# Y_profiling_all = np.sum(Y_profiling_part)
# label_freq_array = Y_profiling_part / Y_profiling_all
# adjustments = np.log(label_freq_array ** tro + 1e-12)


distillation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3)
)
lr_manager = OneCycleLR(len(X_attack[:5000]), 128, 5e-3, end_percentage=0.2, scale_percentage=0.1,
                        maximum_momentum=None, minimum_momentum=None, verbose=True)
start_time = time.perf_counter()
history = distillation_model.fit(
    x=X_profiling[:45000], y=Y_profiling[:45000, :9],
    batch_size=128,
    epochs=50,
)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"训练时间: {execution_time} 秒")
print("**************************************************************************************************")


start_time = time.perf_counter()
all_predictions = distillation_model.predict(X_attack[:10000])
predictions = softmax(all_predictions)

end_time = time.perf_counter()

# 计算并打印执行时间
execution_time = end_time - start_time
print(f"推理时间: {execution_time} 秒")

start_time = time.perf_counter()
attack_traces = perform_attacks(plt_attack[:10000], predictions, "attack_traces",
                                leakage_model, dataset, num_traces_attacks)

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
