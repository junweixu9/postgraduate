import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop


def pick_SOAT(data, leakage_model, length, metric, loss, learning_rate, model='MLP', model_size=64):
    if model == 'CNN':
        if data == 'ASCAD' and leakage_model == 'HW':
            batch_size = 128
            epoch = 50
            return ascad_f_hw_cnn_rs(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'ASCAD' and leakage_model == 'ID':
            batch_size = 50
            epoch = 50
            return ascad_f_id_cnn(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'ASCAD_rand' and leakage_model == 'HW':
            batch_size = 128
            epoch = 50
            return ascad_r_hw_cnn(length, metric, loss, learning_rate), batch_size, epoch
            # return cnn_best(length, metric, loss, classes=9, unit=64), batch_size, epoch
        elif data == 'ASCAD_rand' and leakage_model == 'ID':
            batch_size = 128
            epoch = 50
            return ascad_r_id_cnn(length, metric, loss), batch_size, epoch
        elif data == 'CHES_CTF' and leakage_model == 'HW':
            batch_size = 128
            epoch = 50
            return ches_ctf_hw_cnn_1(length, metric, loss), batch_size, epoch
    elif model == 'MLP':
        if data == 'ASCAD' and leakage_model == 'HW':
            batch_size = 32
            epoch = 10
            return ascad_f_hw_mlp(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'ASCAD' and leakage_model == 'ID':
            batch_size = 32
            epoch = 10
            return ascad_f_id_mlp(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'ASCAD_rand' and leakage_model == 'HW':
            batch_size = 32
            epoch = 10
            return ascad_r_hw_mlp(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'ASCAD_rand' and leakage_model == 'ID':
            batch_size = 32
            epoch = 10
            return ascad_r_id_mlp(length, metric, loss, learning_rate), batch_size, epoch
        elif data == 'CHES_CTF' and leakage_model == 'HW':
            batch_size = 32
            epoch = 10
            return ches_ctf_hw_mlp(length, metric, loss), batch_size, epoch
        elif data == '' and leakage_model == 'HW':
            batch_size = 32
            epoch = 10
            return AESHD_hw_cnn_rs(length, metric, loss, learning_rate), batch_size, epoch
    elif model == 'model_size':
        batch_size = 200
        epoch = 75
        if leakage_model == 'HW':
            return cnn_best(length, metric, loss, classes=9, unit=model_size), batch_size, epoch
        else:
            return cnn_best(length, metric, loss, classes=256, unit=model_size), batch_size, epoch

        ################ SOAT MODELS#####################

def ascad_f_hw_cnn1(length, metric, loss, learning_rate):
    input_shape = (length, 1)
    img_input = Input(shape=input_shape)

    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(
        img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    model.compile(loss=loss, optimizer="adam", metrics=metric)
    model.summary()
    return model

# epoch 50
def ascad_f_hw_cnn(length, metric, loss, learning_rate):
    img_input = Input(shape=(length, 1))
    x = Conv1D(16, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_f_hw_cnn_rs(extra_loss_param,length, metric, loss, learning_rate):
    img_input = Input(shape=(length, 1))
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
    model = Custom_Model(extra_loss_param, img_input, intermediate_output)
    model.compile(optimizer='adam', metrics=metric)
    model.summary()
    return model

class Custom_Model(tf.keras.Model):
    def __init__(self, loss_parameters, *args, **kwargs):
        super(Custom_Model, self).__init__(*args, **kwargs)
        self.loss_type = loss_parameters[0]
        if self.loss_type == "soft_nn" or self.loss_type == "flr_soft_nn":
            self.temperature = loss_parameters[1]
            self.loss_obj = Soft_nearest_neighbour(temperature = self.temperature, distance_fn = "euclidean", name = "soft_nearest_neighbour")
        elif self.loss_type == "center_loss" or self.loss_type == "flr_center_loss":
            self.alpha = loss_parameters[1]
            self.n_classes = loss_parameters[2]
            self.n_features = loss_parameters[3]
            self.loss_obj = Center_Loss(self.n_classes, self.n_features, alpha = self.alpha, update_center = True, name = "center_loss")
        elif self.loss_type == "categorical_crossentropy" or self.loss_type == "flr":
            self.loss_obj = None
        print(self.loss_obj)

class Center_Loss(tf.keras.losses.Loss):
    def __init__(self, n_classes, n_features, alpha = 0.1, update_center = True, name = "center_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.update_centers = update_center
        self.centers = tf.Variable(tf.zeros([n_classes, n_features]),name="centers",
            trainable=False,)

            # in a distributed strategy, we want updates to this variable to be summed.
            #aggregation=tf.VariableAggregation.SUM)
        print("Alpha:",self.alpha)
        print("n_features:",self.n_features)

    def call(self, x_batch,labels_batch):
        # labels_batch = tf.reshape(labels_batch, (-1,))
        labels_batch = tf.math.argmax(labels_batch, axis = 1)
        x_batch = tf.reshape(x_batch,(tf.shape(x_batch)[0], -1))
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

    def update_center(self,x_batch,labels_batch):#,  update_center = True):
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

class Soft_nearest_neighbour(tf.keras.losses.Loss):
    def __init__(self, temperature = 2, distance_fn = "euclidean", name = "soft_nearest_neighbour", **kwargs):
        super().__init__(name=name, **kwargs)
        self.t_placeholder = tf.Variable(1., dtype=tf.float32, trainable=False, name="temp")
        # self.initial_temperature = 1
        self.temperature = temperature #Controls relative importance given to the pair of points.
        self.distance_fn = distance_fn #distance function to compute the pairwise
        print("Temperature:", self.temperature)
    def call(self, x_batch,labels_batch):
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        x_batch = tf.reshape(x_batch,(tf.shape(x_batch)[0], -1))
        # print(x_batch)
        if self.distance_fn == 'euclidean': #This is square euclidean
            x_squared_norm = tf.math.square(x_batch)
            x_squared_norm = tf.math.reduce_sum(x_squared_norm, axis = 1, keepdims = True)
            distances = 2.0 * tf.linalg.matmul(x_batch, x_batch, transpose_b=True)
            distances = x_squared_norm - distances + tf.transpose(x_squared_norm) #this is the expanded form of sum (p-q)^2
            # Avoid NaN and inf gradients when back propagating through the sqrt.
            # values smaller than 1e-18 produce inf for the gradient, and 0.0
            # produces NaN. All values smaller than 1e-13 should produce a gradient
            # of 1.0.
            distances = tf.math.maximum(distances, 1e-18)
        # print(distances)
        batch_size = tf.shape(x_batch)[0]
        eps = tf.cast(1e-9, dtype=x_batch.dtype)
        distances = distances/self.temperature
        # print("distances: ",distances)
        negexpd = tf.math.exp(-distances)
        negexpd = tf.math.maximum(negexpd, 1e-18) #This line makes the soft nn loss stable. If not will cause NaN as stated above.
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


def AESHD_hw_cnn_rs(length, metric, loss, learning_rate):
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape, filters=64, kernel_size=15, padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=15, strides=15))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=2, strides=2))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(embedding_size))
    return model

def ascad_f_hw_mlp(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(496, activation='relu')(img_input)
    x = Dense(496, activation='relu')(x)
    x = Dense(136, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    x = Dense(552, activation='relu')(x)
    x = Dense(408, activation='relu')(x)
    x = Dense(232, activation='relu')(x)
    x = Dense(856, activation='relu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)  # 0.0005
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_f_hw_mlp_my(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(512, activation='relu', kernel_initializer='he_uniform')(img_input)
    x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=learning_rate)  #0.0015
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_f_hw_mlp_1(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(1024, activation='relu')(img_input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(760, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(704, activation='relu')(x)
    x = Dense(1016, activation='relu')(x)
    x = Dense(560, activation='relu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)  #1e-5
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_f_id_cnn(length, metric, loss, learning_rate):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_f_id_cnn_rs(length, metric, loss, learning_rate):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 75, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Conv1D(2, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4)(x)
    x = Conv1D(8, 2, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_f_id_mlp(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(160, activation='relu')(img_input)
    x = Dense(160, activation='relu')(x)
    x = Dense(624, activation='relu')(x)
    x = Dense(776, activation='relu')(x)
    x = Dense(328, activation='relu')(x)
    x = Dense(968, activation='relu')(x)
    x = Dense(256)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)  # 0.0001
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_f_id_mlp_1(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(480, activation='elu')(img_input)
    x = Dense(480, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)  # 1e-5
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


# epoch:50/batch_size:400
def ascad_r_hw_cnn(length, metric, loss,l):
    img_input = Input(shape=(length, 1))
    x = Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_r_hw_cnn_rs(length, metric, loss,learning):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 50, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=7e)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_r_hw_mlp(length, metric, loss, learning_rate):
    img_input = Input(shape=(length,))
    x = Dense(200, activation='elu')(img_input)
    x = Dense(200, activation='elu')(x)
    x = Dense(304, activation='elu')(x)
    x = Dense(832, activation='elu')(x)
    x = Dense(176, activation='elu')(x)
    x = Dense(872, activation='elu')(x)
    x = Dense(608, activation='elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)  # 0.0005
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_r_hw_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(448, activation='elu')(img_input)
    x = Dense(448, activation='elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dense(168, activation='elu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


# epoch:50/batch_size:400
def ascad_r_id_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(75, strides=75)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_r_id_cnn_rs(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(100, strides=75)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_r_id_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(256, activation='elu')(img_input)
    x = Dense(256, activation='elu')(x)
    x = Dense(296, activation='elu')(x)
    x = Dense(840, activation='elu')(x)
    x = Dense(280, activation='elu')(x)
    x = Dense(568, activation='elu')(x)
    x = Dense(672, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_r_id_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(664, activation='elu')(img_input)
    x = Dense(664, activation='elu')(x)
    x = Dense(624, activation='elu')(x)
    x = Dense(816, activation='elu')(x)
    x = Dense(624, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ches_ctf_hw_cnn_1(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ches_ctf_hw_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 2, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(7, strides=7)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ches_ctf_hw_mlp_1(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(696, activation='elu')(x)
    x = Dense(696, activation='elu')(x)
    x = Dense(168, activation='elu')(x)
    x = Dense(184, activation='elu')(x)
    x = Dense(848, activation='elu')(x)
    x = Dense(568, activation='elu')(x)
    x = Dense(328, activation='elu')(x)
    x = Dense(584, activation='elu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ches_ctf_hw_mlp(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(192, activation='elu')(x)
    x = Dense(192, activation='elu')(x)
    x = Dense(616, activation='elu')(x)
    x = Dense(248, activation='elu')(x)
    x = Dense(440, activation='elu')(x)
    x = Dense(9)(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def cnn_best(length, metric, loss, classes=9, unit=64):
    # From VGG16 design
    input_shape = (length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(unit * 1, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(unit * 2, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(unit * 4, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(unit * 8, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(unit * 8, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(unit * 64, activation='relu', name='fc1')(x)
    x = Dense(unit * 64, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

# def mlp_best(length, metric, loss, lr=0.00001, node=200, layer_nb=6, initializer='glorot_uniform'):
#     model = Sequential()
#     model.add(Dense(node, input_dim=length, activation='relu', kernel_initializer=initializer))

#     for i in range(layer_nb - 2):
#         model.add(Dense(node, activation='relu', kernel_initializer=initializer))

#     model.add(Dense(classes, activation='softmax'))
#     optimizer = RMSprop(lr=lr)
#     model.compile(loss=loss, optimizer=optimizer, metrics=metric)
#     model.summary()
#     return model
