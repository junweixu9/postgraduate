import tensorflow.keras as tk
from random import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.utils import to_categorical
from SCA_util_knl import *
from clr import OneCycleLR
import requests
from scipy.special import softmax
import os
from sklearn import preprocessing



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
            X_attack_valid_metric, all_valid_plt_attack_metric, Y_attack_valid = self.validation[0], \
                self.validation[1], \
                self.validation[2]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            y_pred_valid_metric = softmax(y_pred_valid_metric, 1)

            avg_rank_current, avg_attack_traces = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, correct_key,
                                                                                    num_traces_attacks)

            # all_corr_logs.append(avg_corr_current)
            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                # print("corr:", avg_corr_current)

            else:
                print("攻击成功")
                print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                # print("corr:", avg_corr_current)


def shuffle_data(profiling_x, label_y):
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


### CNN network
def cnn_architecture(input_size=1250, learning_rate=0.00001, classes=9):
    # Designing input layer
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_hd_model')

    model.compile(loss=adjustment_loss, optimizer="adam", metrics=None)
    return model


def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]


def compute_adjustment(Y_profiling, tro):
    """compute the base probabilities"""

    Y_profiling = np.argmax(Y_profiling[:, :9], 1)
    label_freq = {}
    for key in Y_profiling:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)

    return adjustments


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
    y_true = y_true[:, :classes]
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


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, epochs=150, batch_size=100,
                max_lr=1e-3):
    # Get the input layer shape

    Reshaped_X_profiling, Reshaped_X_test = X_profiling.reshape(
        (X_profiling.shape[0], X_profiling.shape[1], 1)), X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lr_manager = OneCycleLR(len(X_test), batch_size, learning_rate, end_percentage=0.2, scale_percentage=0.1,
                            maximum_momentum=None, minimum_momentum=None, verbose=True)

    callbacks = [lr_manager]

    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks)
    return history


def cer_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]
    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 10
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = tf.keras.losses.categorical_crossentropy(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / (average_fake_k_loss + 1e-40)
    return loss_cer


def horizontal_standardization(X_profiling, X_attack):
    mn = np.repeat(np.mean(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    std = np.repeat(np.std(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    X_profiling_processed = (X_profiling - mn) / std

    mn = np.repeat(np.mean(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    std = np.repeat(np.std(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    X_attack_processed = (X_attack - mn) / std

    return X_profiling_processed, X_attack_processed


if __name__ == '__main__':

    # Choose the name of the model
    rank_logs = []
    all_rank_logs = []
    corr_logs = []
    all_corr_logs = []
    loss_logs = []
    kl_loss_logs = []
    count = 1
    best_weights = None
    nb_epochs = 50
    classes = 9
    adjust_flag = True
    tro = 0
    batch_size = 256
    input_size = 1250
    learning_rate = 5e-3
    num_traces_attacks = 1500
    nb_attacks = 20
    real_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

    correct_key = real_key[0]
    leakage_model = 'HW'

    # Load the profiling traces
    X_profiling = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\profiling_traces_AES_HD.npy")
    Y_profiling = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\profiling_labels_AES_HD.npy")
    X_attack = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\attack_traces_AES_HD.npy")
    Y_attack = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\attack_labels_AES_HD.npy")
    plt_profiling = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\profiling_ciphertext_AES_HD.npy")
    plt_attack = np.load(
        r"C:\Users\hp\Desktop\logit_adjustment\aes_hd\attack_ciphertext_AES_HD.npy")

    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)  # Y_profiling是十进制，需要转换为二进制，并计算汉明重量
        Y_attack = calculate_HW(Y_attack)  # Y_attack是十进制，需要转换为二进制，并计算汉明重量

    Y_profiling = to_categorical(Y_profiling, num_classes=classes)
    Y_attack = to_categorical(Y_attack, num_classes=classes)

    adjustments = compute_adjustment(Y_profiling, tro)
    adjustments_valid = tf.cast(adjustments, dtype=tf.double)

    # Shuffle data
    (X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    X_profiling = X_profiling.astype('float32')
    X_attack = X_attack.astype('float32')

    # Standardization and Normalization (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    # Choose your model
    model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
    model_name = "AES_HD"

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")

    # Record the metrics
    history = train_model(X_profiling[:45000], Y_profiling[:45000], X_profiling[45000:], Y_profiling[45000:], model,
                          epochs=nb_epochs, batch_size=batch_size,
                          max_lr=learning_rate)

    # log = open('C:/Users/hp/Desktop/AES_hd.txt', mode='a',
    #            encoding='utf-8')

    predictions = model.predict(X_attack)
    predictions = softmax(predictions,1)
    attack_traces = perform_attacks(plt_attack, predictions, "attack_traces",
                                    leakage_model, correct_key, num_traces_attacks)

    # print("xxxxxx", file=log)
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))

    # log.close()

    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')
