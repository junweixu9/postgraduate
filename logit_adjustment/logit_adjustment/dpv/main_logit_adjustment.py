import tensorflow.keras as tk
from tensorflow.keras import backend as K
from random import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.utils import to_categorical
from SCA_util import *
from clr import OneCycleLR
import requests
from keras.optimizers import Adam
from sklearn import preprocessing
from tensorflow.keras.initializers import he_uniform


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
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric, 1)

            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(offset=att_offset[:250], mask=mask,
                                                                                    all_valid_plt_attack=all_valid_plt_attack_metric,
                                                                                    y_pred=y_pred_valid_metric,
                                                                                    output_metric='all',
                                                                                    leakage_model=leakage_model,
                                                                                    correct_key=correct_key,
                                                                                    nb_traces_attacks=num_traces_attacks,
                                                                                    shuffle=True)

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
                    if count == 15:
                        self.model.stop_training = True
                        self.model.set_weights(best_weights)


def shuffle_data(profiling_x, label_y):
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


### CNN network
def cnn_architecture(input_size, learning_rate, classes=9):
    # Designing input layer
    input_shape = (input_size, 1)

    # 1st convolutional block
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    # 1st convolutional block
    x = Conv1D(2, 25, kernel_initializer=he_uniform(), activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer=he_uniform(), activation='selu')(x)
    x = Dense(10, kernel_initializer=he_uniform(), activation='selu')(x)
    x = Dense(4, kernel_initializer=he_uniform(), activation='selu')(x)

    # Logits layer
    x = Dense(classes, name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpv')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=adjustment_loss, optimizer=optimizer, metrics=None)
    model.summary()
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

    callbacks = [lr_manager, all(validation=(Reshaped_X_test, plt_attack[:250], Y_test))]

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
    tro = 0.9
    batch_size = 50
    input_size = 4000
    learning_rate = 1e-3
    num_traces_attacks = 250
    nb_attacks = 20
    real_key = np.load("C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/key.npy")
    mask = np.load("C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/mask.npy")
    att_offset = np.load("C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/attack_offset_dpav4.npy")

    correct_key = real_key[0]

    leakage_model = 'HW'

    # Load the profiling traces
    X_profiling = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/profiling_traces_dpav4.npy")
    Y_profiling = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/profiling_labels_dpav4.npy")
    X_attack = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/attack_traces_dpav4.npy")
    Y_attack = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/attack_labels_dpav4.npy")
    plt_profiling = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/profiling_plaintext_dpav4.npy")
    plt_attack = np.load(
        "C:/Users/Administrator/OneDrive/Desktop/许俊伟/logit_adjustment/dpv/attack_plaintext_dpav4.npy")

    # X_profiling, X_attack = horizontal_standardization(X_profiling, X_attack)

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

    # Standardization + Normalization (between 0 and 1)
    scaler = preprocessing.StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    # Choose your model
    model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
    model_name = "AES_RD"

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")

    # Record the metrics
    history = train_model(X_profiling[:4500], Y_profiling[:4500], X_attack[:250], Y_attack[:250], model,
                          epochs=nb_epochs, batch_size=batch_size,
                          max_lr=learning_rate)

    log = open('F:/result/aes_rd/logit_adjustment.txt', mode='a',
               encoding='utf-8')

    predictions = model.predict(X_attack[250:])
    predictions = tf.nn.softmax(predictions)
    attack_traces = perform_attacks(offset=att_offset[250:], mask=mask,
                                    all_valid_plt_attack=plt_attack[250:],
                                    y_pred=predictions,
                                    output_metric="attack_traces",
                                    leakage_model=leakage_model,
                                    correct_key=correct_key,
                                    nb_traces_attacks=num_traces_attacks,
                                    shuffle=True)

    print("xxxxxx", file=log)
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败", file=log)
        print("GE:", attack_traces[-1, correct_key], file=log)

    else:
        print("攻击成功", file=log)
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1), file=log)

    log.close()

    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')
