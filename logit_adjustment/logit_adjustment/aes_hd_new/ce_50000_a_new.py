import time

from SCA_util_multi import perform_attacks
import random
from clr import OneCycleLR
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import requests
from scipy.special import softmax
from sklearn import preprocessing

def shuffle_data(profiling_x, label_y):
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

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

    # Create model
    output_layer = Dense(8, activation='sigmoid', name='output_8_sigmoid')(x)

    # 构建并返回模型
    model = Model(inputs=img_input, outputs=output_layer, name='InceptionSCAModel')
    return model

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
    leakage_model = "HW"
    correct_key = real_key[0]

    # Load the profiling traces
    X_profiling = np.load(
        r".\profiling_traces_AES_HD.npy")
    Y_profiling = np.load(
        r".\profiling_labels_AES_HD.npy")
    X_attack = np.load(
        r".\attack_traces_AES_HD.npy")
    Y_attack = np.load(
        r".\attack_labels_AES_HD.npy")
    plt_profiling = np.load(
        r".\profiling_ciphertext_AES_HD.npy")
    plt_attack = np.load(
        r".\attack_ciphertext_AES_HD.npy")

    def convert_to_multilabel(byte_values):
        """Converts a 1D array of byte values (0-255) to a 2D numpy array of shape (N, 8)."""
        # Ensure input is a numpy array of type uint8 for unpackbits
        byte_values = np.array(byte_values, dtype=np.uint8)
        # np.unpackbits converts each byte into an array of 8 bits (MSB first)
        return np.unpackbits(byte_values[:, np.newaxis], axis=1)

    Y_profiling = np.squeeze(convert_to_multilabel(Y_profiling),axis=2)
    Y_attack = np.squeeze(convert_to_multilabel(Y_attack),axis=2)

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

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
    )
    model.summary()

    model_name = "AES_HD"

    print('\n Model name = ' + model_name)

    print("\n############### Starting Training #################\n")
    start_time = time.perf_counter()
    # Record the metrics
    history = train_model(X_profiling[:45000], Y_profiling[:45000], X_profiling[45000:], Y_profiling[45000:], model,
                          epochs=nb_epochs, batch_size=batch_size,
                          max_lr=learning_rate)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time} 秒")

    # log = open('C:/Users/hp/Desktop/AES_hd.txt', mode='a',
    #            encoding='utf-8')
    start_time = time.perf_counter()
    predictions = model.predict(X_attack)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"推理时间: {execution_time} 秒")

    start_time = time.perf_counter()
    attack_traces = perform_attacks(plt_attack, predictions, "attack_traces",
                                    leakage_model, correct_key, num_traces_attacks)
    end_time = time.perf_counter()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"密钥恢复: {execution_time} 秒")

    # print("xxxxxx", file=log)
    if attack_traces[-1, correct_key] > 0:
        print("攻击失败")
        print("GE:", attack_traces[-1, correct_key])

    else:
        print("攻击成功")
        print("TGE0:", np.argmax(attack_traces[:, correct_key] < 1))

    # log.close()

    ret = requests.get('https://api.day.app/Cqdu3932Dcbe3dduH2EJPM/训练/完成')