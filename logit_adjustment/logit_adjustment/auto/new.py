import keras_tuner as kt
from kerastuner.tuners import *
from data_load_ascad_50000 import read_data
from SCA_util import perform_attacks
import numpy as np
import tensorflow.keras as tk
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime
from Util.one_cycle_lr import OneCycleLR


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
            logs['tge0'] = float('inf')
            X_attack_valid_metric, all_valid_plt_attack_metric = self.validation[0], self.validation[1]
            y_pred_valid_metric = self.model.predict(X_attack_valid_metric)
            # y_pred_valid_metric = y_pred_valid_metric -0.25 * adjustments
            y_pred_valid_metric = tf.nn.softmax(y_pred_valid_metric)
            avg_rank_current, avg_attack_traces, avg_corr_current = perform_attacks(all_valid_plt_attack_metric,
                                                                                    y_pred_valid_metric,
                                                                                    'all',
                                                                                    leakage_model, dataset,
                                                                                    num_traces_attacks)
            all_corr_logs.append(avg_corr_current)
            # if not corr_logs:
            #     corr_logs.append(avg_corr_current)
            #     best_weights = self.model.get_weights()
            # else:
            #     if corr_logs[-1] < avg_corr_current:
            #         corr_logs.append(avg_corr_current)
            #         best_weights = self.model.get_weights()
            #         count = 0
            #     else:
            #
            #         count = count + 1
            #         print(count)
            #         if count == 5:
            #             all_corr_logs.clear()
            #             self.model.stop_training = True
            #             self.model.set_weights(best_weights)

            if avg_attack_traces[-1, correct_key] > 0:
                print("攻击失败", )
                print("GE:", avg_attack_traces[-1, correct_key])
                print("corr:", avg_corr_current)
                print("攻击失败", file=log)
                print("corr:", avg_corr_current, file=log)
                print("GE:", avg_attack_traces[-1, correct_key], file=log)
                logs['tge0'] = 5000


            else:
                print("攻击成功")
                print("corr:", avg_corr_current)
                if np.argmax(avg_attack_traces[:, correct_key] < 1) != 0:
                    print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1))
                    print("TGE0:", np.argmax(avg_attack_traces[:, correct_key] < 1), file=log)
                    logs['tge0'] = np.argmax(avg_attack_traces[:, correct_key] < 1)
                else:
                    print("TGE0:", np.argmax(avg_attack_traces[1:, correct_key] < 1))
                    print("TGE0:", np.argmax(avg_attack_traces[1:, correct_key] < 1), file=log)
                    logs['tge0'] = np.argmax(avg_attack_traces[1:, correct_key] < 1)

                print("攻击成功", file=log)
                print("corr:", avg_corr_current, file=log)

            return logs['tge0']


def adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)
    return loss


def cer_adjustment_loss(y_true, y_pred):
    y_true = y_true[:, :classes]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    k_star_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    num_of_attacks = 10
    fake_k_store = 0
    for i in range(num_of_attacks):
        shuffled_y_true = tf.random.shuffle(y_true)
        fake_k_loss = tf.keras.losses.categorical_crossentropy(shuffled_y_true, y_pred)
        fake_k_store = fake_k_store + fake_k_loss
    average_fake_k_loss = fake_k_store / num_of_attacks
    loss_cer = k_star_loss / average_fake_k_loss
    return loss_cer


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

    # adaptive_margin =  1.0 / np.sqrt(np.sqrt(label_freq_array))
    # adaptive_margin = adaptive_margin * (0.5 / np.max(adaptive_margin))
    # adaptive_margin = tf.cast(adaptive_margin, dtype=tf.float32)
    return adjustments


def build_model_mlp(hp):
    '''original network'''

    activation = hp.Choice('activation', values=["relu", "tanh", "selu", "elu"])

    model = tk.models.Sequential()
    model.add(layers.Dense(units=hp.Int('dense_' + str(0) + '_units',
                                        min_value=100,
                                        max_value=1000,
                                        step=100),
                           activation=activation,
                           input_shape=(X_profiling.shape[1],)))

    for j in range(hp.Int('n_dense_layers', 1, 7)):
        model.add(layers.Dense(units=hp.Int('dense_' + str(j) + '_units',
                                            min_value=100,
                                            max_value=1000,
                                            step=100),
                               activation=activation,
                               ))
    model.add(Dense(9))
    loss_auto = cer_adjustment_loss
    model.compile(optimizer=RMSprop(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
                  loss=loss_auto, metrics=companion_metric)

    model.summary()
    return model


def build_model_cnn(hp):
    '''original network'''

    model = tk.models.Sequential()
    model.add(Conv1D(hp.Int('conv_0_filters', min_value=4, max_value=16, step=4),
                     hp.Int('conv_0_kernal_size', min_value=26, max_value=52, step=2), padding='same',
                     input_shape=(len(X_profiling[0]), 1)))
    model.add(Activation("selu"))
    model.add(AveragePooling1D(2, strides=2))

    for i in range(hp.Int('n_conv_layers', 1, 2)):
        model.add(Conv1D(hp.Int('conv_' + str(i + 1) + '_filters', min_value=4, max_value=16, step=4),
                         hp.Int('conv_' + str(i + 1) + '_kernal_size', min_value=26, max_value=52, step=2),
                         padding='same'))
        model.add(Activation("selu"))
        model.add(AveragePooling1D(2, strides=2))

    model.add(Flatten())
    for j in range(hp.Int('n_dense_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Choice('dense_' + str(j) + '_units', values=[50, 100, 200, 300, 400, 500]),
                               activation="selu"))

    model.add(Dense(9))
    loss_auto = cer_adjustment_loss
    model.compile(optimizer=Adam(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
                  loss=loss_auto, metrics=companion_metric)

    model.summary()
    return model


def build_model_cnn_1(hp):
    img_input = Input(shape=(len(X_profiling[0]), 1))
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)

    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)

    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)

    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)

    x = Dense(classes)(x)


if __name__ == '__main__':

    """变量配置"""
    searching_method = 'BO'
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'MLP'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    num_traces_attacks = 5000
    save_root = 'root the save the searching history'
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 11
    learning_rate = 0.0005
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    rank_logs = []
    all_rank_logs = []
    corr_logs = []
    all_corr_logs = []
    loss_logs = []
    kl_loss_logs = []
    count = 0
    best_weights = None
    max_m = 0.5
    adjust_flag = True
    current_time = datetime.now()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    experiment_time = 'time_is{}_{}_{}'.format(int(day),
                                               int(hour),
                                               int(minute)
                                               )

    """数据导入"""
    (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key, attack_byte, num_profiling_traces = read_data(
        leakage_model,
        data_arguementation,
        data_arguementation_level,
        attack_model, dataset,
        sigma_hw, sigma_id)

    adjustments = compute_adjustment(Y_profiling, 1)
    adjustments_valid = tf.cast(adjustments, dtype=tf.float32)
    objective = "tge0"
    direction = 'min'
    max_trails = 10
    """创建神经网络模型"""

    if attack_model == 'MLP':
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
        build_model = build_model_mlp
    else:
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
        build_model = build_model_cnn

    log = open('F:/result/auto/' + str(dataset) + '/' + str(X_profiling.shape[0]) + '.txt', mode='a',
               encoding='utf-8')
    print(file=log)
    print(file=log)
    tuner = BayesianOptimization(hypermodel=build_model,
                                 objective=kt.Objective(objective, direction=direction),
                                 max_trials=max_trails,
                                 executions_per_trial=1,
                                 directory=save_root,
                                 project_name=experiment_time + '_' + dataset + '_' + attack_model + '_' + leakage_model + '_' + searching_method + '_' + objective + '_' + str(
                                     2),
                                 overwrite=True)

    """ select the output_metric function """
    callback = [
        all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))
        #     OneCycleLR(len(X_profiling), 256, 5e-3, end_percentage=0.2, scale_percentage=0.1,
        #                           maximum_momentum=None, minimum_momentum=None, verbose=True)
    ]
    batch_size = 32
    tuner.search_space_summary()
    tuner.search(x=X_profiling, y=Y_profiling[:, :9], callbacks=callback, epochs=15, batch_size=batch_size, verbose=2)
    tuner.results_summary()

    print('Retrain the best model with 10 epochs...')
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)
    model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=20, epochs=15,
              callbacks=[all(validation=(X_attack[:5000], plt_attack[:5000], Y_attack[:5000]))])

    # """最佳模型的存储地址"""
    # test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
    #                                                                              epoch,
    #                                                                              batch_size,
    #                                                                              output_metric,
    #                                                                              )
    # model_root = 'Model/'
    #
    # filename = model_root + test_info
    # """开始训练"""
    # history = model.fit(x=X_profiling, y=Y_profiling[:, :9], batch_size=batch_size, verbose=2, epochs=epoch,
    #                     callbacks=callback)
    # history.history.keys()
    # loss = history.history['loss']
    #
    # print(file=log)
    # print(experiment_time, file=log)
    # print("learning_rate_", learning_rate,
    #       "___architecture_", attack_model,
    #       "___dataset_", dataset,
    #       "___epochs_", epoch,
    #       "___num_profiling_traces_", num_profiling_traces,
    #       "___num_attack_traces_", num_traces_attacks,
    #       "___sigma_hw_", sigma_hw,
    #       "___sigma_id_", sigma_id,
    #       "___adjust_flag_", adjust_flag,
    #       "___max_m_", max_m, file=log)
    # print("best_corr", np.max(all_corr_logs), file=log)
    # for epoch in range(len(loss)):
    #     print("轮数", epoch, "__train_loss_", loss[epoch], file=log)
    # log.close()
