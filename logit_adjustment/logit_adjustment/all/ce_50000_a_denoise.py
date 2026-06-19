from data_load_ascad_50000_denoise import read_data
from datetime import datetime
import matplotlib
from keras.callbacks import EarlyStopping

matplotlib.use('TkAgg')
# 设置字体

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Cropping1D
from tensorflow.keras.models import Model


def build_denoising_autoencoder(input_shape=(700, 1)):
    # 定义论文中选择的激活函数
    activation_function = 'selu'

    # --- 编码器 (Encoder) ---
    # 编码器使用卷积和池化层来压缩输入
    input_trace = Input(shape=input_shape)

    # 编码器模块 1
    # 对应表2中的 Conv1 和 MaxPooling1
    x = Conv1D(filters=4, kernel_size=10, activation=activation_function, padding='same')(input_trace)
    x = BatchNormalization()(x)  # 使用BN防止过拟合并加速训练
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # 编码器模块 2
    # 对应表2中的 Conv2 和 MaxPooling2
    x = Conv1D(filters=8, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4, padding='same')(x)

    # 编码器模块 3
    # 对应表2中的 Conv3 和 MaxPooling3
    x = Conv1D(filters=16, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling1D(pool_size=8, padding='same')(x)

    # --- 解码器 (Decoder) ---
    # 解码器使用卷积和上采样层来重建输入

    # 解码器模块 1
    # 对应表2中的 Conv4 和 UpSampling1
    x = Conv1D(filters=16, kernel_size=10, activation=activation_function, padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=8)(x)

    # 解码器模块 2
    # 对应表2中的 Conv5 和 UpSampling2
    x = Conv1D(filters=8, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=4)(x)

    # 解码器模块 3
    # 对应表2中的 Conv6 和 UpSampling3
    x = Conv1D(filters=4, kernel_size=10, activation=activation_function, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(size=2)(x)

    # 输出层
    # 最后的卷积层 (Conv7) 将能量迹重建为原始尺寸
    # 使用大小为1的卷积核来生成最终的输出通道
    # 线性激活函数 (linear) 适合用于重建信号
    x = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')(x)

    decoded = Cropping1D(cropping=(2, 2))(x)  # 新的输出 (None, 700, 1)

    # --- 构建并编译模型 ---
    autoencoder = Model(input_trace, decoded)

    optimizer = tf.keras.optimizers.Adam()
    loss_function = 'mean_squared_error'

    autoencoder.compile(optimizer=optimizer, loss=loss_function)

    return autoencoder


if __name__ == '__main__':

    """变量配置"""
    dataset = 'ASCAD'  # ASCAD/ASCAD_rand/CHES_CTF
    leakage_model = 'HW'
    attack_model = 'CNN'  # MLP/CNN
    sigma_hw = 0  # sigma for the HW leakage model
    sigma_id = 0  # sigma for the ID leakage model
    epoch_count = 0
    # Select leakage model
    if leakage_model == 'ID':
        classes = 256
    else:
        classes = 9
    data_arguementation = False  # enable/disbale data arguementation
    data_arguementation_level = 0.25  # data arguementation level
    epoch = 100
    learning_rate = 5e-3
    output_metric = "all"  # rank/corr
    companion_metric = None  # None/all/kl_loss_model/'categorical_accuracy'
    model_size = 64  # the size of the profiling model
    rank_logs = []
    all_rank_logs = []
    corr_logs = []
    all_corr_logs = []
    loss_logs = []
    kl_loss_logs = []
    count = 1
    best_weights = None
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

    batch_size = 128

    """创建神经网络模型"""
    """ select the output_metric function """

    model = build_denoising_autoencoder(input_shape=(700, 1))
    model.summary()

    """最佳模型的存储地址"""
    test_info = 'dataset{}_leakage_model{}_epoch{}_batch_size{}_output{}'.format(dataset, leakage_model,
                                                                                 epoch,
                                                                                 batch_size,
                                                                                 output_metric,
                                                                                 )
    model_root = 'Model/'

    filename = model_root + test_info
    """开始训练"""
    callback = EarlyStopping(monitor='loss', min_delta=0.0002, patience=6, mode='min',
                             restore_best_weights=True)
    history = model.fit(x=X_profiling, y=Y_profiling, batch_size=128, verbose=2, validation_data=(X_attack, Y_attack),
                        epochs=epoch, callbacks=[callback]
                        )
    history.history.keys()
    loss = history.history['loss']

    model_root = 'E:/logit_backup/ensemble_model/' + test_info + "denoise_ascad" + '.h5'
    model.save(model_root)
