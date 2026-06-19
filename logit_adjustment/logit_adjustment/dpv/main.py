import os.path
import time
import requests
import tensorflow.keras as tk
from tensorflow.keras import backend as K
import tensorflow as tf
from random import random
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import preprocessing
from exp import *
import matplotlib.pyplot as plt


def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

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

def shuffle_data(profiling_x, label_y):
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

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
    y_true = y_true[:, :9]

    if adjust_flag:
        y_pred = y_pred + 1 * adjustments

    y_pred = tf.nn.softmax(y_pred, 1)
    loss = tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss

### CNN network
def cnn_architecture(input_size=4000, learning_rate=0.00001, classes=9):
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
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=flr_adjustment_loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]

#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, epochs=150, batch_size=100):

    # Get the input layer shape

    Reshaped_X_profiling, Reshaped_X_test = X_profiling.reshape(
        (X_profiling.shape[0], X_profiling.shape[1], 1)), X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_data=(Reshaped_X_test, Y_test),
                        batch_size=batch_size, verbose=1, epochs=epochs)
    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root = "./"
DPAv4_data_folder = root



# Choose the hyperparameter's values
nb_epochs = 50
batch_size = 50
input_size = 4000
learning_rate = 1e-3
nb_traces_attacks = 40
adjust_flag = True
tro = 0.45
nb_attacks = 50
real_key = np.load(DPAv4_data_folder + "key.npy")
mask = np.load(DPAv4_data_folder + "mask.npy")
att_offset = np.load(DPAv4_data_folder + "attack_offset_dpav4.npy")

start = time.time()

# Load the profiling traces
(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (
np.load(DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy')), (
np.load(DPAv4_data_folder + 'attack_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_labels_dpav4.npy')), (
np.load(DPAv4_data_folder + 'profiling_plaintext_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_plaintext_dpav4.npy'))
Y_profiling = calculate_HW(Y_profiling)  # Y_profiling是十进制，需要转换为二进制，并计算汉明重量
Y_attack = calculate_HW(Y_attack)
Y_profiling = to_categorical(Y_profiling, num_classes=9)
Y_attack = to_categorical(Y_attack, num_classes=9)
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

#################################################
#################################################

####                Training               ######

#################################################
#################################################

# Choose your model
model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
model_name = "DPA-contest_v4"

print('\n Model name = ' + model_name)

print("\n############### Starting Training #################\n")

# Record the metrics
history = train_model(X_profiling[:4000], Y_profiling[:4000], X_profiling[4000:], Y_profiling[4000:], model,
                       epochs=nb_epochs, batch_size=batch_size)
end = time.time()

print('Execution Time = %d' % (end - start))

print("\n############### Training Done #################\n")



#################################################
#################################################

####               Prediction              ######

#################################################
#################################################


print("\n############### Starting Predictions #################\n")

predictions = model.predict(X_attack)
predictions =   tf.nn.softmax(predictions, 1)

print("\n############### Predictions Done #################\n")


#################################################
#################################################

####            Perform attacks            ######

#################################################
#################################################

print("\n############### Starting Attack on Test Set #################\n")

avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, mask=mask,
                           offset=att_offset, byte=0, filename=model_name)

print("\n t_GE = ")
print(np.where(avg_rank <= 0))

print("\n############### Attack on Test Set Done #################\n")
def generate_image(text="程序已跑完", output_path="completed_matplotlib.png"):
    # 设置图像参数
    fig = plt.figure(figsize=(6, 3), dpi=280)  # 图像尺寸（宽600px，高300px）
    ax = plt.axes([0, 0, 1, 1], frameon=False)  # 全屏显示，无边框
    plt.axis('off')  # 隐藏坐标轴

    # 设置背景颜色（白色）
    ax.set_facecolor('white')

    # 设置中文字体（需要指定字体路径）
    # 中文字体路径示例（根据系统调整）：
    # Windows: 'C:/Windows/Fonts/simhei.ttf'
    # Linux: '/usr/share/fonts/truetype/arphic/uming.ttc'
    # macOS: '/System/Library/Fonts/Supplemental/Song.ttf'
    font_path = 'simhei.ttf'  # 如果使用默认字体可删除此行和下面的 FontProperties

    # 自定义字体（中文字体需要指定路径）
    font_properties = {
        'family': 'SimHei',  # 中文字体名称（需系统支持）
        'size': 24,  # 字体大小
        'color': 'black',  # 字体颜色
        'weight': 'bold'  # 粗体
    }

    # 如果需要指定字体路径，使用FontProperties
    # from matplotlib.font_manager import FontProperties
    # prop = FontProperties(fname=font_path)
    # font_properties['fontproperties'] = prop

    # 添加文本（居中显示）
    ax.text(
        0.5,  # x坐标（0-1比例）
        0.5,  # y坐标（0-1比例）
        text,
        ha='center',  # 水平居中
        va='center',  # 垂直居中
        **font_properties
    )

    plt.show()  # 显示图形


generate_image()