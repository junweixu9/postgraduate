from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers


def feature_extractor_layer():
    f_input = Input(shape=(700,), name='f')
    l_input = Input(shape=(9,), name='l')

    decoder = Dense(9, activation='relu', name="bigman")

    ff_input = Dense(100, activation='relu', name="encoder_f")(f_input)
    f_mu = Dense(100, activation='elu', name="f_mu")(ff_input)
    f_log_var = Dense(100, activation='elu', name="f_var")(ff_input)
    rand_f = tf.random.normal(f_mu.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    df = f_mu + rand_f * tf.exp(f_log_var) ** 0.5
    decoder_f_output = decoder(ff_input)
    decoder_f_output = Reshape((9,), name="output1")(decoder_f_output)

    ll_input = Dense(100, activation='elu', name="encoder_l")(l_input)
    l_mu = Dense(100, activation='elu', name="l_mu")(ll_input)
    l_log_var = Dense(100, activation='elu', name="l_var")(ll_input)
    rand_l = tf.random.normal(l_mu.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    dl = l_mu + rand_l * tf.exp(l_log_var) ** 0.5
    decoder_l_output = decoder(dl)
    decoder_l_output = Reshape((9,), name="output2")(decoder_l_output)

    sigma2 = l_log_var.detach()

    sf = cosine_similarity(df, df).reshape(-1)
    sl = cosine_similarity(dl, dl).reshape(-1)

    model = Model(inputs=[f_input, l_input],
                  outputs=[decoder_f_output, decoder_l_output, sigma2, sf, sl])

    model.compile(loss=[loss_pred, loss_target, loss_recovery, loss_similarity], optimizer='adam',
                  oss_weights=[1.0, 0.2, 0.2, 0.2])
    model.summary()
    return model


def loss_pred(y_true, y_pred):
    decoder_f_output = y_pred
    loss = loss_func(decoder_f_output, y_true)
    return loss


def loss_target(y_true, y_pred):
    f_log_var, sigma2, f_mu, l_mu = y_pred
    loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
        f_log_var - sigma2 - tf.exp(f_log_var) / tf.exp(sigma2) - (f_mu - l_mu.detach()) ** 2 / tf.exp(
            sigma2) + 1, dim=1))
    return loss


def loss_recovery(y_true, y_pred):
    decoder_l_output = y_pred[5]
    loss = tf.keras.losses.categorical_crossentropy(decoder_l_output, y_true)
    return loss


def loss_similarity(y_true, y_pred):
    sf, sl = y_pred[6:]
    loss = tf.reduce_sum(tf.losses.mean_squared_error(sf, sl))
    return loss


def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''
    assert x.size(1) == y.size(1)
    x = tf.norm(x, dim=1)
    y = tf.norm(y, dim=1)
    return x @ y.transpose(0, 1)


from sklearn.datasets import make_classification


def loss_1(y_true, y_pred):
    decoder_l_output = y_pred
    loss = tf.keras.losses.categorical_crossentropy(decoder_l_output, y_true)
    return loss


# X, y = make_classification(n_samples=15, n_features=3, n_redundant=0, n_clusters_per_class=1, n_informative=1,
#                            n_classes=2, random_state=20)
# y = to_categorical(y, num_classes=2)
#
# decoder = Dense(2, name="bigman")
#
# f_input = Input(shape=(3,), name='f')
# ff = Dense(3, activation='relu', name='ff')(f_input)
# fff = decoder(ff)
# f_output = Softmax()(fff)
# f_output = Reshape((2,), name='output1')(f_output)
#
# l_input = Input(shape=(2,), name='l')
# ll = Dense(3, activation='relu', name="ll")(l_input)
# lll = decoder(ll)
# l_output = Softmax()(lll)
# l_output = Reshape((2,), name='output2')(l_output)
#
# model = Model(inputs=[f_input, l_input],
#               outputs=[f_output, l_output])
# model.compile(loss={"output1": loss_1, "output2": loss_1}, optimizer='adam', loss_weights=[1, 0.5])
# model.summary()
# history = model.fit(x={"f": X, "l": y}, y={'output1': y, 'output2': y}, batch_size=10, verbose=2, epochs=5)
# history.history.keys()
# loss = history.history['loss']

# def generator_loss(ll, outputs):
#     gan_loss = tf.losses.categorical_crossentropy(ll, outputs)
#     return gan_loss
#
#
# def Generator():
#     l = tf.keras.layers.Input(shape=[2, ])
#     ll = Dense(2, activation='relu')(l)
#     outputs = Dense(2, activation='softmax')(ll)
#
#     return tf.keras.Model(inputs=l, outputs=[ll, outputs])
#
#
# @tf.function
# def train_step(input_image, target):
#     with tf.GradientTape() as gen_tape:
#         gen_output = generator(input_image, training=True)
#
#         gen_gan_loss = generator_loss(target, gen_output[1])
#
#     generator_gradients = gen_tape.gradient(gen_gan_loss,
#                                             generator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(generator_gradients,
#                                             generator.trainable_variables))
#     print(gen_gan_loss)
#

#
# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# generator = Generator()
#
#     train_step(X, y)
#
#
# train_loss_results = []
# train_accuracy_results = []
#
# num_epochs = 201
#
# for epoch in range(num_epochs):
#   epoch_loss_avg = tf.keras.metrics.Mean()
#   epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#
#   # Training loop - using batches of 32
#   for x, y in ds_train_batch:
#     # Optimize the model
#     loss_value, grads = grad(model, x, y)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     # Track progress
#     epoch_loss_avg.update_state(loss_value)  # Add current batch loss
#     # Compare predicted label to actual label
#     # training=True is needed only if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     epoch_accuracy.update_state(y, model(x, training=True))
#
#   # End epoch
#   train_loss_results.append(epoch_loss_avg.result())
#   train_accuracy_results.append(epoch_accuracy.result())
#
#   if epoch % 50 == 0:
#     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
#                                                                 epoch_loss_avg.result(),
#                                                                 epoch_accuracy.result()))


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self):
        super(CVAE, self).__init__()
        self.f_input = tf.keras.layers.InputLayer(input_shape=(2,))
        # l_input = Input(shape=(9,), name='l')
        #
        # decoder = Dense(9, activation='relu', name="bigman")
        #
        # ff_input = Dense(100, activation='relu', name="encoder_f")(f_input)
        # f_mu = Dense(100, activation='elu', name="f_mu")(ff_input)
        # f_log_var = Dense(100, activation='elu', name="f_var")(ff_input)
        # rand_f = tf.random.normal(f_mu.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        # df = f_mu + rand_f * tf.exp(f_log_var) ** 0.5
        # decoder_f_output = decoder(ff_input)
        # decoder_f_output = Reshape((9,), name="output1")(decoder_f_output)
        #
        # ll_input = Dense(100, activation='elu', name="encoder_l")(l_input)
        # l_mu = Dense(100, activation='elu', name="l_mu")(ll_input)
        # l_log_var = Dense(100, activation='elu', name="l_var")(ll_input)
        # rand_l = tf.random.normal(l_mu.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        # dl = l_mu + rand_l * tf.exp(l_log_var) ** 0.5
        # decoder_l_output = decoder(dl)
        # decoder_l_output = Reshape((9,), name="output2")(decoder_l_output)
        #
        # sigma2 = l_log_var.detach()
        #
        # sf = cosine_similarity(df, df).reshape(-1)
        # sl = cosine_similarity(dl, dl).reshape(-1)
        # self.encoder = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(700,)),
        #         tf.keras.layers.Dense(
        #             4, activation='relu'),
        #         tf.keras.layers.Dense(
        #             2, activation='softmax')
        #     ]
        # )

        # self.decoder = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        #         tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
        #         tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
        #         tf.keras.layers.Conv2DTranspose(
        #             filters=64, kernel_size=3, strides=2, padding='same',
        #             activation='relu'),
        #         tf.keras.layers.Conv2DTranspose(
        #             filters=32, kernel_size=3, strides=2, padding='same',
        #             activation='relu'),
        #         # No activation
        #         tf.keras.layers.Conv2DTranspose(
        #             filters=1, kernel_size=3, strides=1, padding='same'),
        #     ]
        # )

    @tf.function
    def encode(self, x, y):
        mean = self.encoder(x)
        l = self.encoder(y)
        return mean, l

    def decode(self, y):
        decoder = self.f_input(y)
        return decoder


def generator_loss(model, x, target):
    # outputs, oo = model.encode(x, target)
    oo = model.decode(target)
    outputs = tf.nn.softmax(oo, 1)
    loss_1 = tf.reduce_mean(tf.losses.categorical_crossentropy(outputs, target))
    # gan_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(target, outputs))
    # return loss_1, gan_loss
    return loss_1


def train_step(model, x, target, optimizer):
    with tf.GradientTape() as tape:
        loss_1 = generator_loss(model, x, target)
        print(loss_1)
    gradients = tape.gradient(loss_1, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


X, y = make_classification(n_samples=15, n_features=2, n_redundant=0, n_clusters_per_class=1, n_informative=1,
                           n_classes=2, random_state=20)
y = to_categorical(y, num_classes=2)

model = CVAE()
optimizer = tf.keras.optimizers.Adam(1e-4)
for epoch in range(1, 100):
    train_step(model, X, y, optimizer)
