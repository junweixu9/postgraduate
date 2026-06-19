from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
import tensorflow as tf

X, y = make_classification(n_samples=100, n_features=3, n_redundant=0, n_clusters_per_class=1, n_informative=1,
                           n_classes=2, random_state=20)
y = to_categorical(y, num_classes=2)


# f_input = Input(shape=(3,), name='f')
# ff_input_1 = Dense(3, activation='relu', name='tiqiande')(f_input)
# ff_input_2 = Dense(2, activation='softmax', name='output')(ff_input_1)
# l_input = Input(shape=(2,), name='l')
#
# model = Model(inputs=[f_input, l_input],
#               outputs=[ff_input_1, ff_input_2])
# model.compile(loss={"output": loss_1}, optimizer='adam')
# model.summary()
# history = model.fit(x={"f": X, "l": y}, y={'output': y, 'tiqiande': y}, batch_size=10, verbose=2, epochs=5)
# history.history.keys()
# loss = history.history['loss']
# acc = history.history['acc']
def loss_1(y_true, y_pred):
    decoder_l_output = y_pred
    loss = tf.keras.losses.categorical_crossentropy(decoder_l_output, y_true)
    return loss


inputs = keras.Input(shape=(3,), name="digits")
x1 = layers.Dense(2, activation="relu")(inputs)
x2 = layers.Dense(2, activation="softmax")(x1)
model = keras.Model(inputs=inputs, outputs=x2)
model.compile(loss=loss_1, optimizer='adam')
model.summary()
history = model.fit(x=X, y=y, batch_size=10, verbose=2, epochs=5)
history.history.keys()
loss = history.history['loss']
