import tensorflow as tf

x = tf.constant([1, 2, 3])
w = tf.constant([[1, 2, 6], [1, 2, 3]])
y_true = tf.constant([[1, 0, 0], [0, 1, 0]])
cb_factor_power = tf.pow(x, w)
cb_factor_power_y_true = tf.multiply(cb_factor_power, y_true)
print(cb_factor_power_y_true)
