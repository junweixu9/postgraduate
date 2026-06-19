import tensorflow as tf


# 创建自定义优化器
class CustomOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001):
        super(CustomOptimizer, self).__init__()
        self.learning_rate = learning_rate

    def get_config(self):
        config = super(CustomOptimizer, self).get_config()
        config.update({
            'learning_rate': self.learning_rate,
        })
        return config

    def get_gradients(self, loss, params):
        grads = []
        for param in params:
            grad = 2 * param
            grads.append(grad)
        return grads

    def apply_gradients(self, grads_and_vars, name=None):
        for grad, var in grads_and_vars:
            var.assign_sub(self.learning_rate * grad)


# 创建线性回归模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 定义损失函数
loss = tf.losses.MeanSquaredError()

# 定义训练数据
x_train = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_train = tf.constant([[2.0], [4.0], [6.0], [8.0]])

# 编译模型
model.compile(optimizer=CustomOptimizer(learning_rate=0.01), loss=loss)

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=1)
