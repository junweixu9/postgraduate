# import os
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_datasets as tfds
# import argparse
#
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
#
#
# @tf.function
# def true_label_preprocess(image, label):
#     return (image, tf.expand_dims(label, 1)), label
#
#
# @tf.function
# def crop_box_ratio(image, bbox):
#     height = tf.cast(tf.shape(image)[0], tf.float32)
#     width = tf.cast(tf.shape(image)[1], tf.float32)
#     image = tf.image.crop_to_bounding_box(image, tf.cast(bbox[0] * height, tf.int32), tf.cast(bbox[1] * width, tf.int32), tf.cast((bbox[2] - bbox[0]) * height, tf.int32), tf.cast((bbox[3] - bbox[1]) * width, tf.int32))
#     return image
#
#
# @tf.function
# def random_crop(image, minval=0.8):
#     image_size = tf.shape(image)
#     if image_size.shape == 3:
#         image_height = image_size[0]
#         image_width = image_size[1]
#         image = tf.image.random_crop(image,
#                                      [tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_height, tf.float32), tf.int32),
#                                       tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_width, tf.float32), tf.int32), 3])
#
#     if image_size.shape == 4:
#         batchsize = image_size[0]
#         image_height = image_size[1]
#         image_width = image_size[2]
#         image = tf.image.random_crop(image,
#                                      [batchsize, tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_height, tf.float32), tf.int32),
#                                       tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_width, tf.float32), tf.int32), 3])
#     return image
#
#
# @tf.function
# def preprocess_caltech_image(features, img_size=tf.constant(224)):
#     image = features['image']
#     bbox = features['bbox']
#     label = features['label']
#
#     image = crop_box_ratio(image, bbox)
#     image = tf.image.resize_with_pad(image, img_size, img_size)
#     return image, label
#
#
# @tf.function
# def picture_augment(image, img_size=tf.constant(224)):
#     import tensorflow_addons as tfa
#
#     image = tf.image.adjust_brightness(image, tf.random.uniform([], minval=0.5, maxval=1))
#     image = tf.image.adjust_hue(image, tf.random.uniform([], minval=-0.01, maxval=0.01))
#     image = tf.image.adjust_saturation(image, tf.random.uniform([], minval=0.9, maxval=1.1))
#     image = tf.image.random_flip_left_right(image)
#
#     image = tfa.image.rotate(image, tf.constant(np.pi / 8) * tf.random.uniform([], minval=-1, maxval=1), name='rotate')
#     image = random_crop(image)
#     image = tf.image.resize_with_pad(image, img_size, img_size)
#
#     return image
#
#
# def create_caltechbirds2011_dataset(data_dir, BATCH_SIZE=32, n_shuffle=1000):
#     (train_ds, test_ds), info = tfds.load('caltech_birds2011', split=['train', 'test'], shuffle_files=True, data_dir=data_dir, with_info=True)
#
#     train_batches = train_ds.map(lambda x: preprocess_caltech_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(n_shuffle).batch(BATCH_SIZE).map(lambda x, y: (picture_augment(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).map(true_label_preprocess)
#     test_batches = test_ds.map(lambda x: preprocess_caltech_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(BATCH_SIZE).map(true_label_preprocess)
#
#     num_classes = info.features['label'].num_classes
#     return train_batches, test_batches, num_classes
#
#
# def create_model(modelname, num_classes, img_size=224):
#     from CustomLayer import CosineLayer
#
#     feature_extractor_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", name='efficientnetB0')
#     feature_extractor_layer.trainable = True
#
#     input_image = tf.keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name='input_image')
#     efficientnet_output = feature_extractor_layer(input_image)
#
#     y_true = tf.keras.Input(shape=1, dtype=tf.int64, name='true_label')
#
#     if modelname == 'AdaCos':
#         from CustomLayer import AdaCos_logits
#
#         cos_layer = CosineLayer(num_classes=num_classes)
#         cos_layer_output = cos_layer(efficientnet_output)
#
#         logits = AdaCos_logits()([cos_layer_output, y_true])
#
#         model = tf.keras.models.Model(inputs=(input_image, y_true), outputs=tf.keras.layers.Softmax()(logits))
#         return model
#
#     elif modelname == 'fixedAdaCos':
#         from CustomLayer import CorrectCosMean
#
#         cos_layer = CosineLayer(num_classes=num_classes)
#         cos_layer_output = cos_layer(efficientnet_output)
#
#         logits = CorrectCosMean()([cos_layer_output, y_true])
#
#         fixed_s = tf.constant(tf.math.sqrt(2.) * tf.math.log(tf.cast(num_classes - 1, tf.float32)), name='fixed_s')
#
#         model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=tf.keras.layers.Softmax()(fixed_s * logits))
#         return model
#
#     if modelname == 'ArcFace':
#         from CustomLayer import ArcFace_logits
#
#         cos_layer = CosineLayer(num_classes=num_classes)
#         cos_layer_output = cos_layer(efficientnet_output)
#
#         logits = ArcFace_logits()([cos_layer_output, y_true])
#
#         model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=tf.keras.layers.Softmax()(logits))
#         return model
#
#     elif modelname == 'l2-softmax':
#         alpha = 16
#         normalize_output = tf.math.l2_normalize(efficientnet_output)
#
#         top_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
#
#         model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=top_layer(alpha * normalize_output))
#         return model
#
#     elif modelname == 'softmax':
#         top_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
#         model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=top_layer(efficientnet_output))
#
#         return model
#
#
# def train(model, train_batches, test_batches, epochs, logdir):
#     import re
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logdir), profile_batch='800,900')
#
#     checkpoint_prefix = logdir / Path("ckpt_{epoch}")
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(str(checkpoint_prefix), save_weights_only=True, verbose=1)
#
#     model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                   metrics=tf.keras.metrics.SparseCategoricalAccuracy())
#
#     initial_epoch = 0
#     ckpt = tf.train.latest_checkpoint(logdir, latest_filename=None)
#     if ckpt:
#         model.load_weights(ckpt)
#         initial_epoch = int(re.sub(r".*_", "", ckpt))
#
#     history = model.fit(train_batches,
#                         initial_epoch=initial_epoch,
#                         epochs=epochs,
#                         validation_data=test_batches,
#                         callbacks=[tensorboard_callback, cp_callback])
#
#     return history
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-m", "--modelname", choices=["AdaCos", "ArcFace", "fixedAdaCos", "l2-softmax", "softmax"], default="AdaCos")
#     args = parser.parse_args()
#     modelname = args.modelname
#
#     train_batches, test_batches, num_classes = create_caltechbirds2011_dataset(data_dir=Path("tmp"))
#     model = create_model(modelname, num_classes)
#
#     model.summary()
#
#     _ = train(model, train_batches, test_batches, epochs=200, logdir=Path("tflog") / Path(modelname))
#
#
# if __name__ == '__main__':
#     main()
import numpy as np
import tensorflow as tf
# y_true = tf.constant([
#     [0.7, 0.2, 0.1],  # 样本1，最可能的类别是0
#     [0.9, 0, 0.1],  # 样本2，最可能的类别是1
#     [0.2, 0.2, 0.6]   # 样本3，最可能的类别是2
# ])
#
# # 计算每个样本最可能的类别索引
# y_true_index = tf.argmax(y_true, axis=-1)
# y_true_one_hot = tf.cast(tf.one_hot(y_true_index, depth=3),tf.float32)
# #
# ldam = tf.constant([0.1, 0.2, 0.3])
# adjustment = tf.multiply(ldam, y_true_one_hot)
# print()
# a = tf.constant([[3], [2]])  #广播 为[[3, 2], [3, 2]]
# b = tf.constant([[1, 0], [0, 9]])
# c = tf.multiply(a, b)
# print(c)

# noise = tf.random.normal(shape=[4, 1], mean=0, stddev=1 / 3)
#
# f = noise.shape[0]
# h = len(noise)
# g = tf.random.normal(shape=[noise.shape[0], noise.shape[1]], mean=0, stddev=1 / 3)
#
# noise_clip = tf.clip_by_value(noise, -1, 1)
#
# noise_clip_abs = tf.abs(noise_clip)
#
# a = tf.reduce_max(noise_clip, axis=1)
#
# mlist = np.array([[1., 0, 0.0],
#                   [0, 1., 0],
#                   [0, 0, 1]]
#                  )
# list = np.array([[0.4, 0.6, 0.3],
#                  [0.1, 0.7, 0.9],
#                  [0.39, 0.88, 0.2]])
#
# adjustments = np.array([0.1, 0.2, 0.3]
#                        )
#
# cos_adjustment = tf.multiply(mlist, (1 - list) / 2)
# cos_num_adjustment = tf.multiply(cos_adjustment, adjustments)
#
# ok = tf.multiply(mlist, list)
# x = tf.multiply(noise_clip_abs, mlist)
#
# print("noise", noise)
# print("noise_clip", noise_clip)
# print("noise_clip_abs", noise_clip_abs)
# print(a)
# print(g)
# print(h)
# print(x)

# classes = 9
# y = tf.zeros([4, 2, 2], dtype=tf.int32)
# y_true_i = tf.constant([[1], [0]])
# multi = tf.constant([[[1, 2], [3, 4]], [[2, 5], [1, 6]]], dtype=tf.int32)
# y = tf.tensor_scatter_nd_update(y, y_true_i, multi)

# print(y)
# print(y_true)
# print(y_true[0])
# n = tf.expand_dims(
#     y_true[0], 0)
# import scipy.stats as ss
#
# y = np.array([2, 1, 0, 1, 2, 0, 0])
# f = np.argmax(y < 1)
#
# y_pred = tf.constant([[5., 2.], [3., 5], [5., 10], [7., 11]])
# rank_row = ss.rankdata(y_pred, method='dense', axis=1)
# # tf.
#
# print(tf.square(y_pred - 1 / 9))
# y_true = tf.constant([[1., 2.], [3., 5], [5., 10], [7., 11]])
# _, var_true = tf.nn.moments(y_true, axes=[1])
# var_pred_non_sum = tf.square(y_pred - 1 / 9)
# var_pred = tf.reduce_mean(var_pred_non_sum, axis=1)
# diff_square = tf.square(var_true - var_pred)
# diff_square_sum = tf.reduce_sum(diff_square, axis=0)


# y_pred = tf.random.normal(shape=[5, 3], mean=0, stddev=1)
# y_true = tf.random.normal(shape=[5, 3], mean=0, stddev=1)
# square_diff_puv_container_representation = tf.random.normal(shape=[3, 3, 3], mean=0, stddev=1)
# print(square_diff_puv_container_representation)
# rank_puv_container_representation = tf.random.normal(shape=[3, 3, 3], mean=0, stddev=1)
#
# y_true_index = tf.argmax(y_true, 1)
# print(y_true_index)
# square_diff_puv_container = tf.gather(square_diff_puv_container_representation, y_true_index,)
# print(square_diff_puv_container)
# rank_puv_container = tf.tensor_scatter_nd_update(rank_puv_container, y_true_index,
#                                                  rank_puv_container_representation)
# print(rank_puv_container)
#
# # 给每一个样本计算p^uv矩阵：首先创建ijk矩阵和i_j_k矩阵
# ijk = tf.repeat(tf.expand_dims(y_pred, 1), repeats=3, axis=1)
# print(ijk)
# i_j_k = tf.transpose(ijk, perm=[0, 2, 1])
# print(i_j_k)
# predict_puv_container = tf.cast(tf.math.reciprocal(1 + tf.math.exp(-20 * (i_j_k - ijk))), dtype=tf.float32)
# print(predict_puv_container)


# noise = tf.random.normal(shape=[3, 3, 3], mean=0, stddev=1 / 3)
# noise_sun = tf.reduce_sum(noise, axis=[1, 2])
# print(noise)
# print(noise_sun)
#
#
# y_pred = tf.constant([[1, 3, 5], [0, 2, 6], [2, 4, 7]], dtype=tf.float32)
# ijk = tf.repeat(tf.expand_dims(
#     y_pred, 1), repeats=3, axis=1)
# print(ijk)
# i_j_k = tf.transpose(ijk, perm=[0, 2, 1])
# predict_puv_container = tf.cast(tf.math.reciprocal(1 + tf.math.exp(-20 * (i_j_k - ijk))), dtype=tf.float32)
# y_true_index = tf.argmax(y_true, axis=1)
# noise_match = tf.multiply(noise, predict_puv_container)
# print(noise)
# print(predict_puv_container)
# print(noise_match)
# predict_puv_container = tf.cast(1 + tf.math.exp(-20 * (i_j_k - ijk)), dtype=tf.float32)

# # print(n)
# print(repeat_n)

# label = tf.gather(y_true, indices=[1])
# label = tf.squeeze(label)
# x_label = tf.math.reciprocal(y_true)
# print(x_label)
# print(label)


def rank_ldl_loss_metrix(y_true, y_pred):
    # 给每一个样本赋上puv矩阵和（du-dv）^2矩阵
    rank_puv_container = tf.zeros([len(y_pred), 9, 9], dtype=tf.float32)
    square_diff_puv_container = tf.zeros([len(y_pred), 9, 9], dtype=tf.float32)
    y_true_index = tf.argmax(y_true, 1)
    square_diff_puv_container = tf.tensor_scatter_nd_update(square_diff_puv_container, y_true_index,
                                                            square_diff_puv_container_representation)
    rank_puv_container = tf.tensor_scatter_nd_update(rank_puv_container, y_true_index,
                                                     rank_puv_container_representation)

    # 给每一个样本计算p^uv矩阵：首先创建ijk矩阵和i_j_k矩阵
    ijk = tf.repeat(tf.expand_dims(y_pred, 1), repeats=9, axis=1)
    i_j_k = tf.transpose(ijk, perm=[0, 2, 1])
    predict_puv_container = tf.cast(tf.math.reciprocal(1 + tf.math.exp(-20 * (i_j_k - ijk))), dtype=tf.float32)

    # 根据rank_loss计算总值：二项损失*平方差
    binary_puv_predict_uv = tf.multiply(rank_puv_container, tf.math.log(predict_puv_container)) + tf.multiply(
        1 - rank_puv_container, tf.math.log(1 - predict_puv_container))

    all_9_9_rank_loss = tf.multiply(binary_puv_predict_uv,
                                    square_diff_puv_container)

    # mean_rank_loss = N = n * 9 * 9
    mean_all_9_9_rank_loss = tf.reduce_mean(all_9_9_rank_loss)

    return mean_all_9_9_rank_loss

nums = [0,1,0,3,12]
right_flag_0 = left_flag_0 = 0
for num in nums:
    right_flag_0 = right_flag_0 + 1
    if num != 0:
        nums[left_flag_0], nums[right_flag_0] = num, 0
        left_flag_0 = left_flag_0 + 1


