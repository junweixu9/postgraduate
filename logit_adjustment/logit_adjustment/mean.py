# import os
import numpy as np
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
#
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

# for i in range(5):
#     q = tf.random.shuffle(list)
#     print(q)


# Establish an identity operation, but clip during the gradient pass.
# @tf.custom_gradient
# def clip_gradients(y):
#   def backward(dy):
#     return tf.clip_by_norm(dy, 0.5)
#   return y, backward
#
# v = tf.Variable(2.0)
# with tf.GradientTape() as t:
#   output = clip_gradients(v * v)
# print(t.gradient(output, v))  # calls "backward", which clips 4 to 2

# @tf.custom_gradient
# def log1pexp(x):
#   e = tf.exp(x)
#   def grad(upstream):
#     return upstream * (1 - 1 / (1 + e))
#   return tf.math.log(1 + e), grad


# v = tf.Variable(100)
# with tf.GradientTape() as tape:
#   tape.watch(v)
#   y=log1pexp(v)
# dy_dx = tape.gradient(y, v) # Will be NaN when evaluated.


# tt = tf.constant([],dtype =tf.int32)
# for i in range(5):
#   t = tf.constant([i],dtype =tf.int32)
#   tt = tf.concat([tt,t],0)
# print(tt)

import scipy.stats as ss

# y = tf.constant([[1,3,4,5],
#                  [5,1,7,3],
#                  [2,5,6,1]])
#
# for row in y:
#   rank_round = ss.rankdata(row, method='dense')
#   print(rank_round)


# ASCAD_ldl_logit_45000_2000_hw1_tro1 = np.mean([325,195,229,152,252,237,241,205,265,572])
# ASCAD_ldl_45000_2000_hw1 = np.mean([500,724,565,998,627,540,513])
# ASCAD_r_ldl_45000_10000_hw2 = np.mean([389,771,434,747,787,766,677,780,587])
# ASCAD_r_ldl_50000_3000_hw2 = np.mean([385,435,730,1041,729,661,687,872,769])
# ASCAD_r_ldl_logit_45000_10000_hw2_tro25 = np.mean([464,261,526,422,379,439,264,490,343,327])
# ASCAD_r_ldl_logit_50000_3000_hw2_tro25 = np.mean([206,263,366,326,378,207,232,323,480,363])
# ches_ldl_logit_45000_3000 = np.mean([754,1284,456,381,864,685,945,649,578,380])
#
# CHES_5000_5000_tro_25 = np.mean([5000,3369,5000,3128,3273,3836,5000,1560,3774,5000,5000,5000,3173,3599,5000,1591,5000,5000,2451,5000,4179,3559,3825,5000])
# CHES_5000_5000_tro_125 = np.mean([5000,3902,5000,2151,3872,4292,3866,3145,5000,3663,5000,3109,5000,5000,3497,5000,3151])
# CHES_45000_5000_tro_25 = np.mean([535,869,677,675,663,457,380,1314,651,630,714,468,1122,637,435,745,310,554,396,669])
# CHES_10000_5000_tro_25 = np.mean([5000,1703,2773,1200,2663,2331,1012,1244,1637,1299,3892,1505,3631,2260,834,2474,1439,1382,2647,1432])
# CHES_10000_5000_false = np.mean([2422,5000,5000,3755,4090,3389,5000,2825,3566,5000,2201,5000,4555,3893,5000,2652,2744,5000,5000,2855])
# ASCAD_logit_50000_5000_tro1 = np.mean([171,217,139,220,171,97,103,120,242,142,245,168,127,209,140,115,161,109,93,131,199])
#
# ASCAD_logit_10000_5000_tro1 = np.mean([300,213,306,219,831,537,297,278,340,195,251,386,284,1654,293,538,721,423,749,345,873])
# ASCAD_r_logit_10000_5000_tro025 = np.mean([3919,5000,])
# ASCAD_ldl_logit_10000_5000_hw1_tro1 = np.mean([333,638,633,618,311,244,433,407,382,489,271,354,315,230,465,372,639,311,494,324,226,410,386,298])
# ASCAD_ldl_10000_5000_hw1 = np.mean([333,638,633,618,311,244,433,407,382,489,271,354,315,230,465,372,639,311,494,324,226,410,386,298])
# CHES_9000_5000_tro_025_hw2 = np.mean([911,2558,2783,2630,5000,1165,845,3124,1736,3979,1284,2939,3261,2494,1944,2823,1497,1493,1884])
# CHES_9000_5000_hw2 = np.mean([2330,5000,5000,2795,3297,5000,3574,3188,5000,5000,3387,3802,5000,2158,4293,2832,3946,5000])
# CHES_9000_5000_hw2_ge = np.mean([0,3,8,0,2,0,0,0,2,21,0,0,2,0,0,0,0,3])
# CHES_7000_5000_hw2 = np.mean([5000,5000,5000,5000,3620,2927,5000,5000,5000,1817,5000,3753,3288,4614,4348,3818,3849,4566,2252])
# CHES_7000_5000_hw2_ge = np.mean([20,0,16,1,2,0,0,1,2,5,0,2,0,0,0,0,0,0,0,0])
# CHES_7000_5000_tro_025_hw2 = np.mean([4721,3398,1672,2605,1224,743,3708,1203,2722,3452,2570,2827,4505,2714,5000,2858,5000,2607,5000,5000])
#
# ASCAD_5000_5000_hw1 = np.mean([3737,2150,3654,3033,3628,1598,1701,4570,3313,2823,880,3288,1792,1714,1673,5000,1325,2331,1888,5000])
# ASCAD_10000_5000_hw1 = np.mean([1251,1201,646,2183,3027,1992,1172,1889,1111,1292,1458,1167,888,1043,1658,1785,1225,5000,1152,1559])
# ASCAD_5000_5000_hw1_logit_tro1 = np.mean([612,446,535,274,362,489,709,508,529,381,537,528,674,462,321,362,596,442,524,712])
# ASCAD_5000_5000_logit_tro1 = np.mean([929,2260,1068,410,2426,1920,718,1053,5000,5000,1685,2481,2784,2372,2191,424,1667,756,888,756])
# ASCAD_5000_5000_hw1_ge = np.mean([0,0,0,])
#
# CHES_5000_5000 = np.mean([])
#
#

ar_hw2_logit_tro02775_3000 = np.array(
    [2552, 2743, 2868, 2888, 1841, 3747, 3001, 4583, 977, 3760, 2402, 5000, 2774, 2040, 2572, 3460, 2656, 4473, 3668,
     2548, 4453])
ar_hw2_logit_tro02775_3000_mean = np.median(ar_hw2_logit_tro02775_3000)

ar_hw2_logit_tro02725_3000 = np.array(
    [3543, 5000, 1046, 2335, 3964, 3794, 2683, 2387, 5000, 3349, 4267, 2548, 2301, 4217, 4099, 1606, 2073, 2516, 2115,
     3174])
ar_hw2_logit_tro02725_3000_mean = np.median(ar_hw2_logit_tro02725_3000)

ar_hw2_logit_tro028_3000 = np.array(
    [2493, 4347, 3653, 2447, 2314, 4440, 2888, 2424, 2115, 5000, 3824, 3190, 2374, 3911, 2244, 5000, 1977, 2962, 1772,
     4485])
ar_hw2_logit_tro028_3000_mean = np.median(ar_hw2_logit_tro028_3000)

ar_hw2_logit_tro0255_3000 = np.array(
    [2426, 3669, 2803, 3494, 5000, 3273, 3594, 4264, 3783, 4098, 2175, 2386, 2012, 3232, 3368, 2765, 3308, 3166, 2992,
     5000])
ar_hw2_logit_tro0255_3000_mean = np.median(ar_hw2_logit_tro0255_3000)

ar_hw2_logit_tro0275_1000 = np.array([3618, 4325, ])
ar_hw2_logit_tro0275_1000_mean = np.mean(ar_hw2_logit_tro0275_1000)

ar_hw2_logit_tro0275_3000 = np.array(
    [3058, 3407, 5000, 5000, 3340, 1789, 5000, 5000, 2699, 5000, 1597, 3207, 3074, 4664, 5000, 2588, 4513, 2532, 3541,
     4714, 3725])
ar_hw2_logit_tro0275_3000_mean = np.median(ar_hw2_logit_tro0275_3000)

ar_hw2_logit_tro0275_5000 = np.array(
    [1501, 1622, 1694, 1730, 1251, 1235, 1486, 827, 2389, 1824, 1407, 1974, 1104, 1248, 1722, 817, 1839, 1261, 2035,
     2211, 1006])
ar_hw2_logit_tro0275_5000_mean = np.mean(ar_hw2_logit_tro0275_5000)

ar_hw2_logit_tro0275_7000 = np.array(
    [1876, 2012, 1129, 1169, 990, 1410, 1018, 830, 1069, 1470, 922, 1261, 1178, 853, 1299, 941, 1269, 2796, 1701, 722])
ar_hw2_logit_tro0275_7000_mean = np.mean(ar_hw2_logit_tro0275_7000)

ascad_rand_hw2_logit_tro0275_10000 = np.mean(
    [1323, 794, 787, 646, 1003, 883, 1492, 1320, 696, 483, 852, 1597, 621, 1237, 723, 755, 857, 841, 1150, 573, 707,
     761])

ascad_rand_hw2_logit_tro027_10000 = np.mean(
    [1396, 971, 1224, 1273, 741, 557, 1184, 1673, 942, 723, 648, 1112, 1067, 833, 662, 924, 872, 1076, 812, 637, 751,
     680, 853])
ascad_rand_hw2_logit_tro0265_10000 = np.mean(
    [1179, 1074, 1034, 1288, 977, 753, 1655, 901, 1149, 1362, 1290, 682, 1064, 1154, 1718, 517, 1078, 776, 984, 875,
     889, 921, 821])
ascad_rand_hw2_logit_tro026_10000 = np.mean(
    [547, 1346, 1416, 1126, 1103, 1312, 1455, 829, 1016, 1607, 573, 810, 519, 636, 857, 721, 749, 1433, 1285, 1370,
     1649, 1017, 642, 980])
ascad_rand_hw2_logit_tro0255_10000 = np.mean(
    [339, 718, 2145, 525, 715, 680, 743, 976, 932, 931, 1148, 969, 974, 784, 1030, 1033, 522, 1490, 790, 995, 1064,
     1467, 1163, 866])
ascad_rand_hw2_logit_tro0245_10000 = np.mean(
    [1237, 1312, 1228, 1493, 1790, 656, 1238, 1056, 1168, 917, 1178, 775, 955, 585, 926, 725, 572, 991, 870, 703, 1372,
     1266, 1064])
ascad_rand_hw2_logit_tro024_10000 = np.mean(
    [1237, 1312, 1228, 1493, 1790, 656, 1238, 1056, 1168, 917, 1178, 775, 955, 585, 926, 725, 572, 991, 870, 703, 1372,
     1266, 1064])
ascad_rand_hw2_logit_tro0235_10000 = np.mean(
    [1045, 614, 898, 1268, 979, 872, 916, 905, 972, 634, 1441, 748, 981, 764, 528, 2283, 735, 1149, 946, 1258, 1102,
     932, 1077])
ascad_rand_hw2_logit_tro0225_10000 = np.mean(
    [1061, 1163, 1089, 1361, 1581, 1251, 1231, 1126, 918, 1066, 851, 1438, 825, 1501, 492, 1745, 2042, 1138, 1254,
     1651])
#
ascad_rand_hw2_logit_tro1_10000 = np.mean([3287, 2736, 3641, 3460, 2019])
ascad_rand_hw2_logit_tro1_GE = np.mean([1, 2, 3, 4, 3, 6, 0, 4, 1, 1, 0, 3, 6, 1, 1, 0, 0, 5, 1, ])
ascad_rand_hw2_logit_tro075_10000 = np.mean(
    [2358, 2807, 4787, 3776, 2792, 1617, 3412, 1659, 3360, 2952, 2944, 2150, 3039, 1825, 4069, 1948, 2971, 1329])
ascad_rand_hw2_logit_tro05_10000 = np.mean(
    [2400, 1555, 1089, 1169, 781, 1790, 871, 1463, 1452, 573, 908, 1166, 839, 1113, 960, 1259, 1001, 1151])
ascad_rand_hw2_logit_tro035_10000 = np.mean(
    [1714, 1017, 1673, 1393, 773, 1114, 616, 593, 1194, 1143, 735, 685, 1029, 1354, 778, 583, 855, 762])
ascad_rand_hw2_logit_tro0125_10000 = np.mean(
    [1094, 2141, 1805, 1606, 1690, 774, 1022, 1251, 1875, 922, 1328, 654, 1023, 1224, 1115, 1565, 733, 1987, 2124, 2099,
     1114, 846, 1472, 539])
ascad_rand_hw2_logit_tro02_10000 = np.mean(
    [969, 1474, 1092, 1737, 1242, 1051, 753, 987, 1160, 1080, 625, 1651, 1719, 1202, 1271, 1336, 891, 2186])

a_hw1_logit_tro05_10000 = np.array(
    [392 + 465 + 243 + 332 + 431 + 316 + 237 + 188 + 423 + 374 + 284 + 388 + 321 + 175 + 299 + 403 + 277 + 301])
a_hw1_logit_tro05_10000_mean = np.mean(a_hw1_logit_tro05_10000) / 18
a_hw1_logit_tro125_10000 = np.array([573 + 371 + 619 + 760 + 486 + 390 + 387 + 363 + 431])
a_hw1_logit_tro125_10000_mean = np.mean(a_hw1_logit_tro125_10000) / 9
a_hw1_logit_tro05_50000 = np.array(
    [225, 150, 150, 241, 122, 325, 144, 144, 217, 148, 164, 222, 342, 170, 173, 245, 213, 269, 128, 145, 88])
a_hw1_logit_tro05_50000_mean = np.mean(a_hw1_logit_tro05_50000)

a_hw1_logit_tro15_10000 = np.array([623, 522, 1310, 1049, 494, 1239, 606, 917, 755, 417, 886, 1110, 629])
a_hw1_logit_tro15_10000_mean = np.mean(a_hw1_logit_tro15_10000)
a_hw1_logit_tro15_50000 = np.array(
    [364, 538, 1103, 372, 668, 942, 474, 414, 533, 1033, 537, 631, 538, 544, 601, 303, 336, 629])
a_hw1_logit_tro15_50000_mean = np.mean(a_hw1_logit_tro15_50000)

a_hw1_logit_tro025_10000 = np.array([582, 651, 533, 552, 536, 1025])
a_hw1_logit_tro025_10000_mean = np.mean(a_hw1_logit_tro025_10000)
a_hw1_logit_tro025_50000 = np.array([361, 305, 533, 342, 509, 298, 476, 192, 726, 482, 309, 369, 410, 456])
a_hw1_logit_tro025_50000_mean = np.mean(a_hw1_logit_tro025_50000)

a_hw1_logit_tro0775_3000 = np.array(
    [])
a_hw1_logit_tro0775_3000_mean = np.mean(a_hw1_logit_tro0775_3000)

a_hw1_logit_tro0725_3000 = np.array(
    [393, 379, 660, 495, 428, 575, 402, 581, 552, 575, 704, 538, ])
a_hw1_logit_tro0725_3000_mean = np.mean(a_hw1_logit_tro0725_3000)

a_hw1_logit_tro0725_1000 = np.array(
    [2084, 2380, 2300, 2835, 1927, 2081, 1852, 2929, 1657, 2297, 1330, 5000, 3157, 1903, 1847, 2927, 2800, 2107])
a_hw1_logit_tro0725_1000_mean = np.mean(a_hw1_logit_tro0725_1000)

a_hw1_logit_tro075_1000 = np.array(
    [1515, 3483, 861, 2246, 1546, 1827, 2790, 2558, 858, 1725, 1555, 2361, 3486, 3229, 2504, 3097, 2690, 2706, 5000,
     1590, 2702, 4200])
a_hw1_logit_tro075_1000_mean = np.median(a_hw1_logit_tro075_1000)

a_hw1_logit_tro075_3000 = np.array(
    [467, 934, 422, 437, 626, 373, 515, 1294, 460, 605, 739, 388, 630, 830, 623, 483, 447, 576, 524, 752, 374])
a_hw1_logit_tro075_3000_mean = np.median(a_hw1_logit_tro075_3000)

a_hw1_logit_tro075_5000 = np.array(
    [225, 287, 423, 238, 183, 356, 586, 389, 292, 548, 242, 358, 314, 255, 214, 285, 264, 215, 433, 275, 397])
a_hw1_logit_tro075_5000_mean = np.mean(a_hw1_logit_tro075_5000)

a_hw1_logit_tro075_7000 = np.array(
    [213, 221, 257, 192, 254, 336, 292, 327, 283, 178, 219, 229, 315, 246, 302, 205, 168, 349, 158, 172, 150])
a_hw1_logit_tro075_7000_mean = np.mean(a_hw1_logit_tro075_7000)

a_hw1_logit_tro075_10000 = np.array(
    [285, 248, 118, 314, 222, 190, 232, 329, 239, 197, 256, 225, 190, 238, 180, 256, 192, 173, 159])
a_hw1_logit_tro075_10000_mean = np.mean(a_hw1_logit_tro075_10000)

a_hw1_logit_tro075_50000 = np.array(
    [210, 164, 151, 178, 147, 156, 164, 120, 179, 192, 150, 120, 171, 253, 226, 207, 153])
a_hw1_logit_tro075_50000_mean = np.mean(a_hw1_logit_tro075_50000)

a_hw1_logit_tro085_10000 = np.array(
    [288, 180, 322, 146, 304, 423, 188, 359, 306, 243, 150, 346, 346, 245, 198, 305, 257, 343, 210])
a_hw1_logit_tro085_10000_mean = np.mean(a_hw1_logit_tro085_10000)
a_hw1_logit_tro085_50000 = np.array(
    [116, 129, 231, 225, 160, 233, 141, 182, 168, 232, 195, 174, 209, 238, 219, 131, 205, 174, 219, 165, 350, 165, 171,
     126])
a_hw1_logit_tro085_50000_mean = np.mean(a_hw1_logit_tro085_50000)

c_hw2_logit_tro027_45000 = np.array(
    [867, 725, 392, 852, 291, 898, 518, 427, 316, 958, 571, 345, 469, 614, 841, 524, 636, 206, 602, 509])
c_hw2_logit_tro027_45000_mean = np.mean(c_hw2_logit_tro027_45000)

c_hw2_logit_tro026_45000 = np.array(
    [741, 366, 302, 631, 939, 573, 883, 744, 542, 1089, 641, 668, 720, 720, 615, 273, 1100, 524, 858, 860])
c_hw2_logit_tro026_45000_mean = np.mean(c_hw2_logit_tro026_45000)

c_hw2_logit_tro0255_45000 = np.array(
    [359, 709, 519, 460, 661, 651, 632, 556, 790, 564, 411, 737, 757, 615, 616, 670, 772, 637, 777, 866])
c_hw2_logit_tro0255_45000_mean = np.mean(c_hw2_logit_tro0255_45000)

c_hw2_logit_tro0245_45000 = np.array(
    [758, 461, 514, 1058, 908, 748, 772, 1327, 553, 406, 287, 562, 1032, 429, 469, 797, 533, 671, 1302, 789])
c_hw2_logit_tro0245_45000_mean = np.mean(c_hw2_logit_tro0245_45000)

c_hw2_logit_tro024_45000 = np.array(
    [452, 1148, 542, 609, 424, 693, 1117, 611, 1313, 323, 513, 932, 344, 546, 379, 677, 619, 579, 719, 676, 940])
c_hw2_logit_tro024_45000_mean = np.mean(c_hw2_logit_tro024_45000)

c_hw2_logit_tro0235_45000 = np.array(
    [769, 766, 236, 434, 462, 701, 545, 1024, 740, 676, 486, 844, 838, 768, 413, 325, 285, 489, 561, 1031, 567])  #还可以修改
c_hw2_logit_tro0235_45000_mean = np.mean(c_hw2_logit_tro0235_45000)

c_hw2_logit_tro023_45000 = np.array([917, 595, 863, 890, 959, 631, 1570, 761, 519, 654, 487])
c_hw2_logit_tro023_45000_mean = np.mean(c_hw2_logit_tro023_45000)

c_hw2_logit_tro0275_10000 = np.array(
    [3066, 1188, 2924, 1713, 2261, 1895, 1448, 5000, 3401, 1808, 3033, 1008, 1604, 1469, 2848, 2514, 3214])
c_hw2_logit_tro0275_10000_mean = np.mean(c_hw2_logit_tro0275_10000)

c_hw2_logit_tro027_10000 = np.array(
    [1153, 3723, 3802, 2249, 1700, 1062, 1347, 4649, 2007, 1958, 1215, 862, 1860, 3906, 1907, 1587, 1647])
c_hw2_logit_tro027_10000_mean = np.mean(c_hw2_logit_tro027_10000)

c_hw2_logit_tro0265_1000_ge = np.array([11, ])
c_hw2_logit_tro0265_1000_ge_mean = np.mean(c_hw2_logit_tro0265_1000_ge)

c_hw2_logit_tro0265_3000 = np.array(
    [2729, 3672, 2880, 3313, 3553, 4294, 3799, 2431, 4372, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_hw2_logit_tro0265_3000_mean = np.median(c_hw2_logit_tro0265_3000)
c_hw2_logit_tro0265_3000_ge = np.array([9, 3, 3, 7, 14, 2, 3, 7, 12])
c_hw2_logit_tro0265_3000_ge_mean = np.mean(c_hw2_logit_tro0265_3000_ge)

c_hw2_logit_tro0265_5000_new = np.array(
    [2203, 1883, 2884, 1154, 2100, 1921, 3882, 1939, 2489, 2976, 3386, 3308, 3519, 3543, 1727, 2798, 5000, 5000])
c_hw2_logit_tro0265_5000_ge_new = np.array([2, 6])
c_hw2_logit_tro0265_5000_mean_new = np.median(c_hw2_logit_tro0265_5000_new)
c_hw2_logit_tro0265_5000_ge_mean_new = np.mean(c_hw2_logit_tro0265_5000_ge_new)

c_hw2_logit_tro0265_7000_new = np.array(
    [1732, 2474, 1007, 1388, 1449, 1610, 1196, 2221, 1615, 1062, 1699, 1869, 1410, 1560, 1192, 1042, 2305, 1577])
c_hw2_logit_tro0265_7000_mean_new = np.mean(c_hw2_logit_tro0265_7000_new)

c_hw2_logit_tro0265_10000_new = np.array(
    [2082, 992, 2031, 1406, 1033, 463, 987, 1151, 1001, 1161, 2267, 1418, 1691, 1442, 1869, 1183, 1686, 1970, 1488,
     1749])
c_hw2_logit_tro0265_10000_mean_new = np.median(c_hw2_logit_tro0265_10000_new)

c_hw2_logit_tro0265_45000_new = np.array(
    [213, 269, 271, 366, 302, 286, 403, 557, 502, 489, 346, 226, 413])  # 还差五次
c_hw2_logit_tro0265_45000_mean_new = np.mean(c_hw2_logit_tro0265_45000_new)

c_hw2_logit_tro0265_5000 = np.array(
    [4192, 5000, 2990, 4318, 5000, 2317, 2021, 4272, 5000, 5000, 2711, 990, 5000, 2793, 4068, 5000, 5000, 4329, 3205,
     4688])
c_hw2_logit_tro0265_5000_ge = np.array([0, 1, 0, 0, 4, 0, 0, 0, 20, 1, 0, 0, 2, 0, 0, 2, 1, 0, 0, 0])
c_hw2_logit_tro0265_5000_mean = np.mean(c_hw2_logit_tro0265_5000)
c_hw2_logit_tro0265_5000_ge_mean = np.mean(c_hw2_logit_tro0265_5000_ge)

c_hw2_logit_tro0265_7000 = np.array(
    [1580, 1803, 5000, 1501, 3481, 1275, 1103, 2553, 2724, 1164, 918, 1074, 834, 903, 1819, 1179, 2273, 945, 2403,
     3024])
c_hw2_logit_tro0265_7000_mean = np.mean(c_hw2_logit_tro0265_7000)

c_hw2_logit_tro0265_10000 = np.array(
    [2191, 1126, 2349, 1369, 2441, 2874, 1671, 3297, 2353, 2649, 1542, 3736, 851, 1638, 956, 1198, 1465, 1939])
c_hw2_logit_tro0265_10000_mean = np.mean(c_hw2_logit_tro0265_10000)

c_hw2_logit_tro0265_45000 = np.array(
    [354, 417, 333, 350, 363, 238, 312, 376, 482, 312, 263, 383, 361, 185, 355, 237, 231, 254, 325, 322])
c_hw2_logit_tro0265_45000_mean = np.mean(c_hw2_logit_tro0265_45000)

c_hw2_logit_tro026_10000 = np.array(
    [5000, 5000, 1263, 1005, 1580, 2390, 2795, 2946, 1335, 2031, 2657, 985, 3432, 3728, 2866, 1824, 3058, 2888])
c_hw2_logit_tro026_10000_mean = np.mean(c_hw2_logit_tro026_10000)

c_hw2_logit_tro0255_10000 = np.array(
    [2955, 1536, 1719, 1394, 2811, 2809, 1123, 2783, 1044, 3130, 1093, 3773, 3444, 1799, 1590, 2619, 3389, 2003])
c_hw2_logit_tro0255_10000_mean = np.mean(c_hw2_logit_tro0255_10000)

c_hw2_logit_tro0245_10000 = np.array(
    [3294, 1911, 3341, 3762, 1841, 2428, 479, 744, 2079, 4109, 1421, 786, 2996, 2926, 2567, 2712, 1111, 3956, 1701])
c_hw2_logit_tro0245_10000_mean = np.mean(c_hw2_logit_tro0245_10000)

c_hw2_logit_tro024_10000 = np.array(
    [2177, 1495, 1208, 3126, 848, 1640, 1713, 1563, 4334, 1912, 2961, 2793, 3826, 3130, 3482, 1677, 1468, 5000, 2838,
     565, 3362])
c_hw2_logit_tro024_10000_mean = np.mean(c_hw2_logit_tro024_10000)

c_hw2_logit_tro0235_10000 = np.array(
    [2154, 1415, 3084, 2452, 2360, 2105, 3373, 3562, 1173, 1670, 1968, 1622, 3778, 1616, 1876, 1588, 2697, 577, 2595,
     1104, 2177, 1844])  #还可以修改
c_hw2_logit_tro0235_10000_mean = np.mean(c_hw2_logit_tro0235_10000)

c_hw2_logit_tro023_10000 = np.array(
    [2495, 1065, 1220, 1835, 3149, 1526, 1870, 717, 2285, 1944, 1962, 853, 1931, 1933, 3532, 2148, 4090, 941, 1935,
     4036, 1069])
c_hw2_logit_tro023_10000_mean = np.mean(c_hw2_logit_tro023_10000)

# new tro027
c_hw2_logit_tro027_3000 = np.array(
    [5000, 3437, 5000, 4894, 2545, 5000, 3433, 4547, 5000, 5000, 4058, 5000, 3394, 2036, 5000, 4582, 5000, 4486])
c_hw2_logit_tro027_3000_mean = np.mean(c_hw2_logit_tro027_3000)
c_hw2_logit_tro027_3000_ge = np.array([3, 21, 1, 2, 3, 8, 5, 2, ])
c_hw2_logit_tro027_3000_ge_mean = np.mean(c_hw2_logit_tro027_3000_ge)

c_hw2_logit_tro027_5000_new = np.array(
    [2046, 2515, 2876, 2957, 1791, 1550, 3205, 1175, 1836, 4155, 4348, 3946, 2507, 3747, 2651, 5000, 5000])
c_hw2_logit_tro027_5000_ge_new = np.array([1, 13, 21])
c_hw2_logit_tro027_5000_mean_new = np.mean(c_hw2_logit_tro027_5000_new)
c_hw2_logit_tro027_5000_ge_mean_new = np.mean(c_hw2_logit_tro027_5000_ge_new)

c_hw2_logit_tro027_7000_new = np.array(
    [])
c_hw2_logit_tro027_7000_mean_new = np.mean(c_hw2_logit_tro027_7000_new)

c_hw2_logit_tro027_10000_new = np.array(
    [])
c_hw2_logit_tro027_10000_mean_new = np.mean(c_hw2_logit_tro027_10000_new)

c_hw2_logit_tro027_45000_new = np.array(
    [])
c_hw2_logit_tro027_45000_mean_new = np.mean(c_hw2_logit_tro027_45000_new)

# class balance focal

"ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad ascad"

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

a_c_f_hw1_gamma2_1000 = np.array(
    [])
a_c_f_hw1_gamma2_1000_mean = np.mean(a_c_f_hw1_gamma2_1000)

a_c_f_hw1_gamma2_3000 = np.array(
    [499, 453, 476, 810, 530, 899, 925, 661, 1546, 634, 667, 826, 567, 439, 538, 1083, 414, 406])
a_c_f_hw1_gamma2_3000_mean = np.mean(a_c_f_hw1_gamma2_3000)

a_c_f_hw1_gamma2_5000 = np.array(
    [414, 901, 494, 334, 510, 545, 636, 607, 614, 396, 513, 477, 363, 494, 422, 579, 414, 363])
a_c_f_hw1_gamma2_5000_mean = np.mean(a_c_f_hw1_gamma2_5000)

a_c_f_hw1_gamma2_7000 = np.array(
    [371, 562, 524, 462, 426, 332, 652, 572, 447, 477, 577, 415, 801, 790, 351, 319, 539, 363])
a_c_f_hw1_gamma2_7000_mean = np.mean(a_c_f_hw1_gamma2_7000)

a_c_f_hw1_gamma2_10000_new = np.array(
    [846, 567, 418, 626, 492, 521, 812, 715, 307, 414, 620, 494, 732, 510, 373, 745, 962, 407])
a_c_f_hw1_gamma2_10000_new_mean = np.mean(a_c_f_hw1_gamma2_10000_new)

a_c_f_hw1_gamma2_50000_new = np.array(
    [])
a_c_f_hw1_gamma2_50000_new_mean = np.mean(a_c_f_hw1_gamma2_50000_new)

# 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105

a_c_f_hw1_gamma105_1000 = np.array(
    [])
a_c_f_hw1_gamma105_1000_mean = np.mean(a_c_f_hw1_gamma105_1000)

a_c_f_hw1_gamma105_3000 = np.array(
    [885, 648, 332, 395, 617, 804, 482, 600, 519, 404, 697, 1218, 253, 1419, 997, 587, 725, 488])
a_c_f_hw1_gamma105_3000_mean = np.mean(a_c_f_hw1_gamma105_3000)

a_c_f_hw1_gamma105_5000 = np.array(
    [563, 575, 436, 337, 710, 449, 380, 350, 519, 430, 365, 528, 546, 645, 386, 399, 587, 652])
a_c_f_hw1_gamma105_5000_mean = np.mean(a_c_f_hw1_gamma105_5000)

a_c_f_hw1_gamma105_7000 = np.array(
    [551, 506, 400, 524, 403, 372, 528, 322, 296, 460, 478, 429, 553, 345, 525, 588, 404, 285])
a_c_f_hw1_gamma105_7000_mean = np.mean(a_c_f_hw1_gamma105_7000)

a_c_f_hw1_gamma105_10000_new = np.array(
    [667, 522, 745, 699, 309, 475, 354, 812, 363, 361, 578, 382, 656, 483, 482, 443, 381, 552])
a_c_f_hw1_gamma105_10000_new_mean = np.mean(a_c_f_hw1_gamma105_10000_new)

a_c_f_hw1_gamma105_50000_new = np.array(
    [])
a_c_f_hw1_gamma105_50000_new_mean = np.mean(a_c_f_hw1_gamma105_50000_new)

# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

a_c_f_hw1_gamma1_1000 = np.array(
    [2616, 2772, 1346, 1773, 2706, 3572, 2262, 3136, 3242, 3701, 2842, 2053, 2118, 2914, 3304, 2785, 3033, 5000])
a_c_f_hw1_gamma1_1000_mean = np.mean(a_c_f_hw1_gamma1_1000)

a_c_f_hw1_gamma1_1000_ge = np.array(
    [1, ])
a_c_f_hw1_gamma1_1000_ge_mean = np.mean(a_c_f_hw1_gamma1_1000_ge)

a_c_f_hw1_gamma1_3000 = np.array(
    [1277, 621, 538, 595, 626, 1338, 370, 756, 614, 600, 670, 518])  # 再重新跑六次
a_c_f_hw1_gamma1_3000_mean = np.mean(a_c_f_hw1_gamma1_3000)

a_c_f_hw1_gamma1_5000 = np.array(
    [693, 503, 878, 725, 413, 525, 310, 614, 436, 436, 301, 403, 393, 237, 458, 433, 486])  # 再重新跑一次
a_c_f_hw1_gamma1_5000_mean = np.mean(a_c_f_hw1_gamma1_5000)

a_c_f_hw1_gamma1_7000 = np.array(
    [545, 556, 485, 398, 335, 426, 424, 499, 467, 452, 376, 511, 582, 448, 222, 289, 455, 544])
a_c_f_hw1_gamma1_7000_mean = np.mean(a_c_f_hw1_gamma1_7000)

a_c_f_hw1_gamma1_10000_new = np.array(
    [422, 421, 557, 521, 536, 225, 411, 464, 413, 661, 760, 387, 564, 651, 373, 388, 580, 809])
a_c_f_hw1_gamma1_10000_new_mean = np.mean(a_c_f_hw1_gamma1_10000_new)

a_c_f_hw1_gamma1_50000_new = np.array(
    [])
a_c_f_hw1_gamma1_50000_new_mean = np.mean(a_c_f_hw1_gamma1_50000_new)

# 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05

a_c_f_hw1_gamma05_1000 = np.array(
    [2576, 1569, 2059, 2962, 1959, 3141, 1623, 2391, 1651, 2367, 1218, 3188, 5000, 3670, 2624, 2229, 3148, 2021, 2358,
     776])
a_c_f_hw1_gamma05_1000_mean = np.median(a_c_f_hw1_gamma05_1000)

a_c_f_hw1_gamma045_1000 = np.array(
    [1091, 1680, 3480, 1943, 3602, 2763, 1815, 2609, 1963, 2049, 4049, 2479, 2258, 2447, 1390, 2294, 3554, 1921, 3218,
     1958])
a_c_f_hw1_gamma045_1000_mean = np.mean(a_c_f_hw1_gamma045_1000)

a_c_f_hw1_gamma05_3000 = np.array(
    [365, 523, 478, 769, 413, 629, 407, 479, 381, 712, 895, 550, 677, 384, 483, 523, 478, 769, 413, 629, 407])
a_c_f_hw1_gamma05_3000_mean = np.median(a_c_f_hw1_gamma05_3000)

a_c_f_hw1_gamma05_5000 = np.array(
    [410, 504, 300, 347, 448, 374, 427, 584, 300, 472, 239, 382, 246, 465, 286, 434, 484, 419, 607, 561, 375])
a_c_f_hw1_gamma05_5000_mean = np.mean(a_c_f_hw1_gamma05_5000)

a_c_f_hw1_gamma05_7000 = np.array(
    [339, 651, 460, 651, 460, 225, 244, 276, 312, 357, 405, 310, 481, 365, 344, 305, 545, 733, 459, 228, 262])
a_c_f_hw1_gamma05_7000_mean = np.mean(a_c_f_hw1_gamma05_7000)

a_c_f_hw1_gamma05_10000_new = np.array(
    [355, 511, 580, 321, 562, 803, 414, 331, 459, 383, 354, 729, 458, 706, 375, 421, 398, 585, 433, 323, 597])
a_c_f_hw1_gamma05_10000_new_mean = np.mean(a_c_f_hw1_gamma05_10000_new)

a_c_f_hw1_gamma045_10000_new = np.array(
    [507, 451, 293, 230, 398, 379, 513, 430, 585, 566, 276, 258, 686, 355, 388, 495, 274, 515, 197, 344, 388])
a_c_f_hw1_gamma045_10000_new_mean = np.mean(a_c_f_hw1_gamma045_10000_new)

a_c_f_hw1_gamma05_50000_new = np.array(
    [1234, 740, 848, 984, 1027, 632, 605, 561, 948, 645, 776, 505, 515, 956, 690, 533, 541, 502, 577, 698, 599])
a_c_f_hw1_gamma05_50000_new_mean = np.mean(a_c_f_hw1_gamma05_50000_new)

# 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025

a_c_f_hw1_gamma025_1000 = np.array(
    [3566, 1721, 3630, 2455, 3830, 2421, 2165, 1978, 2416, 3431, 2691, 2391, 1775, 1919, 3564, 2352])
a_c_f_hw1_gamma025_1000_mean = np.mean(a_c_f_hw1_gamma025_1000)

a_c_f_hw1_gamma025_1000_ge = np.array(
    [3, 4, ])
a_c_f_hw1_gamma025_1000_ge_mean = np.mean(a_c_f_hw1_gamma025_1000_ge)

a_c_f_hw1_gamma025_3000 = np.array(
    [497, 529, 403, 632, 688, 488, 523, 585, 1608, 823, 520, 487, 577, 477, 697, 544, 414, 446])
a_c_f_hw1_gamma025_3000_mean = np.mean(a_c_f_hw1_gamma025_3000)

a_c_f_hw1_gamma025_5000 = np.array(
    [441, 364, 536, 442, 295, 565, 510, 429, 340, 585, 583, 260, 359, 402, 475, 395, 370, 649])
a_c_f_hw1_gamma025_5000_mean = np.mean(a_c_f_hw1_gamma025_5000)

a_c_f_hw1_gamma025_7000 = np.array(
    [312, 301, 461, 285, 350, 704, 579, 557, 253, 364, 299, 536, 452, 408, 343, 371, 532, 456])
a_c_f_hw1_gamma025_7000_mean = np.mean(a_c_f_hw1_gamma025_7000)

a_c_f_hw1_gamma025_10000_new = np.array(
    [359, 345, 650, 535, 707, 665, 394, 461, 986, 409, 452, 431, 606, 419, 355, 456, 558, 362])
a_c_f_hw1_gamma025_10000_new_mean = np.mean(a_c_f_hw1_gamma025_10000_new)

a_c_f_hw1_gamma025_50000_new = np.array(
    [])
a_c_f_hw1_gamma025_50000_new_mean = np.mean(a_c_f_hw1_gamma025_50000_new)

"ascad_rand"

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

ar_c_f_hw2_gamma2_1000 = np.array(
    [4806])
ar_c_f_hw2_gamma2_1000_mean = np.mean(ar_c_f_hw2_gamma2_1000)

ar_c_f_hw2_gamma2_1000_ge = np.array(
    [30, 30, 22, 4, 55, 72, 2, 60, 4, 1, 3, 26, 62, 31, 45, 1, 11])
ar_c_f_hw2_gamma2_1000_mean_ge = np.mean(ar_c_f_hw2_gamma2_1000_ge)

ar_c_f_hw2_gamma2_3000 = np.array(
    [2802, 3268, 2060, 2931, 3735, 4514, 4913, 2769, 2306, 3160, 4297, 3035, 2194, 3646, 2708, 4029, 5000, 5000])
ar_c_f_hw2_gamma2_3000_mean = np.mean(ar_c_f_hw2_gamma2_3000)

ar_c_f_hw2_gamma2_3000_ge = np.array(
    [3, 1, ])
ar_c_f_hw2_gamma2_3000_mean_ge = np.mean(ar_c_f_hw2_gamma2_3000_ge)

ar_c_f_hw2_gamma2_5000 = np.array(
    [])
ar_c_f_hw2_gamma2_5000_mean = np.mean(ar_c_f_hw2_gamma2_5000)

ar_c_f_hw2_gamma2_7000 = np.array(
    [1402, 2158, 1904, 1581, 1400, 1081, 2729, 3125, 1063, 1725, 2589, 941, 951, 1319, 1442, 1640, 1037, 1016, ])
ar_c_f_hw2_gamma2_7000_mean = np.mean(ar_c_f_hw2_gamma2_7000)

ar_c_f_hw2_gamma2_10000_new = np.array(
    [851, 1203, 1453, 1602, 2338, 2750, 793, 1106, 1973, 2082, 1548, 2194, 1777, 1279, 777, 1154, 1790, 1391, ])
ar_c_f_hw2_gamma2_10000_new_mean = np.mean(ar_c_f_hw2_gamma2_10000_new)

ar_c_f_hw2_gamma2_50000_new = np.array(
    [])
ar_c_f_hw2_gamma2_50000_new_mean = np.mean(ar_c_f_hw2_gamma2_50000_new)

# 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105

ar_c_f_hw2_gamma105_1000 = np.array(
    [])
ar_c_f_hw2_gamma105_1000_mean = np.mean(ar_c_f_hw2_gamma105_1000)

ar_c_f_hw2_gamma105_1000_ge = np.array(
    [5, 3, 36, 40, 7, 11, 90, 26, 29, 2, 119, 14, 24, 33, 43, 5, 4, 1])
ar_c_f_hw2_gamma105_1000_mean_ge = np.mean(ar_c_f_hw2_gamma105_1000_ge)

ar_c_f_hw2_gamma105_3000 = np.array(
    [2482, 3184, 2470, 2661, 2131, 3967, 2291, 2790, 3250, 2580, 3988, 2769, 3166, 2923, 2989, 3703, 4525, 5000])
ar_c_f_hw2_gamma105_3000_mean = np.mean(ar_c_f_hw2_gamma105_3000)

ar_c_f_hw2_gamma105_3000_ge = np.array(
    [1])
ar_c_f_hw2_gamma105_3000_mean_ge = np.mean(ar_c_f_hw2_gamma105_3000_ge)

ar_c_f_hw2_gamma105_5000 = np.array(
    [])
ar_c_f_hw2_gamma105_5000_mean = np.mean(ar_c_f_hw2_gamma105_5000)

ar_c_f_hw2_gamma105_7000 = np.array(
    [1350, 1362, 2559, 1359, 1598, 1577, 1400, 2521, 1406, 1734, 1269, 1234, 2061, 1298, 1975, 818, 921, 2081])
ar_c_f_hw2_gamma105_7000_mean = np.mean(ar_c_f_hw2_gamma105_7000)

ar_c_f_hw2_gamma105_10000_new = np.array(
    [996, 1308, 1622, 1864, 1112, 1212, 1409, 1351, 2404, 1895, 1423, 1586, 891, 1840, 1902, 1085, 1948, 1396])
ar_c_f_hw2_gamma105_10000_new_mean = np.mean(ar_c_f_hw2_gamma105_10000_new)

ar_c_f_hw2_gamma105_50000_new = np.array(
    [])
ar_c_f_hw2_gamma105_50000_new_mean = np.mean(ar_c_f_hw2_gamma105_50000_new)

# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

ar_c_f_hw2_gamma1_1000 = np.array(
    [])
ar_c_f_hw2_gamma1_1000_mean = np.mean(ar_c_f_hw2_gamma1_1000)

ar_c_f_hw2_gamma1_1000_ge = np.array(
    [])
ar_c_f_hw2_gamma1_1000_mean_ge = np.mean(ar_c_f_hw2_gamma1_1000_ge)  # 再重新跑6次

ar_c_f_hw2_gamma1_3000 = np.array(
    [2176, 2181, 2627, 2086, 3409, 2150, 3605, 3061, 3355, 4407, 2927, 2388, 3813, 5000, 5000, 5000, 5000])
ar_c_f_hw2_gamma1_3000_mean = np.mean(ar_c_f_hw2_gamma1_3000)

ar_c_f_hw2_gamma1_3000_ge = np.array(
    [1, 1, 1, 1])
ar_c_f_hw2_gamma1_3000_mean_ge = np.mean(ar_c_f_hw2_gamma1_3000_ge)

ar_c_f_hw2_gamma1_5000 = np.array(
    [2383, 2039, 2031, 1557, 1800, 1112, 2411, 2352, 1648, 3227, 1068, 2625])  # 再重新跑6次
ar_c_f_hw2_gamma1_5000_mean = np.mean(ar_c_f_hw2_gamma1_5000)

ar_c_f_hw2_gamma1_7000 = np.array(
    [1454, 1706, 1395, 1965, 1982, 1697, 2429, 1048, 2381, 1492, 1148, 1398, 1444, 1067, 1438, 2006, 1483, 1304])
ar_c_f_hw2_gamma1_7000_mean = np.mean(ar_c_f_hw2_gamma1_7000)

ar_c_f_hw2_gamma1_10000_new = np.array(
    [869, 1175, 1184, 1244, 1698, 1451, 947, 1438, 1787, 3360, 1201, 1149, 1057, 1040, 842, 769, 1473, 1085])
ar_c_f_hw2_gamma1_10000_new_mean = np.mean(ar_c_f_hw2_gamma1_10000_new)

ar_c_f_hw2_gamma1_50000_new = np.array(
    [435, 688, 638, 1110, 793, 908, 788, 444, 1014, 688])  # 再重新8次
ar_c_f_hw2_gamma1_50000_new_mean = np.mean(ar_c_f_hw2_gamma1_50000_new)

# 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05

ar_c_f_hw2_gamma05_1000_ge = np.array([12, 69, 4, 118, 4, 31, 86, 136, 36, 19, 21, 12, 192, 0, 13, 6, 3, 44, 62])
ar_c_f_hw2_gamma05_1000_ge_mean = np.mean(ar_c_f_hw2_gamma05_1000_ge)

ar_c_f_hw2_gamma05_3000 = np.array(
    [3589, 1941, 4299, 2988, 3024, 1487, 5000, 2819, 5000, 2337, 3681, 2592, 1885, 2718, 1533, 5000, 2621, 5000, 2325,
     3735])
ar_c_f_hw2_gamma05_3000_mean = np.median(ar_c_f_hw2_gamma05_3000)

ar_c_f_hw2_gamma05_5000 = np.array(
    [1511, 1750, 1542, 1766, 1536, 985, 852, 2090, 1553, 2014, 1680, 1148, 3621, 1579, 2024, 2603, 1604, 2732, 1577,
     2461])
ar_c_f_hw2_gamma05_5000_mean = np.mean(ar_c_f_hw2_gamma05_5000)

ar_c_f_hw2_gamma05_7000 = np.array(
    [1540, 2008, 1312, 1562, 1063, 1737, 1451, 1861, 1271, 1051, 1280, 2106, 1045, 1411, 1916, 1422, 1408, 969, 1391,
     1024])
ar_c_f_hw2_gamma05_7000_mean = np.mean(ar_c_f_hw2_gamma05_7000)

ar_c_f_hw2_gamma05_10000 = np.array([1260, 1416, 599, 837, 766, 627, 780, 1212, 799, 1052, 652, 2492, 1805, 925, 1362])
ar_c_f_hw2_gamma05_10000_mean = np.mean(ar_c_f_hw2_gamma05_10000)

ar_c_f_hw3_gamma05_10000 = np.array([2123, 1232, 1454, 1006, 917, 1652, 872, 971, 847, ])
ar_c_f_hw3_gamma05_10000_mean = np.mean(ar_c_f_hw3_gamma05_10000)

ar_c_f_hw1_gamma05_50000 = np.array(
    [702, 497, 639, 574, 672, 589, 709, 702, 897, 692, 575, 605, 357, 602, 794, 624, 552])
ar_c_f_hw1_gamma05_50000_mean = np.mean(ar_c_f_hw1_gamma05_50000)

# 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025

ar_c_f_hw2_gamma025_1000 = np.array(
    [])
ar_c_f_hw2_gamma025_1000_mean = np.mean(ar_c_f_hw2_gamma025_1000)

ar_c_f_hw2_gamma025_1000_ge = np.array(
    [5, 3, 36, 40, 7, 11, 90, 26, 29, 2, 119, 14, 24, 33, 43, 5, 4, 1])
ar_c_f_hw2_gamma025_1000_mean_ge = np.mean(ar_c_f_hw2_gamma025_1000_ge)

ar_c_f_hw2_gamma025_3000 = np.array(
    [2567, 2678, 2660, 5000, 3667, 3756, 3389, 4380, 3395, 2499, 3717, 2865, 4504, 5000, 4295, 2723, 2446, 5000])
ar_c_f_hw2_gamma025_3000_mean = np.mean(ar_c_f_hw2_gamma025_3000)

ar_c_f_hw2_gamma025_3000_ge = np.array(
    [6, 3, 2])
ar_c_f_hw2_gamma025_3000_mean_ge = np.mean(ar_c_f_hw2_gamma025_3000_ge)

ar_c_f_hw2_gamma025_5000 = np.array(
    [1350, 2900, 1872, 1009, 1900, 3203, 2119, 2291, 1676, 2313, 1492, 1698])  # 还差6次
ar_c_f_hw2_gamma025_5000_mean = np.mean(ar_c_f_hw2_gamma025_5000)

ar_c_f_hw2_gamma025_7000 = np.array(
    [1007, 1182, 1106, 1150, 1172, 1344, 1534, 1364, 1677, 4447, 1665, 1011, 1799, 1398, 2666, 1979, 919, 1235])
ar_c_f_hw2_gamma025_7000_mean = np.mean(ar_c_f_hw2_gamma025_7000)

ar_c_f_hw2_gamma025_10000_new = np.array(
    [1845, 2415, 1015, 1548, 1635, 1390, 1206, 804, 1263, 940, 733, 1685, 1014, 981, 1410, 1815, 1555, 1222])
ar_c_f_hw2_gamma025_10000_new_mean = np.mean(ar_c_f_hw2_gamma025_10000_new)

ar_c_f_hw2_gamma025_50000_new = np.array(
    [514, 877, 786, 1210, 1070, 1177, 783, 788, 1219])  # 还差12次
ar_c_f_hw2_gamma025_50000_new_mean = np.mean(ar_c_f_hw2_gamma025_50000_new)

"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

c_c_f_hw2_gamma2_45000 = np.array(
    [944, 1110, 610, 529, 740, 665, 733, 984, 721, 820])
c_c_f_hw2_gamma2_45000_mean = np.mean(c_c_f_hw2_gamma2_45000)

c_c_f_hw2_gamma2_10000 = np.array(
    [3206, 1277, 1096, 1242, 1510, 1222, 1325, 2193, 1514, 1258, 977, 1494, 1793, 3497, 1220, 2279, 1816, 5000])
c_c_f_hw2_gamma2_10000_mean = np.mean(c_c_f_hw2_gamma2_10000)

c_c_f_hw2_gamma2_7000 = np.array(
    [2379, 3824, 1658, 976, 1774, 1255, 1655, 2536, 954, 3203, 2638, 2516, 1305, 2683, 3023, 3367, 1476, 1140, 2670,
     1953])
c_c_f_hw2_gamma2_7000_mean = np.mean(c_c_f_hw2_gamma2_7000)

c_c_f_hw2_gamma2_5000 = np.array(
    [1655, 2271, 2970, 4012, 1528])
c_c_f_hw2_gamma2_5000_mean = np.mean(c_c_f_hw2_gamma2_5000)

c_c_f_hw2_gamma2_5000_ge = np.array([1])
c_c_f_hw2_gamma2_5000_ge_mean = np.mean(c_c_f_hw2_gamma2_5000_ge)

c_c_f_hw2_gamma2_3000 = np.array(
    [3119, 3265, 4459, 4159, 4833, 4840, 4780, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma2_3000_mean = np.mean(c_c_f_hw2_gamma2_3000)

c_c_f_hw2_gamma2_3000_ge = np.array([1, 6, 6, 8, 1, 3, 2, 4, 4, 1, 11, 0, 0, 0, 0, 0, 0, 0, ])
c_c_f_hw2_gamma2_3000_ge_mean = np.mean(c_c_f_hw2_gamma2_3000_ge)

# 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105

c_c_f_hw2_gamma105_45000 = np.array(
    [1360, 789, 753, 644, 1096, 526, 461, 807, 653, 634, 1030, 748, 1175, 567])
c_c_f_hw2_gamma105_45000_mean = np.mean(c_c_f_hw2_gamma105_45000)

c_c_f_hw2_gamma105_10000 = np.array(
    [2021, 1775, 1640, 1956, 1964, 1517, 1281, 3521, 1071, 581, 2304, 1876, 1723, 1692, 2458, 1654, 1512, 2448])
c_c_f_hw2_gamma105_10000_mean = np.mean(c_c_f_hw2_gamma105_10000)

c_c_f_hw2_gamma105_7000 = np.array(
    [1982, 1562, 3150, 1979, 2932, 1939, 1940, 1685, 1701, 1776, 1718, 1219, 1507, 1025, 1546, 2435, 1340, 1334])
c_c_f_hw2_gamma105_7000_mean = np.mean(c_c_f_hw2_gamma105_7000)

c_c_f_hw2_gamma105_5000 = np.array(
    [2423, 3022, 2008, 2745, 1319, 2562, 2931, 1559, 3002, 2120, 3559, 2415, 3489, 3518, 2399, 5000, 5000, 5000])
c_c_f_hw2_gamma105_5000_mean = np.mean(c_c_f_hw2_gamma105_5000)

c_c_f_hw2_gamma105_5000_ge = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
c_c_f_hw2_gamma105_5000_ge_mean = np.mean(c_c_f_hw2_gamma105_5000_ge)

c_c_f_hw2_gamma105_3000 = np.array(
    [2327, 3005, 3075, 3924, 3623, 4511, 2683, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma105_3000_mean = np.mean(c_c_f_hw2_gamma105_3000)

c_c_f_hw2_gamma105_3000_ge = np.array([25, 2, 3, 3, 6, 2, 1, 11, 11, 5, 3, 0, 0, 0, 0, 0, 0, 0])
c_c_f_hw2_gamma105_3000_ge_mean = np.mean(c_c_f_hw2_gamma105_3000_ge)

# 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025

c_c_f_hw2_gamma1025_45000 = np.array(
    [907, 1158, 719, ])
c_c_f_hw2_gamma1025_45000_mean = np.mean(c_c_f_hw2_gamma1025_45000)

c_c_f_hw2_gamma1025_10000 = np.array(
    [922, 1563, 1267, 1671, 1265, 1502, 2038, 2330, 2741, 1211, 2091, 3070, 1278, 2572, 1294, 1587, 1158, 932, 1150])
c_c_f_hw2_gamma1025_10000_mean = np.mean(c_c_f_hw2_gamma1025_10000)

c_c_f_hw2_gamma1025_7000 = np.array(
    [2310, 2915, 1173, 2521, 1372, 3695, 1667, 2306, 2087, 1405, 1000, 1094, 1312, 2078, 1687, 1637, 1319, 1530])
c_c_f_hw2_gamma1025_7000_mean = np.mean(c_c_f_hw2_gamma1025_7000)

c_c_f_hw2_gamma1025_5000 = np.array(
    [3566, 3329, 2091, 3383, 2355, 1349, 4597, 2366, 1719, 2304, 3154, 1806, 2028, 1425, 4596, 2596, 5000, 5000])
c_c_f_hw2_gamma1025_5000_mean = np.mean(c_c_f_hw2_gamma1025_5000)

c_c_f_hw2_gamma1025_5000_ge = np.array([1, 4, ])
c_c_f_hw2_gamma1025_5000_ge_mean = np.mean(c_c_f_hw2_gamma1025_5000_ge)

c_c_f_hw2_gamma1025_3000 = np.array(
    [4509, 3809, 3324, 4757, 4333, 3727, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma1025_3000_mean = np.mean(c_c_f_hw2_gamma1025_3000)

c_c_f_hw2_gamma1025_3000_ge = np.array([1, 35, 4, 1, 8, 1, 7, 14, 1, 11, 1, 82, 0, 0, 0, 0, 0, 0, ])
c_c_f_hw2_gamma1025_3000_ge_mean = np.mean(c_c_f_hw2_gamma1025_3000_ge)

# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

c_c_f_hw2_gamma1_45000 = np.array(
    [])
c_c_f_hw2_gamma1_45000_mean = np.mean(c_c_f_hw2_gamma1_45000)

c_c_f_hw2_gamma1_10000 = np.array(
    [2515, 1396, 1339, 1637, 654, 2313, 1844, 1505, 1356, 853, 932, 1076, 947, 1142, 1418, 2615, 1150])
c_c_f_hw2_gamma1_10000_mean = np.mean(c_c_f_hw2_gamma1_10000)

c_c_f_hw2_gamma1_7000 = np.array(
    [2078, 1069, 1747, 2361, 1066, 1110, 1297, 968, 2851, 2846, 1870, 1251, 1242, 1136, 2104, 1580, 4442, 5000])
c_c_f_hw2_gamma1_7000_mean = np.mean(c_c_f_hw2_gamma1_7000)

c_c_f_hw2_gamma1_7000_ge = np.array([2, ])
c_c_f_hw2_gamma1_7000_ge_mean = np.mean(c_c_f_hw2_gamma1_7000_ge)

c_c_f_hw2_gamma1_5000 = np.array(
    [1977, 4178, 4274, 2003, 1891, 3482, 2047, 1900, 2611, 3102, 1922, 1510, 3141, 2411, 1882, 3666, 5000, 5000])
c_c_f_hw2_gamma1_5000_mean = np.mean(c_c_f_hw2_gamma1_5000)

c_c_f_hw2_gamma1_5000_ge = np.array([1, 1])
c_c_f_hw2_gamma1_5000_ge_mean = np.mean(c_c_f_hw2_gamma1_5000_ge)

c_c_f_hw2_gamma1_3000 = np.array(
    [4155, 4971, 2383, 3327, 2809, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma1_3000_mean = np.mean(c_c_f_hw2_gamma1_3000)

c_c_f_hw2_gamma1_3000_ge = np.array([12, 39, 13, 2, 15, 14, 2, 1, 4, 1, 3, 0, 0, 0, 0, 0, 7])
c_c_f_hw2_gamma1_3000_ge_mean = np.mean(c_c_f_hw2_gamma1_3000_ge)

# 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075 075

c_c_f_hw2_gamma075_45000 = np.array(
    [])
c_c_f_hw2_gamma075_45000_mean = np.mean(c_c_f_hw2_gamma075_45000)

c_c_f_hw2_gamma075_10000 = np.array(
    [])
c_c_f_hw2_gamma075_10000_mean = np.mean(c_c_f_hw2_gamma075_10000)

c_c_f_hw2_gamma075_7000 = np.array(
    [])
c_c_f_hw2_gamma075_7000_mean = np.mean(c_c_f_hw2_gamma075_7000)

c_c_f_hw2_gamma075_5000 = np.array(
    [])
c_c_f_hw2_gamma075_5000_mean = np.mean(c_c_f_hw2_gamma075_5000)

c_c_f_hw2_gamma075_5000_ge = np.array([])
c_c_f_hw2_gamma075_5000_ge_mean = np.mean(c_c_f_hw2_gamma075_5000_ge)

# 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05

c_c_f_hw2_gamma05_45000 = np.array(
    [])
c_c_f_hw2_gamma05_45000_mean = np.mean(c_c_f_hw2_gamma05_45000)

c_c_f_hw2_gamma05_10000 = np.array(
    [3252, 1453, 1873, 1421, 1310, 1670, 1525, 2110, 680, 1177, 1276, 1484, 1223, 1359, 3744, 1323, 1341, 1943])
c_c_f_hw2_gamma05_10000_mean = np.mean(c_c_f_hw2_gamma05_10000)

c_c_f_hw2_gamma05_7000 = np.array(
    [1689, 1410, 1601, 2227, 1994, 2467, 3944, 2004, 1351, 1018, 1468, 1432, 2006, 1964, 2830, 2433, 2675, 2892])
c_c_f_hw2_gamma05_7000_mean = np.mean(c_c_f_hw2_gamma05_7000)

c_c_f_hw2_gamma05_5000 = np.array(
    [1486, 2344, 3693, 2323, 2376, 3568, 2542, 1600, 3899, 2863, 3230, 1883, 1368, 2518, 3938, 2651, 5000, 5000])
c_c_f_hw2_gamma05_5000_mean = np.median(c_c_f_hw2_gamma05_5000)

c_c_f_hw2_gamma05_5000_ge = np.array([2, 1])
c_c_f_hw2_gamma05_5000_ge_mean = np.mean(c_c_f_hw2_gamma05_5000_ge)

c_c_f_hw2_gamma05_3000 = np.array(
    [2590, 3605, 4187, 3030, 3255, 3424, 3398, 3918, 3766, 3331, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma05_3000_mean = np.median(c_c_f_hw2_gamma05_3000)

c_c_f_hw2_gamma05_3000_ge = np.array([8, 12, 8, 2, 1, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0])
c_c_f_hw2_gamma05_3000_ge_mean = np.mean(c_c_f_hw2_gamma05_3000_ge)

# 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025

c_c_f_hw2_gamma025_45000 = np.array(
    [969, 717, 1063, 901, 649, 1658, 1001, 574, 783, 511, 651, 1347, 1078, 1044, 750, 650, 1420])
c_c_f_hw2_gamma025_45000_mean = np.mean(c_c_f_hw2_gamma025_45000)

c_c_f_hw2_gamma025_10000 = np.array(
    [2019, 1588, 1314, 908, 796, 1645, 1614, 1438, 1199, 986, 2732, 919, 1779, 1450, 1630, 1223, 1442, 2024])
c_c_f_hw2_gamma025_10000_mean = np.mean(c_c_f_hw2_gamma025_10000)

c_c_f_hw2_gamma025_7000 = np.array(
    [1946, 3545, 2610, 1706, 2027, 1308, 3793, 1177, 1038, 1650, 3495, 2183, 2874, 874, 2378, 2686, 1691, 2075])
c_c_f_hw2_gamma025_7000_mean = np.mean(c_c_f_hw2_gamma025_7000)

c_c_f_hw2_gamma025_5000 = np.array(
    [3306, 2806, 1855, 3780, 2619, 2605, 2808, 3115, 1649, 1926, 2261, 2513, 1776, 3054, 1616, 3460, 5000, 5000])
c_c_f_hw2_gamma025_5000_mean = np.mean(c_c_f_hw2_gamma025_5000)

c_c_f_hw2_gamma025_5000_ge = np.array([2, 1])
c_c_f_hw2_gamma025_5000_ge_mean = np.mean(c_c_f_hw2_gamma025_5000_ge)

c_c_f_hw2_gamma025_3000 = np.array(
    [3100, 2785, 2580, 4299, 3607, 2695, 3128, 2851, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
c_c_f_hw2_gamma025_3000_mean = np.mean(c_c_f_hw2_gamma025_3000)

c_c_f_hw2_gamma025_3000_ge = np.array([1, 1, 25, 3, 5, 1, 1, 2, 10, 17, ])
c_c_f_hw2_gamma025_3000_ge_mean = np.mean(c_c_f_hw2_gamma025_3000_ge)

print()

"focal"
############################                            ches                                ############################

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

c_f_hw2_gamma2_5000 = np.array(
    [2143, 2512, 3278, 3292, 2552, 2856, 3134, 5000, 2484, 2401, 5000, 2968, 5000, 4341, 2460, 2578, 2440, 3935])
c_f_hw2_gamma2_5000_mean = np.mean(c_f_hw2_gamma2_5000)

c_f_hw2_gamma2_5000_ge = np.array([1, 1, 8])
c_f_hw2_gamma2_5000_ge_mean = np.mean(c_f_hw2_gamma2_5000_ge)

c_f_hw2_gamma2_3000 = np.array(
    [3969, 5000, 5000, 5000, 5000, 5000, 3466, 5000, 5000, 5000, 5000, 4662, 5000, 5000, 5000, 3982, 4204, 5000])
c_f_hw2_gamma2_3000_mean = np.mean(c_f_hw2_gamma2_3000)

c_f_hw2_gamma2_3000_ge = np.array([10, 6, 1, 5, 1, 6, 13, 8, 37, 5, 3, 9])
c_f_hw2_gamma2_3000_ge_mean = np.mean(c_f_hw2_gamma2_3000_ge)

# 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105

c_f_hw2_gamma105_5000 = np.array(
    [5000, 2845, 3416, 5000, 2751, 5000, 5000, 3054, 3434, 3521, 4043, 3837, 2926, 2253, 3997, 2950, 3283, 3446])
c_f_hw2_gamma105_5000_mean = np.mean(c_f_hw2_gamma105_5000)

c_f_hw2_gamma105_5000_ge = np.array([])
c_f_hw2_gamma105_5000_ge_mean = np.mean(c_f_hw2_gamma105_5000_ge)

c_f_hw2_gamma105_3000 = np.array(
    [4643, 5000, 5000, 5000, 5000, 5000, 3139, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 4062, 5000, 5000])
c_f_hw2_gamma105_3000_mean = np.mean(c_f_hw2_gamma105_3000)

c_f_hw2_gamma105_3000_ge = np.array([4, 5, 2, 20, 1, 1, 1, 1, 19, 7, 35, 6, 2, 6, 5, ])
c_f_hw2_gamma105_3000_ge_mean = np.mean(c_f_hw2_gamma105_3000_ge)

print()

# 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025

c_f_hw2_gamma1025_5000 = np.array(
    [3358, 3564, 5000, 3727, 1955, 3288, 2006, 2710, 5000, 2874, 3742, 4325, 5000, 3595, 3039, 3113, 5000, 4040])
c_f_hw2_gamma1025_5000_mean = np.mean(c_f_hw2_gamma1025_5000)

c_f_hw2_gamma1025_5000_ge = np.array([])
c_f_hw2_gamma1025_5000_ge_mean = np.mean(c_f_hw2_gamma1025_5000_ge)

c_f_hw2_gamma1025_3000 = np.array(
    [5000, 5000, 4178, 5000, 5000, 3615, 4748, 4708, 5000, 4285, 5000, 5000, 5000, 5000, 5000, 5000, 2857, 5000])
c_f_hw2_gamma1025_3000_mean = np.mean(c_f_hw2_gamma1025_3000)

c_f_hw2_gamma1025_3000_ge = np.array([])
c_f_hw2_gamma1025_3000_ge_mean = np.mean(c_f_hw2_gamma1025_3000_ge)

print()

# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

c_f_hw2_gamma1_5000 = np.array(
    [2894, 2963, 4610, 3750, 5000, 4620, 2793, 5000, 3757, 2482, 5000, 2957, 2526, 3697, 3588, 4472, 2597, 1621])
c_f_hw2_gamma1_5000_mean = np.mean(c_f_hw2_gamma1_5000)

c_f_hw2_gamma1_5000_ge = np.array([])
c_f_hw2_gamma1_5000_ge_mean = np.mean(c_f_hw2_gamma1_5000_ge)

c_f_hw2_gamma1_3000 = np.array(
    [3603, 5000, 5000, 3597, 5000, 5000, 5000, 5000, 4059, 4525, 5000, 3790, 4982, 5000, 2852, 5000, 4650, 4644])
c_f_hw2_gamma1_3000_mean = np.mean(c_f_hw2_gamma1_3000)

c_f_hw2_gamma1_3000_ge = np.array([])
c_f_hw2_gamma1_3000_ge_mean = np.mean(c_f_hw2_gamma1_3000_ge)

# 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05

c_f_hw2_gamma05_5000 = np.array(
    [3367, 2459, 2792, 1980, 4250, 2154, 5000, 3937, 2564, 4694, 5000, 5000, 5000, 3326, 1772, 3335, 3717, 2306])
c_f_hw2_gamma05_5000_mean = np.mean(c_f_hw2_gamma05_5000)

c_f_hw2_gamma05_5000_ge = np.array([])
c_f_hw2_gamma05_5000_ge_mean = np.mean(c_f_hw2_gamma05_5000_ge)

c_f_hw2_gamma05_3000 = np.array(
    [5000, 5000, 5000, 5000, 3039, 4346, 5000, 5000, 5000, 3356, 5000, 5000, 5000, 3555, 5000, 5000, 5000, 5000])
c_f_hw2_gamma05_3000_mean = np.mean(c_f_hw2_gamma05_3000)

c_f_hw2_gamma05_3000_ge = np.array([])
c_f_hw2_gamma05_3000_ge_mean = np.mean(c_f_hw2_gamma05_3000_ge)

# 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025

c_f_hw2_gamma025_5000 = np.array(
    [2858, 2437, 2587, 5000, 3657, 4295, 5000, 5000, 4444, 2894, 2417, 3587, 3696, 2922, 2695, 2147, 2319, 3963])
c_f_hw2_gamma025_5000_mean = np.mean(c_f_hw2_gamma025_5000)

c_f_hw2_gamma025_5000_ge = np.array([])
c_f_hw2_gamma025_5000_ge_mean = np.mean(c_f_hw2_gamma025_5000_ge)

c_f_hw2_gamma025_3000 = np.array(
    [3414, 3653, 5000, 5000, 5000, 4398, 5000, 5000, 4991, 3449, 5000, 3215, 5000, 5000, 4441, 3815, 5000, 4584])
c_f_hw2_gamma025_3000_mean = np.mean(c_f_hw2_gamma025_3000)

c_f_hw2_gamma025_3000_ge = np.array([])
c_f_hw2_gamma025_3000_ge_mean = np.mean(c_f_hw2_gamma025_3000_ge)

############################                            ascad                                ############################

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

a_f_hw2_gamma2_5000 = np.array(
    [2822, 2856, 3170, 1458, 2120, 3625, 3148, 2439, 2105, 2468, 3202, 3613, 5000, 5000, 5000, 5000, 5000, 5000])
a_f_hw2_gamma2_5000_mean = np.mean(a_f_hw2_gamma2_5000)

a_f_hw2_gamma2_5000_ge = np.array([])
a_f_hw2_gamma2_5000_ge_mean = np.mean(a_f_hw2_gamma2_5000_ge)

a_f_hw2_gamma2_3000 = np.array(
    [4371, 2140, 5000, 5000, 5000, 5000, ])
a_f_hw2_gamma2_3000_mean = np.mean(a_f_hw2_gamma2_3000)

a_f_hw2_gamma2_3000_ge = np.array([6, 6, 20, 13, 11, 96, 20, 17, 20, 24, 28, 62, 19, 25, 18, 3])
a_f_hw2_gamma2_3000_ge_mean = np.mean(a_f_hw2_gamma2_3000_ge)

# 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105 105

a_f_hw2_gamma105_5000 = np.array(
    [2413, 3193, 1734, 3014, 1207, 1668, 2938, 3548, 3968, 2466, 1951, 3611, 3662, 3794, 1556, 2540, 5000, 5000])
a_f_hw2_gamma105_5000_mean = np.mean(a_f_hw2_gamma105_5000)

a_f_hw2_gamma105_5000_ge = np.array([1, 1])
a_f_hw2_gamma105_5000_ge_mean = np.mean(a_f_hw2_gamma105_5000_ge)

a_f_hw2_gamma105_3000 = np.array(
    [])
a_f_hw2_gamma105_3000_mean = np.mean(a_f_hw2_gamma105_3000)

a_f_hw2_gamma105_3000_ge = np.array([])
a_f_hw2_gamma105_3000_ge_mean = np.mean(a_f_hw2_gamma105_3000_ge)

print()

# 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025 1025

a_f_hw2_gamma1025_5000 = np.array(
    [3302, 3660, 1682, 3048, 2868, 3129, 2888, 4501, 1904, 1265, 2818, 3175, 2313, 3355, 5000, 5000, 5000, 5000])
a_f_hw2_gamma1025_5000_mean = np.mean(a_f_hw2_gamma1025_5000)

a_f_hw2_gamma1025_5000_ge = np.array([3, 1, 9, 17, ])
a_f_hw2_gamma1025_5000_ge_mean = np.mean(a_f_hw2_gamma1025_5000_ge)

a_f_hw2_gamma1025_3000 = np.array(
    [])
a_f_hw2_gamma1025_3000_mean = np.mean(a_f_hw2_gamma1025_3000)

a_f_hw2_gamma1025_3000_ge = np.array([1, 4, 50, 3, 57, 34, 30, 5, 4, 21, 6, 3, 6, 6, 45, 5, 4, 14])
a_f_hw2_gamma1025_3000_ge_mean = np.mean(a_f_hw2_gamma1025_3000_ge)

print()

# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

a_f_hw2_gamma1_5000 = np.array(
    [3164, 3051, 2596, 3124, 3763, 3351, 1683, 2178, 3245, 2415, 2905, 3606, 5000, 5000, 5000, 5000, 5000, 5000])
a_f_hw2_gamma1_5000_mean = np.mean(a_f_hw2_gamma1_5000)

a_f_hw2_gamma1_5000_ge = np.array([3, 1, 1, 2, 4, 2, ])
a_f_hw2_gamma1_5000_ge_mean = np.mean(a_f_hw2_gamma1_5000_ge)

a_f_hw2_gamma1_3000 = np.array(
    [])
a_f_hw2_gamma1_3000_mean = np.mean(a_f_hw2_gamma1_3000)

a_f_hw2_gamma1_3000_ge = np.array([])
a_f_hw2_gamma1_3000_ge_mean = np.mean(a_f_hw2_gamma1_3000_ge)

# 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05 05

a_f_hw2_gamma05_5000 = np.array(
    [3277, 2303, 3237, 4131, 3276, 1935, 4127, 2918, 2345, 1801, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
a_f_hw2_gamma05_5000_mean = np.mean(a_f_hw2_gamma05_5000)

a_f_hw2_gamma05_5000_ge = np.array([6, 3, 31, 15, 4, 13, 1, 9, ])
a_f_hw2_gamma05_5000_ge_mean = np.mean(a_f_hw2_gamma05_5000_ge)

a_f_hw2_gamma05_3000 = np.array(
    [])
a_f_hw2_gamma05_3000_mean = np.mean(a_f_hw2_gamma05_3000)

a_f_hw2_gamma05_3000_ge = np.array([])
a_f_hw2_gamma05_3000_ge_mean = np.mean(a_f_hw2_gamma05_3000_ge)

# 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025 025

a_f_hw2_gamma025_5000 = np.array(
    [1952, 3114, 3430, 3035, 2523, 3424, 3929, 2545, 2046, 3386, 3935, 2724, 3131, 5000, 5000, 5000, 5000, 5000])
a_f_hw2_gamma025_5000_mean = np.mean(a_f_hw2_gamma025_5000)

a_f_hw2_gamma025_5000_ge = np.array([2, 2, 1, 2, 4])
a_f_hw2_gamma025_5000_ge_mean = np.mean(a_f_hw2_gamma025_5000_ge)

a_f_hw2_gamma025_3000 = np.array(
    [])
a_f_hw2_gamma025_3000_mean = np.mean(a_f_hw2_gamma025_3000)

a_f_hw2_gamma025_3000_ge = np.array([])
a_f_hw2_gamma025_3000_ge_mean = np.mean(a_f_hw2_gamma025_3000_ge)

print()

"cb"
############################                            ches                                ############################

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

c_cb_hw2_45000 = np.array(
    [592 + 813 + 1038 + 669 + 1263 + 1321 + 728 + 476 + 617 + 1125 + 948 + 803 + 1082 + 763 + 623 + 622 + 563 + 878])
c_cb_hw2_45000_mean = np.mean(c_cb_hw2_45000) / 18

c_cb_hw2_10000 = np.array(
    [
        1915 + 1086 + 1026 + 1811 + 1662 + 2063 + 1571 + 751 + 1321 + 1401 + 1631 + 1296 + 542 + 2271 + 1692 + 2313 + 1746 + 1637])
c_cb_hw2_10000_mean = np.mean(c_cb_hw2_10000) / 18

c_cb_hw2_7000 = np.array(
    [1560, 3978, 872, 1139, 3182, 1672, 1721, 1113, 771, 1833, 2571, 1812, 869, 2085, 1824, 2592, 5000, 2973])
c_cb_hw2_7000_mean = np.mean(c_cb_hw2_7000)

c_cb_hw2_5000 = np.array(
    [2611, 2533, 1470, 1587, 1708, 2235, 1808, 4058, 3306, 4076, 4155, 2661, 1449, 3267, 4017, 3666, 3012, 2856])
c_cb_hw2_5000_mean = np.mean(c_cb_hw2_5000)

c_cb_hw2_5000_ge = np.array([])
c_cb_hw2_5000_ge_mean = np.mean(c_cb_hw2_5000_ge)

c_cb_hw2_3000 = np.array(
    [3958, 5000, 5000, 5000, 5000, 3289, 4453, 2628, 4423, 5000, 5000, 4170, 5000, 5000, 5000, 2563, 3350, 4176])
c_cb_hw2_3000_mean = np.mean(c_cb_hw2_3000)

c_cb_hw2_3000_ge = np.array([])
c_cb_hw2_3000_ge_mean = np.mean(c_cb_hw2_3000_ge)

############################                            ascad                                ############################

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
a_cb_hw1_10000 = np.array(
    [570, 402, 374, 580, 849, 329, 452, 505, 679, 449, 451, 825, 811, 413, 371, 697, 583, 491, ])
a_cb_hw1_10000_mean = np.mean(a_cb_hw1_10000)

a_cb_hw1_7000 = np.array(
    [254, 352, 442, 339, 374, 399, 433, 329, 286, 401, 304, 432, 345, 326, 356, 404, 432, 466, ])
a_cb_hw1_7000_mean = np.mean(a_cb_hw1_7000)

a_cb_hw1_5000 = np.array(
    [361, 324, 588, 282, 374, 524, 542, 369, 353, 362, 444, 331, 306, 359, 392, 424, 268, 682, ])
a_cb_hw1_5000_mean = np.mean(a_cb_hw1_5000)

a_cb_hw1_3000 = np.array(
    [904, 949, 841, 616, 401, 757, 444, 411, 511, 377, 539, 496, 501, 538, 486, 674, 452, 705, 499, 663])
a_cb_hw1_3000_mean = np.mean(a_cb_hw1_3000)

a_cb_hw1_3000_ge = np.array([])
a_cb_hw1_3000_ge_mean = np.mean(a_cb_hw1_3000_ge)

a_cb_hw1_1000 = np.array(
    [1839, 2339, 1767, 3856, 3012, 3106, 3837, 1499, 2758, 3427, 3493, 2083, 2231, 1429, 5000, 3835, 2562, 2000])
a_cb_hw2_1000_mean = np.mean(a_cb_hw1_1000)

a_cb_hw1_1000_ge = np.array([])
a_cb_hw1_1000_ge_mean = np.mean(a_cb_hw1_1000_ge)

############################                            ascad_rand                                ############################

# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
ar_cb_hw2_50000 = np.array(
    [1024 + 778 + 889 + 456 + 767 + 894 + 850 + 928 + 1134])
ar_cb_hw2_50000_mean = np.mean(ar_cb_hw2_50000) / 9

ar_cb_hw2_10000 = np.array(
    [
        1095 + 1291 + 1236 + 1020 + 1670 + 1297 + 2015 + 1321 + 1745 + 1049 + 2080 + 1567 + 885 + 2466 + 1455 + 1159 + 778 + 1936])
ar_cb_hw2_10000_mean = np.mean(ar_cb_hw2_10000) / 18

ar_cb_hw2_7000 = np.array(
    [
        857 + 2416 + 1179 + 1309 + 1044 + 1639 + 873 + 898 + 2368 + 1211 + 1035 + 1335 + 1442 + 1619 + 763 + 1237 + 2300 + 1289])
ar_cb_hw2_7000_mean = np.mean(ar_cb_hw2_7000) / 18

ar_cb_hw2_5000 = np.array(
    [
        2640 + 1817 + 1732 + 1523 + 2344 + 2495 + 1786 + 1329 + 1557 + 1832 + 1590 + 1099 + 1059 + 1990 + 2371 + 1434 + 2167 + 4594])
ar_cb_hw2_5000_mean = np.mean(ar_cb_hw2_5000) / 18

ar_cb_hw2_3000 = np.array(
    [
        2974 + 2772 + 3327 + 2698 + 5000 + 2635 + 1904 + 2516 + 3817 + 3080 + 1888 + 5000 + 4126 + 2145 + 2496 + 3254 + 3758 + 5000])
ar_cb_hw2_3000_mean = np.mean(ar_cb_hw2_3000) / 18

ar_cb_hw2_3000_ge = np.array([])
ar_cb_hw2_3000_ge_mean = np.mean(ar_cb_hw2_3000_ge)

ar_cb_hw2_1000 = np.array(
    [])
ar_cb_hw2_1000_mean = np.mean(ar_cb_hw2_1000)

ar_cb_hw2_1000_ge = np.array([])
ar_cb_hw2_1000_ge_mean = np.mean(ar_cb_hw2_1000_ge)

"LDAM"
ar_ldam_hw2_50000 = np.array(
    [595 + 544 + 639 + 640 + 973 + 1054 + 1106 + 723 + 503 + 401 + 688 + 830 + 1020 + 796 + 517 + 482])
ar_ldam_hw2_50000_mean = np.mean(ar_ldam_hw2_50000) / 16

ar_ldam_hw2_10000 = np.array(
    [1667 + 1858 + 2037 + 1710 + 1153 + 1447 + 921 + 1794 + 1375 + 1514 + 1589 + 2081 + 2435 + 1497 + 1970 + 1442])
ar_ldam_hw2_10000_mean = np.mean(ar_ldam_hw2_10000) / 16

ar_ldam_hw2_7000 = np.array(
    [2745 + 1526 + 1560 + 2208 + 1808 + 2066 + 999 + 2467 + 2978 + 1969 + 2531 + 1698 + 1869 + 2289 + 2566 + 1046])
ar_ldam_hw2_7000_mean = np.mean(ar_ldam_hw2_7000) / 16

ar_ldam_hw2_5000 = np.array(
    [5000 + 2493 + 1676 + 2224 + 2426 + 4156 + 2375 + 1625 + 2927 + 1689 + 1378 + 2446 + 2326 + 2502 + 2119 + 2544])
ar_ldam_hw2_5000_mean = np.mean(ar_ldam_hw2_5000) / 16

ar_ldam_hw2_3000 = np.array(
    [2797 + 3609 + 2051 + 2522 + 4214 + 5000 + 3984 + 2905 + 4880 + 2169 + 3817 + 2913 + 5000 + 2890 + 2395 + 4755])
ar_ldam_hw2_3000_mean = np.mean(ar_ldam_hw2_3000) / 16

ar_ldam_hw2_3000_ge = np.array([])
ar_ldam_hw2_3000_ge_mean = np.mean(ar_ldam_hw2_3000_ge)

a_ldam_hw2_50000 = np.array(
    [])
a_ldam_hw2_50000_mean = np.mean(a_ldam_hw2_50000) / 16

a_ldam_hw2_10000 = np.array(
    [1148 + 1672 + 2101 + 841 + 1549 + 1189 + 834 + 1610 + 833 + 1045 + 1105 + 1493 + 1288 + 949 + 1838 + 1673])
a_ldam_hw2_10000_mean = np.mean(a_ldam_hw2_10000) / 16

a_ldam_hw2_7000 = np.array(
    [
        2013 + 1923 + 2376 + 1453 + 1599 + 885 + 2732 + 2685 + 1994 + 1236 + 1322 + 1484 + 1861 + 2348 + 2558 + 2228 + 2266 + 2038])
a_ldam_hw2_7000_mean = np.mean(a_ldam_hw2_7000) / 18

a_ldam_hw2_5000 = np.array(
    [
        2362 + 1143 + 3317 + 5000 + 1572 + 2155 + 1297 + 2887 + 1693 + 4792 + 1908 + 3069 + 1871 + 2200 + 2232 + 1573 + 3873 + 3820])
a_ldam_hw2_5000_mean = np.mean(a_ldam_hw2_5000) / 18

a_ldam_hw2_3000 = np.array(
    [
        2773 + 1246 + 4326 + 2435 + 2380 + 3780 + 3004 + 1426 + 2011 + 3224 + 3110 + 3462 + 3719 + 5000 + 5000 + 5000 + 5000 + 5000])
a_ldam_hw2_3000_mean = np.mean(a_ldam_hw2_3000) / 18

a_ldam_hw2_3000_ge = np.array([2 + 1 + 1 + 1 + 1])
a_ldam_hw2_3000_ge_mean = np.mean(a_ldam_hw2_3000_ge)

a_ldam_hw2_1000 = np.array(
    [])
a_ldam_hw2_1000_mean = np.mean(a_ldam_hw2_1000)

a_ldam_hw2_1000_ge = np.array([])
a_ldam_hw2_1000_ge_mean = np.mean(a_ldam_hw2_1000_ge)

c_ldam_hw2_50000 = np.array(
    [
        676 + 548 + 621 + 701 + 778 + 847 + 769 + 897 + 634 + 586 + 1223 + 839 + 917 + 592 + 942 + 634 + 769 + 491 + 442 + 894])
c_ldam_hw2_50000_mean = np.mean(c_ldam_hw2_50000) / 20

c_ldam_hw2_10000 = np.array(
    [1829 + 1433 + 1513 + 1201 + 1639 + 1806 + 1371 + 2205 + 1664 + 2349 + 4222 + 2760 + 1815 + 1639])
c_ldam_hw2_10000_mean = np.mean(c_ldam_hw2_10000) / 14

c_ldam_hw2_7000 = np.array(
    [])
c_ldam_hw2_7000_mean = np.mean(c_ldam_hw2_7000) / 18

c_ldam_hw2_5000 = np.array(
    [])
c_ldam_hw2_5000_mean = np.mean(c_ldam_hw2_5000) / 18

c_ldam_hw2_3000 = np.array(
    [])
c_ldam_hw2_3000_mean = np.mean(c_ldam_hw2_3000) / 18

c_ldam_hw2_3000_ge = np.array([])
c_ldam_hw2_3000_ge_mean = np.mean(c_ldam_hw2_3000_ge)

c_ldam_ce_45000 = np.array(
    [481 + 435 + 475 + 435 + 327 + 717 + 471 + 472 + 477 + 432 + 727 + 324 + 497 + 489 + 299 + 410])
c_ldam_ce_45000_mean = np.mean(c_ldam_ce_45000) / 16

ar_ldam_ce_45000 = np.array(
    [559 + 796 + 602 + 864 + 499 + 605 + 610 + 565 + 799 + 689 + 1046 + 661 + 651 + 809 + 683 + 440])
ar_ldam_ce_45000_mean = np.mean(ar_ldam_ce_45000) / 16

a_ldam_ce_45000 = np.array(
    [643 + 1091 + 1092 + 807 + 1085 + 677 + 758 + 865 + 1154 + 1348 + 695 + 1127 + 1152 + 980 + 1667])
a_ldam_ce_45000_mean = np.mean(a_ldam_ce_45000) / 16

"logit"
a_logit_50000 = np.array(
    [])
a_logit_50000_mean = np.mean(a_logit_50000) / 18

a_logit_7000 = np.array(
    [431 + 286 + 397 + 825 + 300 + 475 + 377 + 470 + 1407 + 310 + 325 + 470 + 231 + 418 + 1147 + 782 + 317 + 668])
a_logit_7000_mean = np.mean(a_logit_7000) / 18

a_logit_5000 = np.array(
    [324 + 315 + 727 + 418 + 710 + 381 + 281])
a_logit_5000_mean = np.mean(a_logit_5000) / 18

ar_logit_50000 = np.array(
    [242 + 372 + 320 + 365 + 361 + 426 + 390 + 255 + 425 + 270 + 313 + 421 + 288 + 348 + 399])
ar_logit_50000_mean = np.mean(ar_logit_50000) / 15

c_logit_45000 = np.array(
    [318 + 390 + 345 + 289 + 223 + 232 + 191 + 253 + 232 + 248 + 203 + 250 + 280 + 312 + 178 + 243 + 502 + 202])
c_logit_45000_mean = np.mean(c_logit_45000) / 18

" ce_logit&ldl_logit vs flr "
c_ce_logit_45000_3000_5000 = np.array([302 + 447 + 487 + 308 + 230 + 347 + 295 + 358 + 256 + 446 + 256 + 229])
c_ce_logit_45000_3000_5000_mean = np.mean(c_ce_logit_45000_3000_5000) / 12
print()

"tao"
# c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c
########################################################### hw2 ##########################################################

# 10000
c_0_10000_ldl = np.array([
    5000 + 2021 + 1516 + 3298 + 2022 + 2596 + 1087 + 1639 + 1875 + 1919 + 1967 + 1184 + 2529 + 2006 + 1905 + 1433 + 3034 + 2511])
c_0_10000_ldl_mean = np.mean(c_0_10000_ldl) / 18

c_08_10000_ldl = np.array([
    3146 + 3790 + 3061 + 4848 + 2031 + 2142 + 2033 + 2276 + 2978 + 2086 + 4901 + 4048 + 2934 + 1832 + 5000 + 3757 + 2613])
c_08_10000_ldl_mean = np.mean(c_08_10000_ldl) / 17

c_07_10000_ldl = np.array([2681 + 2949 + 4347 + 3426 + 1486 + 2663 + 2695 + 2781 + 3391 + 3233 + 2808 + 3258 + 2109])
c_07_10000_ldl_mean = np.mean(c_07_10000_ldl) / 13

c_06_10000_ldl = np.array([2060 + 983 + 1227 + 1658 + 3637 + 1537 + 1185 + 1933 + 2840 + 2671 + 1033 + 1573])
c_06_10000_ldl_mean = np.mean(c_06_10000_ldl) / 12

c_05_10000_ldl = np.array([
    2756 + 853 + 1319 + 1610 + 1124 + 1282 + 1721 + 2785 + 770 + 1508 + 1226 + 1303 + 2168 + 1551 + 2507 + 1475 + 2150 + 1130])
c_05_10000_ldl_mean = np.mean(c_05_10000_ldl) / 18

c_04_10000_ldl = np.array([
    1083 + 2130 + 877 + 1981 + 1021 + 1116 + 2682 + 2472 + 1120 + 590 + 2515 + 2254 + 1774 + 2372 + 2365 + 1131 + 679 + 1363])
c_04_10000_ldl_mean = np.mean(c_04_10000_ldl) / 18

c_02_10000_ldl = np.array([1145 + 1300 + 2109 + 1824 + 1721 + 1205 + 2247 + 1834 + 674 + 1874 + 1304 + 677])
c_02_10000_ldl_mean = np.mean(c_02_10000_ldl) / 12

c_01_10000_ldl = np.array([
    2283 + 1241 + 4252 + 1714 + 1812 + 1247 + 1877 + 2537 + 1231 + 2241 + 1106 + 1848 + 2085 + 1117 + 1444 + 972 + 1028 + 1214])
c_01_10000_ldl_mean = np.mean(c_01_10000_ldl) / 18

# 50000

c_1_50000_ldl = np.array([1868 + 2198 + 2072 + 2060 + 1310 + 2484 + 1242 + 1885 + 1445 + 2023 + 2435 + 2486])
c_1_50000_ldl_mean = np.mean(c_1_50000_ldl) / 12

c_08_50000_ldl = np.array([2857 + 1256 + 1307 + 1306 + 1513 + 1430 + 996 + 937])
c_08_50000_ldl_mean = np.mean(c_08_50000_ldl) / 8

c_07_50000_ldl = np.array([1033 + 756 + 777 + 784 + 1376])
c_07_50000_ldl_mean = np.mean(c_07_50000_ldl) / 5

c_06_50000_ldl = np.array([])
c_06_50000_ldl_mean = np.mean(c_06_50000_ldl) / 12

c_05_50000_ldl = np.array([747 + 479 + 407 + 756 + 361 + 685 + 904 + 593
                           ])
c_05_50000_ldl_mean = np.mean(c_05_50000_ldl) / 8

c_04_50000_ldl = np.array([508 + 411 + 347 + 441 + 336 + 368 + 314 + 860 + 408 + 251 + 427 + 282
                           ])
c_04_50000_ldl_mean = np.mean(c_04_50000_ldl) / 12

c_025_50000_ldl = np.array([326 + 292 + 309 + 306 + 536 + 310 + 329 + 317 + 285 + 272 + 391 + 607])
c_025_50000_ldl_mean = np.mean(c_025_50000_ldl) / 12

c_02_50000_ldl = np.array([457 + 667 + 441 + 344 + 490 + 464 + 255 + 379])
c_02_50000_ldl_mean = np.mean(c_02_50000_ldl) / 8

c_01_50000_ldl = np.array([407 + 840 + 559 + 598 + 553 + 647 + 399 + 593 + 884 + 493 + 378])
c_01_50000_ldl_mean = np.mean(c_01_50000_ldl) / 11

########################################################### hw1 ##########################################################
#5000

c_0_5000_ldl_hw1 = np.array([2279 + 2701 + 2737])
c_0_5000_ldl_hw1_mean = np.mean(c_0_5000_ldl_hw1) / 3

c_1_5000_ldl_hw1 = np.array([2071 + 5000 + 5000 + 3190 + 5000 + 5000 + 3451 + 4028 + 2114 + 2935 + 2716 + 2654])
c_1_5000_ldl_hw1_mean = np.mean(c_1_5000_ldl_hw1) / 12

c_08_5000_ldl_hw1 = np.array([2828 + 1382 + 1886 + 2866 + 5000 + 3277 + 3420 + 1562 + 2035 + 3899 + 1795 + 2428
                              ])
c_08_5000_ldl_hw1_mean = np.mean(c_08_5000_ldl_hw1) / 12

c_06_5000_ldl_hw1 = np.array([1927 + 2025 + 2254 + 2840 + 2026 + 3256 + 3970 + 4445 + 3403 + 2190 + 923 + 3366])
c_06_5000_ldl_hw1_mean = np.mean(c_06_5000_ldl_hw1) / 12

c_05_5000_ldl_hw1 = np.array([5000 + 1968 + 2230 + 5000 + 2723 + 2846 + 3381 + 951 + 3882 + 4248 + 2412 + 3257
                              ])
c_05_5000_ldl_hw1_mean = np.mean(c_05_5000_ldl_hw1) / 12

c_04_5000_ldl_hw1 = np.array([2378 + 2061 + 2883 + 5000 + 2139 + 2793 + 2771 + 2544 + 3786 + 2326 + 1923 + 1878])
c_04_5000_ldl_hw1_mean = np.mean(c_04_5000_ldl_hw1) / 12

c_02_5000_ldl_hw1 = np.array([4152 + 5000 + 3236 + 4625 + 4211 + 2431 + 5000 + 3825 + 2920 + 1491 + 2705 + 3746])
c_02_5000_ldl_hw1_mean = np.mean(c_02_5000_ldl_hw1) / 12

c_01_5000_ldl_hw1 = np.array([2186 + 4613 + 4728 + 4093 + 2314 + 5000 + 2760 + 2100 + 2271 + 3296 + 3800 + 1884
                              ])
c_01_5000_ldl_hw1_hw1__mean = np.mean(c_01_5000_ldl_hw1) / 12

# 10000
c_11_10000_ldl_hw1 = np.array([1899 + 2095 + 936 + 894 + 3285 + 1181 + 1598 + 994 + 983 + 1998])
c_11_10000_ldl_hw1_mean = np.mean(c_11_10000_ldl_hw1) / 10

c_0_10000_ldl_hw1 = np.array([1867 + 2641 + 3322 + 1726 + 2409 + 1912 + 2239 + 2045 + 1955 + 1818 + 3013])
c_0_10000_ldl_hw1_mean = np.mean(c_0_10000_ldl_hw1) / 11

c_08_10000_ldl_hw1 = np.array([806 + 2594 + 1121 + 1512 + 1245 + 1331 + 1349 + 653 + 1859 + 953 + 932
                               ])
c_08_10000_ldl_hw1_mean = np.mean(c_08_10000_ldl_hw1) / 11

c_07_10000_ldl_hw1 = np.array([921 + 2011 + 2495 + 1291])
c_07_10000_ldl_hw1_mean = np.mean(c_07_10000_ldl_hw1) / 4

c_06_10000_ldl_hw1 = np.array([1051 + 651 + 851 + 778 + 1165 + 1676 + 1095 + 1261])
c_06_10000_ldl_hw1_mean = np.mean(c_06_10000_ldl_hw1) / 8

c_05_10000_ldl_hw1 = np.array([1098 + 1418 + 634 + 1159 + 1058 + 1701 + 1157 + 813 + 1395
                               ])
c_05_10000_ldl_hw1_mean = np.mean(c_05_10000_ldl_hw1) / 9

c_04_10000_ldl_hw1 = np.array([1007 + 2524 + 2610 + 1630 + 977 + 914 + 1937 + 424 + 1349
                               ])
c_04_10000_ldl_hw1_mean = np.mean(c_04_10000_ldl_hw1) / 9

c_02_10000_ldl_hw1 = np.array([2404 + 996 + 1136 + 2027 + 1008 + 1382 + 1349 + 1574 + 1577])
c_02_10000_ldl_hw1_mean = np.mean(c_02_10000_ldl_hw1) / 9

c_01_10000_ldl_hw1 = np.array([1658 + 1592 + 1032 + 1190 + 1975
                               ])
c_01_10000_ldl_hw1_hw1__mean = np.mean(c_01_10000_ldl_hw1) / 5

# 50000

c_0_50000_ldl_hw1 = np.array([433 + 569 + 671 + 714 + 456 + 638 + 352 + 550])
c_0_50000_ldl_hw1_mean = np.mean(c_0_50000_ldl_hw1) / 8

c_1_50000_ldl_hw1 = np.array([247 + 428 + 531 + 674 + 468 + 296 + 547 + 400 + 691 + 406 + 413 + 445])
c_1_50000_ldl_hw1_mean = np.mean(c_1_50000_ldl_hw1) / 12

c_08_50000_ldl_hw1 = np.array([263 + 177 + 448 + 279 + 295 + 235 + 317 + 342 + 251 + 325 + 283 + 299
                               ])
c_08_50000_ldl_hw1_mean = np.mean(c_08_50000_ldl_hw1) / 12

c_06_50000_ldl_hw1 = np.array([315 + 275 + 223 + 335 + 413 + 223 + 288 + 237])
c_06_50000_ldl_hw1_mean = np.mean(c_06_50000_ldl_hw1) / 8

c_07_50000_ldl_hw1 = np.array([260 + 233 + 294 + 345 + 253 + 345 + 310 + 314 + 202 + 280 + 314])
c_07_50000_ldl_hw1_mean = np.mean(c_07_50000_ldl_hw1) / 11

c_075_50000_ldl_hw1 = np.array([221 + 238 + 236 + 230 + 322 + 234 + 380 + 361 + 324 + 285])
c_075_50000_ldl_hw1_mean = np.mean(c_075_50000_ldl_hw1) / 10

c_05_50000_ldl_hw1 = np.array([239 + 226 + 212 + 600 + 454 + 242 + 538 + 525 + 463 + 285 + 372 + 441
                               ])
c_05_50000_ldl_hw1_mean = np.mean(c_05_50000_ldl_hw1) / 12

c_04_50000_ldl_hw1 = np.array([250 + 416 + 241 + 348 + 280 + 510 + 586 + 389 + 318 + 448 + 369
                               ])
c_04_50000_ldl_hw1_mean = np.mean(c_04_50000_ldl_hw1) / 11

c_02_50000_ldl_hw1 = np.array([425 + 862 + 609 + 362 + 658 + 326 + 443])
c_02_50000_ldl_hw1_mean = np.mean(c_02_50000_ldl_hw1) / 7

c_01_50000_ldl_hw1 = np.array([337 + 359 + 435 + 807 + 359 + 749 + 422 + 377 + 416 + 511 + 514 + 824
                               ])
c_01_50000_ldl_hw1_hw1__mean = np.mean(c_01_50000_ldl_hw1) / 12

# a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
########################################################### hw2 ##########################################################


#10000
a_hw2_08_10000_ldl = np.array([1403 + 640 + 1030 + 652 + 1024 + 748 + 1098 + 1262 + 835 + 1108 + 1087 + 1070])
a_hw2_08_10000_ldl_mean = np.mean(a_hw2_08_10000_ldl) / 12

a_hw2_07_10000_ldl = np.array([902 + 541 + 988 + 940 + 1424 + 913 + 1004 + 599 + 871 + 847 + 560 + 636])
a_hw2_07_10000_ldl_mean = np.mean(a_hw2_07_10000_ldl) / 12

a_hw2_06_10000_ldl = np.array([577 + 340 + 408 + 468 + 687 + 1091 + 652 + 476 + 553 + 371 + 639 + 602])
a_hw2_06_10000_ldl_mean = np.mean(a_hw2_06_10000_ldl) / 12

a_hw2_05_10000_ldl = np.array([461 + 295 + 385 + 520 + 465 + 387 + 499 + 415 + 387 + 421 + 449 + 642])
a_hw2_05_10000_ldl_mean = np.mean(a_hw2_05_10000_ldl) / 12

a_hw2_04_10000_ldl = np.array([260 + 299 + 168 + 153 + 175 + 178 + 379 + 299 + 359 + 271 + 251 + 298])
a_hw2_04_10000_ldl_mean = np.mean(a_hw2_04_10000_ldl) / 12

a_hw2_025_10000_ldl = np.array([406 + 232 + 420 + 240 + 322 + 218 + 400 + 336 + 284 + 256 + 283 + 352])
a_hw2_025_10000_ldl_mean = np.mean(a_hw2_025_10000_ldl) / 12

a_hw2_02_10000_ldl = np.array([425 + 227 + 379 + 319 + 247 + 297 + 277 + 296 + 373 + 333 + 460])
a_hw2_02_10000_ldl_mean = np.mean(a_hw2_02_10000_ldl) / 11

a_hw2_01_10000_ldl = np.array([1042 + 615 + 654 + 524 + 759 + 718 + 784 + 975 + 1509 + 707 + 812 + 647])
a_hw2_01_10000_ldl_mean = np.mean(a_hw2_01_10000_ldl) / 12
#50000

a_hw2_0_50000_ldl = np.array([
                                 770 + 1040 + 1106 + 1014 + 751 + 722 + 1339 + 1249 + 618 + 1104 + 890 + 2095 + 1630 + 1158 + 1071 + 711 + 840 + 569])
a_hw2_0_50000_ldl_mean = np.mean(a_hw2_0_50000_ldl) / 18

a_hw2_1_50000_ldl = np.array([1245 + 996 + 1093 + 1306 + 1831 + 713 + 507 + 1073 + 832 + 1280 + 769 + 797])
a_hw2_1_50000_ldl_mean = np.mean(a_hw2_1_50000_ldl) / 12

a_hw2_08_50000_ldl = np.array([995 + 1028 + 752 + 706 + 1042 + 718 + 1044 + 891 + 856 + 958 + 1073 + 990])
a_hw2_08_50000_ldl_mean = np.mean(a_hw2_08_50000_ldl) / 12

a_hw2_06_50000_ldl = np.array([336 + 408 + 417 + 572 + 526 + 616 + 393 + 749 + 404 + 567 + 546 + 363])
a_hw2_06_50000_ldl_mean = np.mean(a_hw2_06_50000_ldl) / 12

a_hw2_05_50000_ldl = np.array([344 + 295 + 280 + 367 + 213 + 291 + 245 + 262])
a_hw2_05_50000_ldl_mean = np.mean(a_hw2_05_50000_ldl) / 8

a_hw2_04_50000_ldl = np.array([209 + 153 + 207 + 195 + 112 + 141 + 181 + 209 + 211 + 116 + 138 + 146])
a_hw2_04_50000_ldl_mean = np.mean(a_hw2_04_50000_ldl) / 12

a_hw2_02_50000_ldl = np.array([344 + 359 + 261 + 428 + 302 + 231 + 436 + 292 + 320 + 383 + 261 + 360])
a_hw2_02_50000_ldl_mean = np.mean(a_hw2_02_50000_ldl) / 12

a_hw2_01_50000_ldl = np.array([768 + 397 + 560 + 408 + 483 + 542 + 578 + 1008 + 503 + 578 + 371])
a_hw2_01_50000_ldl_mean = np.mean(a_hw2_01_50000_ldl) / 11

#7000

a_hw2_0_7000_ldl = np.array([1172 + 1558 + 1175 + 1715])
a_hw2_0_7000_ldl_mean = np.mean(a_hw2_0_7000_ldl) / 4

a_hw2_1_7000_ldl = np.array([1514 + 143 + 761 + 1100 + 1654 + 1369 + 1395 + 1211 + 1027 + 1424 + 1302 + 868])
a_hw2_1_7000_ldl_mean = np.mean(a_hw2_1_7000_ldl) / 12

a_hw2_08_7000_ldl = np.array([773 + 828 + 1156 + 1077 + 662 + 830 + 896 + 689])
a_hw2_08_7000_ldl_mean = np.mean(a_hw2_08_7000_ldl) / 8

a_hw2_06_7000_ldl = np.array([599 + 613 + 469 + 442 + 629 + 459 + 389 + 515 + 829 + 562 + 514 + 599])
a_hw2_06_7000_ldl_mean = np.mean(a_hw2_06_7000_ldl) / 12

a_hw2_05_7000_ldl = np.array([331 + 640 + 478 + 270 + 399 + 330 + 297 + 565 + 445 + 377 + 358 + 527])
a_hw2_05_7000_ldl_mean = np.mean(a_hw2_05_7000_ldl) / 12

a_hw2_04_7000_ldl = np.array([319 + 447 + 230 + 287 + 239 + 219 + 227 + 326 + 213 + 387 + 341 + 324])
a_hw2_04_7000_ldl_mean = np.mean(a_hw2_04_7000_ldl) / 12

a_hw2_025_7000_ldl = np.array([272 + 256 + 378 + 342 + 226 + 317 + 446 + 413 + 341 + 292 + 347])
a_hw2_025_7000_ldl_mean = np.mean(a_hw2_025_7000_ldl) / 11

a_hw2_02_7000_ldl = np.array([525 + 617 + 321 + 434 + 550 + 344 + 591 + 310 + 357 + 415 + 534 + 216])
a_hw2_02_7000_ldl_mean = np.mean(a_hw2_02_7000_ldl) / 12

a_hw2_01_7000_ldl = np.array([911 + 769 + 603 + 583 + 644 + 969 + 945 + 1078 + 763 + 614 + 654 + 655])
a_hw2_01_7000_ldl_mean = np.mean(a_hw2_01_7000_ldl) / 12

#5000

a_hw2_0_5000_ldl = np.array([
                                1041 + 1252 + 1288 + 1726 + 537 + 2110 + 1145 + 1614 + 1422 + 759 + 1061 + 1950 + 1532 + 1494 + 1927 + 1717 + 1355 + 1129 + 1453 + 1049])
a_hw2_0_5000_ldl_mean = np.mean(a_hw2_0_5000_ldl) / 20

a_hw2_1_5000_ldl = np.array([1493 + 881 + 1686 + 1221 + 1349 + 1132 + 1702 + 1243 + 1094 + 1774 + 1711 + 1062])
a_hw2_1_5000_ldl_mean = np.mean(a_hw2_1_5000_ldl) / 12

a_hw2_08_5000_ldl = np.array([909 + 1230 + 931 + 1374 + 1199 + 989 + 877 + 991 + 754 + 1354 + 1045 + 972])
a_hw2_08_5000_ldl_mean = np.mean(a_hw2_08_5000_ldl) / 12

a_hw2_06_5000_ldl = np.array([555 + 836 + 920 + 791 + 740 + 846 + 586 + 521 + 451 + 573 + 770 + 866])
a_hw2_06_5000_ldl_mean = np.mean(a_hw2_06_5000_ldl) / 12

a_hw2_05_5000_ldl = np.array([546 + 325 + 504 + 533 + 491 + 553 + 440 + 441 + 575 + 443 + 520 + 538])
a_hw2_05_5000_ldl_mean = np.mean(a_hw2_05_5000_ldl) / 12

a_hw2_04_5000_ldl = np.array([339 + 391 + 303 + 278 + 463 + 424 + 400 + 407 + 318 + 332 + 383 + 222])
a_hw2_04_5000_ldl_mean = np.mean(a_hw2_04_5000_ldl) / 12

a_hw2_025_5000_ldl = np.array([442 + 345 + 403 + 343 + 390 + 515 + 476 + 276 + 567 + 513])
a_hw2_025_5000_ldl_mean = np.mean(a_hw2_025_5000_ldl) / 10

a_hw2_02_5000_ldl = np.array([458 + 695 + 329 + 485 + 515 + 557 + 525 + 385 + 393 + 429 + 549 + 419])
a_hw2_02_5000_ldl_mean = np.mean(a_hw2_02_5000_ldl) / 12

a_hw2_01_5000_ldl = np.array([996 + 849 + 875 + 796 + 714 + 1317 + 547 + 905 + 946 + 936 + 638 + 618])
a_hw2_01_5000_ldl_mean = np.mean(a_hw2_01_5000_ldl) / 12

#3000
a_hw2_08_3000_ldl = np.array([1571 + 1096 + 1558 + 1057 + 1228 + 1867 + 1171 + 1461 + 1442 + 1233 + 1424 + 1649])
a_hw2_08_3000_ldl_mean = np.mean(a_hw2_08_3000_ldl) / 12

a_hw2_07_3000_ldl = np.array([1925 + 1384 + 951 + 1476 + 1081 + 853 + 756 + 1193 + 1361 + 1151 + 1550 + 1026])
a_hw2_07_3000_ldl_mean = np.mean(a_hw2_07_3000_ldl) / 12

a_hw2_06_3000_ldl = np.array([885 + 1037 + 670 + 1348 + 1129 + 825 + 907 + 936 + 1600 + 577 + 807 + 683])
a_hw2_06_3000_ldl_mean = np.mean(a_hw2_06_3000_ldl) / 12

a_hw2_05_3000_ldl = np.array([526 + 824 + 845 + 1009 + 597 + 632 + 415 + 838 + 537 + 708 + 510 + 746])
a_hw2_05_3000_ldl_mean = np.mean(a_hw2_05_3000_ldl) / 12

a_hw2_04_3000_ldl = np.array([356 + 503 + 520 + 554 + 850 + 653 + 619 + 615 + 670 + 642 + 437 + 542])
a_hw2_04_3000_ldl_mean = np.mean(a_hw2_04_3000_ldl) / 12

a_hw2_025_3000_ldl = np.array([461 + 434 + 932 + 774 + 683 + 581 + 719 + 464 + 656 + 783 + 737 + 429])
a_hw2_025_3000_ldl_mean = np.mean(a_hw2_025_3000_ldl) / 12

a_hw2_01_3000_ldl = np.array([1265 + 2127 + 1296 + 2549 + 1181 + 1783 + 1730 + 2104 + 2018 + 1956 + 1166 + 1168])
a_hw2_01_3000_ldl_mean = np.mean(a_hw2_01_3000_ldl) / 12

#1000
a_hw2_0_1000_ldl = np.array([])
a_hw2_0_1000_ldl_mean = np.mean(a_hw2_0_1000_ldl) / 18

a_hw2_1_1000_ldl = np.array([3509 + 1397 + 2314 + 1774 + 1866 + 2368 + 2980 + 2243 + 2731 + 3305 + 2045 + 2799])
a_hw2_1_1000_ldl_mean = np.mean(a_hw2_1_1000_ldl) / 12

a_hw2_08_1000_ldl = np.array([1987 + 2320 + 1610 + 3273 + 2504 + 2476 + 1800 + 1717 + 2121 + 1411 + 1632 + 1079])
a_hw2_08_1000_ldl_mean = np.mean(a_hw2_08_1000_ldl) / 12

a_hw2_06_1000_ldl = np.array([1718 + 1698 + 1459 + 1647 + 1231 + 1587 + 2388 + 2648 + 1322 + 1730 + 1870 + 1722])
a_hw2_06_1000_ldl_mean = np.mean(a_hw2_06_1000_ldl) / 12

a_hw2_05_1000_ldl = np.array(
    [2816 + 1060 + 889 + 1377 + 3260 + 2187 + 2014 + 1840 + 1725 + 1591 + 1544 + 1220 + 1930 + 2195])
a_hw2_05_1000_ldl_mean = np.mean(a_hw2_05_1000_ldl) / 14

a_hw2_04_1000_ldl = np.array([1337 + 5000 + 1375 + 1773 + 1240 + 1862 + 1583 + 1397 + 1506 + 1741 + 1164 + 952])
a_hw2_04_1000_ldl_mean = np.mean(a_hw2_04_1000_ldl) / 12

a_hw2_02_1000_ldl = np.array([1473 + 1728 + 3603 + 2945 + 4793 + 2029 + 3935 + 5000 + 3171 + 3974 + 3190 + 2865])
a_hw2_02_1000_ldl_mean = np.mean(a_hw2_02_1000_ldl) / 12

a_hw2_01_1000_ldl = np.array([5000 + 5000 + 2677 + 2190 + 4332 + 5000 + 2826 + 3202 + 3692 + 4105 + 3175 + 2362])
a_hw2_01_1000_ldl_mean = np.mean(a_hw2_01_1000_ldl) / 12

########################################################### hw1 ##########################################################
# 50000
a_hw1_1_50000_ldl = np.array([244 + 273 + 207 + 306 + 181 + 199])
a_hw1_1_50000_ldl_mean = np.mean(a_hw1_1_50000_ldl) / 6

a_hw1_08_50000_ldl = np.array([172 + 204 + 142 + 180 + 326 + 198])
a_hw1_08_50000_ldl_mean = np.mean(a_hw1_08_50000_ldl) / 6

a_hw1_075_50000_ldl = np.array([155 + 191 + 138 + 165 + 179 + 176 + 111 + 173 + 192 + 118])
a_hw1_075_50000_ldl_mean = np.mean(a_hw1_075_50000_ldl) / 11

a_hw1_07_50000_ldl = np.array([])
a_hw1_07_50000_ldl_mean = np.mean(a_hw1_07_50000_ldl) / 8

a_hw1_05_50000_ldl = np.array([289 + 244 + 193 + 259 + 152 + 141 + 227 + 216])
a_hw1_05_50000_ldl_mean = np.mean(a_hw1_05_50000_ldl) / 8

a_hw1_04_50000_ldl = np.array([255 + 249 + 306 + 251 + 228])
a_hw1_04_50000_ldl_mean = np.mean(a_hw1_04_50000_ldl) / 5

a_hw1_025_50000_ldl = np.array([457 + 399 + 447 + 356 + 254 + 580 + 475 + 331])
a_hw1_025_50000_ldl_mean = np.mean(a_hw1_025_50000_ldl) / 8

a_hw1_02_50000_ldl = np.array([])
a_hw1_02_50000_ldl_mean = np.mean(a_hw1_02_50000_ldl) / 8

a_hw1_01_50000_ldl = np.array([533 + 766 + 585 + 627 + 779 + 568 + 1050 + 709])
a_hw1_01_50000_ldl_mean = np.mean(a_hw1_01_50000_ldl) / 8

#1000
a_hw1_0_1000_ldl = np.array([2783 + 4038 + 5000 + 5000 + 2768 + 3212 + 3974 + 5000])
a_hw1_0_1000_ldl_mean = np.mean(a_hw1_0_1000_ldl) / 8

a_hw1_1_1000_ldl = np.array([1860 + 2178 + 2250 + 5000 + 3356 + 1373 + 2245 + 1587 + 2015 + 1654 + 1343])
a_hw1_1_1000_ldl_mean = np.mean(a_hw1_1_1000_ldl) / 11

a_hw1_08_1000_ldl = np.array([1866 + 3464 + 1928 + 4458 + 1921 + 2150 + 1322 + 1986 + 2943 + 2424 + 3265 + 1810])
a_hw1_08_1000_ldl_mean = np.mean(a_hw1_08_1000_ldl) / 12

a_hw1_06_1000_ldl = np.array([4058 + 1962 + 2606 + 2649 + 3232 + 3448 + 2175 + 1921 + 3385 + 3138 + 2960 + 2513])
a_hw1_06_1000_ldl_mean = np.mean(a_hw1_06_1000_ldl) / 12

a_hw1_05_1000_ldl = np.array([2205 + 2301 + 1822 + 2184 + 1862 + 2964 + 2212 + 2013 + 2472 + 3790 + 3230 + 3417])
a_hw1_05_1000_ldl_mean = np.mean(a_hw1_05_1000_ldl) / 12

a_hw1_04_1000_ldl = np.array([2718 + 3796 + 5000 + 3440 + 4595 + 2430 + 5000 + 2783 + 4884 + 1496 + 2946 + 5000])
a_hw1_04_1000_ldl_mean = np.mean(a_hw1_04_1000_ldl) / 12

a_hw1_02_1000_ldl = np.array([2848 + 5000 + 4104 + 4165 + 3649 + 5000 + 5000 + 3838 + 4151 + 2425 + 3910 + 1838])
a_hw1_02_1000_ldl_mean = np.mean(a_hw1_02_1000_ldl) / 12

a_hw1_01_1000_ldl = np.array([1743 + 2589 + 5000 + 3695])
a_hw1_01_1000_ldl_mean = np.mean(a_hw1_01_1000_ldl) / 4

#ar  ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar
########################################################### hw2 ##########################################################
#7000
ar_hw2_08_7000_ldl = np.array([2354 + 4257 + 5000 + 4171 + 3917 + 2843 + 2805 + 3265 + 2581 + 3630 + 5000 + 5000])
ar_hw2_08_7000_ldl_mean = np.mean(ar_hw2_08_7000_ldl) / 12

ar_hw2_07_7000_ldl = np.array([1552 + 2923 + 2421 + 2255 + 3979 + 3072 + 2374])
ar_hw2_07_7000_ldl_mean = np.mean(ar_hw2_07_7000_ldl) / 7

ar_hw2_025_7000_ldl = np.array([1104 + 911 + 1175 + 1080 + 1209 + 1611 + 2793 + 1242 + 1753 + 1255 + 1419 + 1665])
ar_hw2_025_7000_ldl_mean = np.mean(ar_hw2_025_7000_ldl) / 12

ar_hw2_05_7000_ldl = np.array([1193 + 1764 + 1587 + 1746 + 1723 + 2062 + 1826 + 1576 + 1550 + 2190 + 1286 + 963])
ar_hw2_05_7000_ldl_mean = np.mean(ar_hw2_05_7000_ldl) / 12

ar_hw2_04_7000_ldl = np.array([1200 + 1489 + 889 + 1100 + 1022 + 1540 + 1227 + 2966 + 1106 + 1042 + 1617 + 1125])
ar_hw2_04_7000_ldl_mean = np.mean(ar_hw2_04_7000_ldl) / 12

ar_hw2_02_7000_ldl = np.array([1419 + 1435 + 948 + 1601 + 1679 + 1177 + 1583 + 2613 + 2057 + 1213 + 1652 + 1101])
ar_hw2_02_7000_ldl_mean = np.mean(ar_hw2_02_7000_ldl) / 12

ar_hw2_01_7000_ldl = np.array([1314 + 2043 + 1527 + 1744 + 2947 + 1552 + 1997 + 2851 + 1035 + 2090 + 1769 + 1453])
ar_hw2_01_7000_ldl_mean = np.mean(ar_hw2_01_7000_ldl) / 12
#50000
ar_hw2_1_50000_ldl = np.array([2001 + 1781 + 1468 + 2453])
ar_hw2_1_50000_ldl_mean = np.mean(ar_hw2_1_50000_ldl) / 4

ar_hw2_08_50000_ldl = np.array([1345 + 719 + 991 + 1086 + 1049 + 1292 + 712 + 785])
ar_hw2_08_50000_ldl_mean = np.mean(ar_hw2_08_50000_ldl) / 8

ar_hw2_07_50000_ldl = np.array([971 + 785 + 897 + 1114 + 1116 + 993 + 1054 + 616])
ar_hw2_07_50000_ldl_mean = np.mean(ar_hw2_07_50000_ldl) / 8

ar_hw2_05_50000_ldl = np.array([496 + 509 + 268 + 433 + 594 + 603])
ar_hw2_05_50000_ldl_mean = np.mean(ar_hw2_05_50000_ldl) / 6

ar_hw2_04_50000_ldl = np.array([519 + 442 + 308 + 305 + 512 + 388])
ar_hw2_04_50000_ldl_mean = np.mean(ar_hw2_04_50000_ldl) / 6

ar_hw2_025_50000_ldl = np.array([437 + 432 + 614 + 314 + 371 + 451 + 345 + 419])
ar_hw2_025_50000_ldl_mean = np.mean(ar_hw2_025_50000_ldl) / 8

ar_hw2_02_50000_ldl = np.array([410 + 416 + 342 + 380 + 296 + 571 + 402 + 445])
ar_hw2_02_50000_ldl_mean = np.mean(ar_hw2_02_50000_ldl) / 8

ar_hw2_01_50000_ldl = np.array([472 + 287 + 534 + 494 + 950 + 581 + 594 + 668])
ar_hw2_01_50000_ldl_mean = np.mean(ar_hw2_01_50000_ldl) / 8
########################################################### hw1 ##########################################################
#50000

ar_hw1_0_50000_ldl = np.array([647 + 636])
ar_hw1_0_50000_ldl_mean = np.mean(ar_hw1_0_50000_ldl) / 2

ar_hw1_1_50000_ldl = np.array([489 + 557 + 426 + 361])
ar_hw1_1_50000_ldl_mean = np.mean(ar_hw1_1_50000_ldl) / 4

ar_hw1_08_50000_ldl = np.array([376 + 232 + 452 + 233 + 286 + 504 + 294 + 498 + 333 + 503])
ar_hw1_08_50000_ldl_mean = np.mean(ar_hw1_08_50000_ldl) / 10

ar_hw1_06_50000_ldl = np.array([233 + 256 + 249 + 327 + 285 + 330 + 289 + 300 + 371])
ar_hw1_06_50000_ldl_mean = np.mean(ar_hw1_06_50000_ldl) / 9

ar_hw1_05_50000_ldl = np.array([275 + 359 + 394 + 414 + 276 + 396 + 424 + 406 + 454])
ar_hw1_05_50000_ldl_mean = np.mean(ar_hw1_05_50000_ldl) / 9

ar_hw1_04_50000_ldl = np.array([358 + 337 + 336 + 359 + 433 + 449 + 401 + 494 + 286])
ar_hw1_04_50000_ldl_mean = np.mean(ar_hw1_04_50000_ldl) / 9

ar_hw1_02_50000_ldl = np.array([520 + 500 + 500 + 642 + 312 + 426 + 504])
ar_hw1_02_50000_ldl_mean = np.mean(ar_hw1_02_50000_ldl) / 7

ar_hw1_01_50000_ldl = np.array([934 + 416 + 451 + 591 + 838 + 402 + 621])
ar_hw1_01_50000_ldl_mean = np.mean(ar_hw1_01_50000_ldl) / 7

#10000
ar_hw1_0_10000_ldl = np.array([1390 + 1826 + 1677 + 1477 + 1632 + 2707 + 1663 + 2230 + 1911 + 2876 + 1376 + 1499])
ar_hw1_0_10000_ldl_mean = np.mean(ar_hw1_0_10000_ldl) / 12

ar_hw1_11_10000_ldl = np.array([876 + 1205 + 5000 + 626 + 781 + 1929 + 1510 + 1603 + 1772 + 609 + 5000 + 1141 + 840])
ar_hw1_11_10000_ldl_mean = np.mean(ar_hw1_11_10000_ldl) / 12

ar_hw1_1_10000_ldl = np.array([1027 + 1457 + 744 + 930 + 1201 + 1033 + 1269 + 2178 + 1005 + 1348 + 841 + 807])
ar_hw1_1_10000_ldl_mean = np.mean(ar_hw1_1_10000_ldl) / 12

ar_hw1_08_10000_ldl = np.array([811 + 859 + 1333 + 932 + 1868 + 769 + 5000 + 1146 + 1198 + 1022])
ar_hw1_08_10000_ldl_mean = np.mean(ar_hw1_08_10000_ldl) / 10

ar_hw1_06_10000_ldl = np.array([1021 + 1405 + 774 + 743 + 1101 + 722 + 1285 + 1021 + 820 + 1628 + 967 + 945])
ar_hw1_06_10000_ldl_mean = np.mean(ar_hw1_06_10000_ldl) / 12

ar_hw1_05_10000_ldl = np.array([920 + 1371 + 861 + 933 + 814 + 955 + 1418 + 1160 + 1231 + 1280 + 705 + 1192])
ar_hw1_05_10000_ldl_mean = np.mean(ar_hw1_05_10000_ldl) / 12

ar_hw1_04_10000_ldl = np.array([1102 + 1495 + 950 + 2225 + 1310 + 1451 + 1090 + 1369 + 1023 + 2027 + 1909 + 1302])
ar_hw1_04_10000_ldl_mean = np.mean(ar_hw1_04_10000_ldl) / 12

ar_hw1_02_10000_ldl = np.array([1627 + 1452 + 1354 + 1616 + 872 + 1679 + 1269 + 954 + 1280 + 1403 + 1038 + 1861])
ar_hw1_02_10000_ldl_mean = np.mean(ar_hw1_02_10000_ldl) / 12

ar_hw1_01_10000_ldl = np.array([1208 + 1327 + 1799 + 1951 + 5000 + 1091 + 1434 + 1628 + 1509 + 864 + 1066 + 5000])
ar_hw1_01_10000_ldl_mean = np.mean(ar_hw1_01_10000_ldl) / 12

#5000
ar_hw1_0_5000_ldl = np.array([2163 + 5000 + 5000 + 3596 + 2015 + 4060 + 5000 + 3028 + 3799 + 2073 + 2658 + 1693])
ar_hw1_0_5000_ldl_mean = np.mean(ar_hw1_0_5000_ldl) / 12

ar_hw1_1_5000_ldl = np.array([1091 + 910 + 2331 + 1576 + 5000 + 1466 + 2400 + 1936 + 1875 + 2025 + 2450 + 1272])
ar_hw1_1_5000_ldl_mean = np.mean(ar_hw1_1_5000_ldl) / 12

ar_hw1_08_5000_ldl = np.array([2372 + 1711 + 2007 + 2017 + 1378 + 1838 + 2104 + 2758 + 4487 + 1896 + 1951 + 1789])
ar_hw1_08_5000_ldl_mean = np.mean(ar_hw1_08_5000_ldl) / 12

ar_hw1_06_5000_ldl = np.array([1550 + 1684 + 1415 + 1803 + 2316 + 1448 + 1284 + 1437 + 1212 + 2015 + 2158 + 2383])
ar_hw1_06_5000_ldl_mean = np.mean(ar_hw1_06_5000_ldl) / 12

ar_hw1_05_5000_ldl = np.array([1279 + 1648 + 1893 + 1339 + 2106 + 1968 + 2943 + 5000 + 2448 + 897 + 970 + 1461])
ar_hw1_05_5000_ldl_mean = np.mean(ar_hw1_05_5000_ldl) / 12

ar_hw1_04_5000_ldl = np.array([5000 + 1820 + 2195 + 1864 + 3753 + 2799 + 1914 + 2988 + 4457 + 1623 + 1876 + 5000])
ar_hw1_04_5000_ldl_mean = np.mean(ar_hw1_04_5000_ldl) / 12

ar_hw1_02_5000_ldl = np.array([2563 + 2384 + 5000 + 1322 + 5000 + 5000 + 1874])
ar_hw1_02_5000_ldl_mean = np.mean(ar_hw1_02_5000_ldl) / 7

ar_hw1_01_5000_ldl = np.array([2072 + 2642 + 5000 + 1302 + 1790 + 1894 + 3355 + 1805 + 2442 + 5000 + 2379 + 3684])
ar_hw1_01_5000_ldl_mean = np.mean(ar_hw1_01_5000_ldl) / 12
print()

"HW"
a_HW2_5000_ldl = np.array([
    1080 + 2123 + 1227 + 1424 + 2202 + 1581 + 1103 + 1105 + 5000 + 1960 + 2189 + 1943 + 1829 + 1405 + 1070 + 1696 + 1261 + 2558])
a_HW2_5000_ldl_mean = np.mean(a_HW2_5000_ldl) / 18

a_HW2_10000_ldl = np.array([
    2219 + 745 + 1208 + 1320 + 2353 + 1358 + 808 + 1743 + 1567 + 1603 + 2088 + 1191 + 1095 + 814 + 2452 + 1285 + 2578 + 1465])
a_HW2_10000_ldl_mean = np.mean(a_HW2_10000_ldl) / 18

a_HW1_5000_ldl = np.array([
    2235 + 2534 + 2478 + 2105 + 3143 + 1228 + 3494 + 2560 + 3284 + 2374 + 4001 + 2032 + 1441 + 4382 + 2799 + 2128 + 3401 + 4309])
a_HW1_5000_ldl_mean = np.mean(a_HW1_5000_ldl) / 18

print()

"CE VS CE_LOGIT"
ar_50000_ce = np.array([425 + 757 + 812 + 496 + 660 + 616 + 590 + 701 + 706 + 575])
ar_50000_ce_mean = np.mean(ar_50000_ce) / 10

ar_50000_ce_v = np.array([15 + 15 + 13 + 11 + 15 + 13 + 12 + 14 + 15 + 14])
ar_50000_ce_v_mean = np.mean(ar_50000_ce_v) / 10

ar_50000_ce_l = np.array([470 + 376 + 452 + 581 + 380 + 477 + 306 + 380 + 329 + 41])
ar_50000_ce_l_mean = np.mean(ar_50000_ce_l) / 10

ar_50000_ce_l_v = np.array([12 + 10 + 15 + 15 + 10 + 14 + 15 + 12 + 9 + 13])
ar_50000_ce_l_v_mean = np.mean(ar_50000_ce_l_v) / 10

c_50000_ce = np.array([610 + 420 + 385 + 636 + 372 + 345 + 624 + 593 + 890 + 417])
c_50000_ce_mean = np.mean(c_50000_ce) / 10
c_50000_ce_v = np.array([6 + 5 + 5 + 8 + 5 + 5 + 3 + 3 + 4 + 3])
c_50000_ce_v_mean = np.mean(c_50000_ce_v) / 10

c_50000_ce_l = np.array([235 + 363 + 307 + 378 + 245 + 301 + 243 + 257 + 234 + 183])
c_50000_ce_l_mean = np.mean(c_50000_ce_l) / 10

c_50000_ce_l_v = np.array([4 + 5 + 4 + 5 + 4 + 4 + 5 + 3 + 5 + 5])
c_50000_ce_l_v_mean = np.mean(c_50000_ce_l_v) / 10

a_50000_ce = np.array([1517 + 1088 + 711 + 654 + 1249 + 1067 + 1481 + 680 + 1191 + 724])
a_50000_ce_mean = np.mean(a_50000_ce) / 10
a_50000_ce_v = np.array([])
a_50000_ce_v_mean = np.mean(a_50000_ce_v) / 10

a_50000_ce_l = np.array([171 + 280 + 260 + 129 + 178 + 135 + 261 + 110 + 191 + 125])
a_50000_ce_l_mean = np.mean(a_50000_ce_l) / 10

a_50000_ce_l_v = np.array([])
a_50000_ce_l_v_mean = np.mean(a_50000_ce_l_v) / 10
print()

"LDL VS LDL_LOGIT"
ar_50000_ldl = np.array([555 + 597 + 393 + 597 + 379 + 461 + 762 + 417 + 634])
ar_50000_ldl_mean = np.mean(ar_50000_ldl) / 9

ar_50000_ldl_v = np.array([8 + 5 + 9 + 15 + 13 + 7 + 13 + 11 + 7])
ar_50000_ldl_v_mean = np.mean(ar_50000_ldl_v) / 9

ar_50000_ldl_l = np.array([343 + 337 + 322 + 415 + 259 + 280 + 501 + 327 + 278])
ar_50000_ldl_l_mean = np.mean(ar_50000_ldl_l) / 9

ar_50000_ldl_l_v = np.array([6 + 12 + 9 + 7 + 7 + 7 + 7 + 5 + 8])
ar_50000_ldl_l_v_mean = np.mean(ar_50000_ldl_l_v) / 9

c_50000_ldl = np.array([688 + 782 + 571 + 1009 + 512 + 566 + 806 + 902 + 532])
c_50000_ldl_mean = np.mean(c_50000_ldl) / 9
c_50000_ldl_v = np.array([4 + 9 + 2 + 10 + 4 + 9 + 9 + 3 + 5])
c_50000_ldl_v_mean = np.mean(c_50000_ldl_v) / 9

c_50000_ldl_l = np.array([355 + 285 + 329 + 538 + 323 + 253 + 286 + 346 + 378])
c_50000_ldl_l_mean = np.mean(c_50000_ldl_l) / 9

c_50000_ldl_l_v = np.array([5 + 3 + 6 + 2 + 3 + 3 + 5 + 4 + 4])
c_50000_ldl_l_v_mean = np.mean(c_50000_ldl_l_v) / 9

a_50000_ldl = np.array([621 + 1052 + 560 + 883 + 1033 + 454 + 670 + 1755 + 520])
a_50000_ldl_mean = np.mean(a_50000_ldl) / 9
a_50000_ldl_v = np.array([12 + 9 + 11 + 14 + 11 + 9 + 10 + 9 + 15])
a_50000_ldl_v_mean = np.mean(a_50000_ldl_v) / 9

a_50000_ldl_l = np.array([186 + 163 + 233 + 146 + 199 + 187 + 181 + 177 + 228])
a_50000_ldl_l_mean = np.mean(a_50000_ldl_l) / 9

a_50000_ldl_l_v = np.array([4 + 13 + 8 + 12 + 6 + 12 + 6 + 6 + 13])
a_50000_ldl_l_v_mean = np.mean(a_50000_ldl_l_v) / 9

a_10000_ldl = np.array([1134 + 709 + 1367 + 653 + 1071 + 912 + 770 + 1458 + 950])
a_10000_ldl_mean = np.mean(a_10000_ldl) / 9
a_10000_ldl_v = np.array([11 + 14 + 14 + 15 + 12 + 13 + 12 + 12 + 14])
a_10000_ldl_v_mean = np.mean(a_10000_ldl_v) / 9

a_10000_ldl_l = np.array([239 + 280 + 231 + 293 + 268 + 224 + 229 + 295])
a_10000_ldl_l_mean = np.mean(a_10000_ldl_l) / 8

a_10000_ldl_l_v = np.array([15 + 12 + 12 + 14 + 14 + 12 + 11 + 10])
a_10000_ldl_l_v_mean = np.mean(a_10000_ldl_l_v) / 8

ar_5000_ldl = np.array([])
ar_5000_ldl_mean = np.mean(ar_5000_ldl) / 8

ar_5000_ldl_v = np.array([9 + 11 + 13 + 13 + 9 + 11 + 10 + 10])
ar_5000_ldl_v_mean = np.mean(ar_5000_ldl_v) / 8

ar_5000_ldl_l = np.array([])
ar_5000_ldl_l_mean = np.mean(ar_5000_ldl_l) / 8

ar_5000_ldl_l_v = np.array([11 + 9 + 8 + 9 + 10 + 11 + 11 + 11 + 13 + 9])
ar_5000_ldl_l_v_mean = np.mean(ar_5000_ldl_l_v) / 10

ar_7000_ldl = np.array([1927 + 2256 + 1927 + 2725 + 1509 + 1924 + 2252 + 1707])
ar_7000_ldl_mean = np.mean(ar_7000_ldl) / 8

ar_7000_ldl_v = np.array([15 + 11 + 13 + 15 + 12 + 11 + 12 + 14])
ar_7000_ldl_v_mean = np.mean(ar_7000_ldl_v) / 8

ar_7000_ldl_l = np.array([1878 + 815 + 1733 + 1705 + 2503 + 1323 + 1393 + 1205])
ar_7000_ldl_l_mean = np.mean(ar_7000_ldl_l) / 8

ar_7000_ldl_l_v = np.array([9 + 13 + 11 + 9 + 9 + 13 + 9 + 9])
ar_7000_ldl_l_v_mean = np.mean(ar_7000_ldl_l_v) / 8

a_7000_ldl = np.array([])
a_7000_ldl_mean = np.mean(a_7000_ldl) / 8

a_7000_ldl_v = np.array([15 + 12 + 14 + 11 + 14 + 15 + 12 + 15])
a_7000_ldl_v_mean = np.mean(a_7000_ldl_v) / 8

a_7000_ldl_l = np.array([])
a_7000_ldl_l_mean = np.mean(a_7000_ldl_l) / 8

a_7000_ldl_l_v = np.array([9 + 11 + 14 + 11 + 13 + 9 + 11 + 13])
a_7000_ldl_l_v_mean = np.mean(a_7000_ldl_l_v) / 8

a_3000_ldl = np.array([])
a_3000_ldl_mean = np.mean(a_3000_ldl) / 8

a_3000_ldl_v = np.array([9 + 10 + 9 + 11 + 11 + 8 + 11 + 11 + 11 + 12 + 12])
a_3000_ldl_v_mean = np.mean(a_3000_ldl_v) / 11

a_3000_ldl_l = np.array([])
a_3000_ldl_l_mean = np.mean(a_3000_ldl_l) / 8

a_3000_ldl_l_v = np.array([9 + 9 + 10 + 8 + 10 + 9 + 8 + 8 + 8 + 10 + 7 + 11])
a_3000_ldl_l_v_mean = np.mean(a_3000_ldl_l_v) / 11

c_7000_ldl = np.array([1413 + 2172 + 2721 + 3537 + 2091])
c_7000_ldl_mean = np.mean(c_7000_ldl) / 5

c_7000_ldl_v = np.array([15 + 12 + 7 + 5 + 5])
c_7000_ldl_v_mean = np.mean(c_7000_ldl_v) / 8

c_7000_ldl_l = np.array([])
c_7000_ldl_l_mean = np.mean(c_7000_ldl_l) / 8

c_7000_ldl_l_v = np.array([15 + 7 + 5 + 9 + 9 + 6 + 7 + 12])
c_7000_ldl_l_v_mean = np.mean(c_7000_ldl_l_v) / 8

c_5000_ldl = np.array([2856 + 2594 + 4006 + 2761 + 2689 + 1994 + 3842 + 2349])
c_5000_ldl_mean = np.mean(c_5000_ldl) / 8

c_5000_ldl_v = np.array([6 + 6 + 5 + 5 + 12 + 12 + 6 + 8])
c_5000_ldl_v_mean = np.mean(c_5000_ldl_v) / 8

c_5000_ldl_l = np.array([4407 + 2781 + 1371 + 1934 + 1550 + 1571 + 1577 + 4619])
c_5000_ldl_l_mean = np.mean(c_5000_ldl_l) / 8

c_5000_ldl_l_v = np.array([3 + 4 + 7 + 11 + 15 + 5 + 9 + 8])
c_5000_ldl_l_v_mean = np.mean(c_5000_ldl_l_v) / 8

ar_10000_ldl = np.array([2942 + 2978 + 1912 + 2112 + 992 + 2305 + 1535 + 2946 + 1578])
ar_10000_ldl_mean = np.mean(ar_10000_ldl) / 9

ar_10000_ldl_v = np.array([11 + 8 + 14 + 13 + 8 + 10 + 9 + 14 + 7])
ar_10000_ldl_v_mean = np.mean(ar_10000_ldl_v) / 9

ar_10000_ldl_l = np.array([101])
ar_10000_ldl_l_mean = np.mean(ar_10000_ldl_l) / 8

ar_10000_ldl_l_v = np.array([9 + 9 + 12 + 13 + 13 + 14 + 9 + 8 + 9])
ar_10000_ldl_l_v_mean = np.mean(ar_10000_ldl_l_v) / 9

c_3000_ldl = np.array([])
c_3000_ldl_mean = np.mean(c_3000_ldl) / 9
c_3000_ldl_v = np.array([])
c_3000_ldl_v_mean = np.mean(c_3000_ldl_v) / 9

c_3000_ldl_l = np.array([4287 + 3327 + 5000 + 1980 + 2845 + 2019 + 5000 + 5000 + 3478 + 5000 + 5000 + 2710])
c_3000_ldl_l_mean = np.mean(c_3000_ldl_l) / 12

c_3000_ldl_l_v = np.array([])
c_3000_ldl_l_v_mean = np.mean(c_3000_ldl_l_v) / 9

c_10000_ldl = np.array([2086])
c_10000_ldl_mean = np.mean(c_10000_ldl) / 9
c_10000_ldl_v = np.array([5 + 6 + 5 + 5 + 11 + 4 + 3 + 6 + 6 + 5 + 7 + 6 + 4 + 8 + 6 + 15 + 5])
c_10000_ldl_v_mean = np.mean(c_10000_ldl_v) / 17

c_10000_ldl_l = np.array([])
c_10000_ldl_l_mean = np.mean(c_10000_ldl_l) / 12

c_10000_ldl_l_v = np.array([4 + 14 + 4 + 6 + 4 + 14 + 5 + 4 + 11 + 6 + 4 + 4 + 5 + 7 + 6 + 8 + 3])
c_10000_ldl_l_v_mean = np.mean(c_10000_ldl_l_v) / 17

ar_3000_ldl = np.array([4478 + 5000 + 3453 + 4739 + 4429 + 3772 + 3750 + 5000 + 3337 + 4012 + 4238 + 3729])
ar_3000_ldl_mean = np.mean(ar_3000_ldl) / 12
ar_3000_ldl_v = np.array([])
ar_3000_ldl_v_mean = np.mean(ar_3000_ldl_v) / 12

ar_3000_ldl_l = np.array([2560 + 2344 + 3308 + 2486 + 4500 + 2195 + 4543 + 2788 + 2745 + 3990 + 2176 + 5000])
ar_3000_ldl_l_mean = np.mean(ar_3000_ldl_l) / 12

ar_3000_ldl_l_v = np.array([])
ar_3000_ldl_l_v_mean = np.mean(ar_3000_ldl_l_v) / 12
print()

# hw1

# CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA CE-LA-PA
"c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c "
# 50000
c_50000_ce_l_01 = np.array([478 + 674 + 526 + 314 + 629 + 349 + 519 + 392 + 302 + 682 + 472 + 632])
c_50000_ce_l_01_mean = np.mean(c_50000_ce_l_01) / 12

c_50000_ce_l_02 = np.array([472 + 419 + 499 + 328 + 610 + 410 + 444 + 354 + 356 + 612 + 402 + 452])
c_50000_ce_l_02_mean = np.mean(c_50000_ce_l_02) / 12

c_50000_ce_l_04 = np.array([480 + 489 + 523 + 542 + 377 + 299 + 600 + 337 + 218 + 325 + 314 + 354])
c_50000_ce_l_04_mean = np.mean(c_50000_ce_l_04) / 12

c_50000_ce_l_05 = np.array([423 + 656 + 313 + 517 + 256 + 438 + 431 + 293 + 695 + 428 + 394 + 309])
c_50000_ce_l_05_mean = np.mean(c_50000_ce_l_05) / 12

c_50000_ce_l_06 = np.array([370 + 585 + 307 + 167 + 392 + 519 + 285 + 336 + 277 + 338 + 361 + 687])
c_50000_ce_l_06_mean = np.mean(c_50000_ce_l_06) / 12

c_50000_ce_l_08 = np.array([292 + 421 + 338 + 406 + 511 + 236 + 312 + 613 + 845 + 497 + 324 + 609])
c_50000_ce_l_08_mean = np.mean(c_50000_ce_l_08) / 12

c_50000_ce_l_11 = np.array([233 + 253 + 238 + 276 + 258 + 230 + 253 + 216 + 470 + 345 + 381 + 297])
c_50000_ce_l_11_mean = np.mean(c_50000_ce_l_11) / 12

c_50000_ce_l_12 = np.array([301 + 337 + 307 + 412 + 277 + 156 + 283 + 293 + 265 + 319 + 189 + 197])
c_50000_ce_l_12_mean = np.mean(c_50000_ce_l_12) / 12

c_50000_ce_l_15 = np.array([317 + 254 + 324 + 313 + 425 + 260 + 274 + 369 + 204 + 317 + 305 + 225])
c_50000_ce_l_15_mean = np.mean(c_50000_ce_l_15) / 12

# 10000
c_10000_ce_l_01 = np.array([])
c_10000_ce_l_01_mean = np.mean(c_10000_ce_l_01) / 12

c_10000_ce_l_02 = np.array([])
c_10000_ce_l_02_mean = np.mean(c_10000_ce_l_02) / 12

c_10000_ce_l_04 = np.array([])
c_10000_ce_l_04_mean = np.mean(c_10000_ce_l_04) / 12

c_10000_ce_l_05 = np.array([])
c_10000_ce_l_05_mean = np.mean(c_10000_ce_l_05) / 12

c_10000_ce_l_06 = np.array([])
c_10000_ce_l_06_mean = np.mean(c_10000_ce_l_06) / 12

c_10000_ce_l_08 = np.array([])
c_10000_ce_l_08_mean = np.mean(c_10000_ce_l_08) / 12

c_10000_ce_l_11 = np.array([])
c_10000_ce_l_11_mean = np.mean(c_10000_ce_l_11) / 12

c_10000_ce_l_12 = np.array([])
c_10000_ce_l_12_mean = np.mean(c_10000_ce_l_12) / 12

c_10000_ce_l_15 = np.array([])
c_10000_ce_l_15_mean = np.mean(c_10000_ce_l_15) / 12

"a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a "
# 50000
a_50000_ce_l_01 = np.array([813 + 1254 + 1199 + 738 + 1880 + 1259 + 1045 + 932 + 1719 + 801 + 495])
a_50000_ce_l_01_mean = np.mean(a_50000_ce_l_01) / 11

a_50000_ce_l_02 = np.array([999 + 1131 + 842 + 612 + 612 + 1143 + 773 + 544 + 773 + 598 + 875 + 625])
a_50000_ce_l_02_mean = np.mean(a_50000_ce_l_02) / 12

a_50000_ce_l_04 = np.array([676 + 656 + 519 + 598 + 753 + 594 + 603 + 1363 + 650 + 421 + 476 + 302])
a_50000_ce_l_04_mean = np.mean(a_50000_ce_l_04) / 12

a_50000_ce_l_05 = np.array([436 + 765 + 414 + 665 + 637 + 929 + 309 + 388 + 339 + 550 + 413 + 346])
a_50000_ce_l_05_mean = np.mean(a_50000_ce_l_05) / 12

a_50000_ce_l_06 = np.array([262 + 362 + 366 + 517 + 409 + 367 + 334 + 230 + 524 + 365 + 288 + 376])
a_50000_ce_l_06_mean = np.mean(a_50000_ce_l_06) / 12

a_50000_ce_l_08 = np.array([169 + 221 + 277 + 465 + 319 + 337 + 249 + 313 + 213 + 248 + 263 + 271])
a_50000_ce_l_08_mean = np.mean(a_50000_ce_l_08) / 12

a_50000_ce_l_09 = np.array([374 + 190 + 289 + 187 + 324 + 223 + 309 + 142 + 299 + 408 + 290 + 289])
a_50000_ce_l_09_mean = np.mean(a_50000_ce_l_09) / 12

a_50000_ce_l_11 = np.array([153 + 191 + 198 + 216 + 183 + 194 + 266 + 137 + 131 + 129 + 260 + 224])
a_50000_ce_l_11_mean = np.mean(a_50000_ce_l_11) / 12

a_50000_ce_l_12 = np.array([160 + 147 + 215 + 147 + 179 + 267 + 316 + 128 + 228 + 145 + 283 + 243])
a_50000_ce_l_12_mean = np.mean(a_50000_ce_l_12) / 12

a_50000_ce_l_15 = np.array([274 + 183 + 370 + 288 + 280 + 204 + 203 + 291 + 257 + 259 + 222 + 227])
a_50000_ce_l_15_mean = np.mean(a_50000_ce_l_15) / 12

#10000
a_10000_ce_l_01 = np.array([5000 * 12])
a_10000_ce_l_01_mean = np.mean(a_10000_ce_l_01) / 12

a_10000_ce_l_02 = np.array([2852 + 2329 + 3343 + 3809 + 2550 + 1307 + 1923 + 1856 + 1614 + 3239 + 2184 + 5000])
a_10000_ce_l_02_mean = np.mean(a_10000_ce_l_02) / 12

a_10000_ce_l_04 = np.array([2034 + 3101 + 1611 + 1648 + 4500 + 1966 + 1353 + 1605 + 1389 + 3207 + 1116 + 3042])
a_10000_ce_l_04_mean = np.mean(a_10000_ce_l_04) / 12

a_10000_ce_l_05 = np.array([1670 + 2040 + 1538 + 1638 + 1271 + 1165 + 1307 + 1402 + 1306 + 2093 + 3351 + 1117])
a_10000_ce_l_05_mean = np.mean(a_10000_ce_l_05) / 12

a_10000_ce_l_06 = np.array([1440 + 713 + 863 + 1613 + 2101 + 611 + 1281 + 1886 + 2832 + 1983 + 1317])
a_10000_ce_l_06_mean = np.mean(a_10000_ce_l_06) / 11

a_10000_ce_l_07 = np.array([850 + 750 + 1206 + 893 + 1978 + 474 + 1531 + 641 + 628 + 910 + 1330 + 1084])
a_10000_ce_l_07_mean = np.mean(a_10000_ce_l_07) / 12

a_10000_ce_l_08 = np.array([1013 + 410 + 1049 + 846 + 365 + 726 + 883 + 374 + 584 + 633 + 1889 + 577])
a_10000_ce_l_08_mean = np.mean(a_10000_ce_l_08) / 12

a_10000_ce_l_09 = np.array([555 + 317 + 511 + 654 + 504 + 933 + 266 + 360 + 614 + 893 + 425 + 305])
a_10000_ce_l_09_mean = np.mean(a_10000_ce_l_09) / 12

a_10000_ce_l_11 = np.array([])
a_10000_ce_l_11_mean = np.mean(a_10000_ce_l_11) / 12

a_10000_ce_l_12 = np.array([])
a_10000_ce_l_12_mean = np.mean(a_10000_ce_l_12) / 12

a_10000_ce_l_15 = np.array([359 + 494 + 509 + 441 + 359 + 455 + 498 + 311 + 454 + 354 + 521 + 529])
a_10000_ce_l_15_mean = np.mean(a_10000_ce_l_15) / 12

"ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar ar "
# 50000
ar_50000_ce_l_01 = np.array([])
ar_50000_ce_l_01_mean = np.mean(ar_50000_ce_l_01) / 12

ar_50000_ce_l_02 = np.array([857+549+827+659+560+604+748+644+547+465+597+708])
ar_50000_ce_l_02_mean = np.mean(ar_50000_ce_l_02) / 12

ar_50000_ce_l_04 = np.array([390+296+537+442+506+317+380+604+547+397+493+614])
ar_50000_ce_l_04_mean = np.mean(ar_50000_ce_l_04) / 12

ar_50000_ce_l_05 = np.array([512+506+375+476+325+486+342+534+423+721+510+604])
ar_50000_ce_l_05_mean = np.mean(ar_50000_ce_l_05) / 12

ar_50000_ce_l_06 = np.array([1335+336+561+403+621+431+463+757+400+415+609])
ar_50000_ce_l_06_mean = np.mean(ar_50000_ce_l_06) / 11

ar_50000_ce_l_08 = np.array([402+258+296+478+356+336+262+321+265+373+608+511])
ar_50000_ce_l_08_mean = np.mean(ar_50000_ce_l_08) / 12

ar_50000_ce_l_09 = np.array([365+393+375+745+497+407+330+351+533+454+339+393])
ar_50000_ce_l_09_mean = np.mean(ar_50000_ce_l_09) / 12

ar_50000_ce_l_11 = np.array([297+267+312+347+355+447+280+523+279+345+470+331])
ar_50000_ce_l_11_mean = np.mean(ar_50000_ce_l_11) / 12

ar_50000_ce_l_12 = np.array([268+484+347+630+297+346+486+384+359+369+285])
ar_50000_ce_l_12_mean = np.mean(ar_50000_ce_l_12) / 11

ar_50000_ce_l_15 = np.array([364+245+337+500+417+647+406+379+452+494+434+300])
ar_50000_ce_l_15_mean = np.mean(ar_50000_ce_l_15) / 12

# 10000
ar_10000_ce_l_01 = np.array([])
ar_10000_ce_l_01_mean = np.mean(ar_10000_ce_l_01) / 12

ar_10000_ce_l_02 = np.array([])
ar_10000_ce_l_02_mean = np.mean(ar_10000_ce_l_02) / 12

ar_10000_ce_l_04 = np.array([])
ar_10000_ce_l_04_mean = np.mean(ar_10000_ce_l_04) / 12

ar_10000_ce_l_05 = np.array([])
ar_10000_ce_l_05_mean = np.mean(ar_10000_ce_l_05) / 12

ar_10000_ce_l_06 = np.array([])
ar_10000_ce_l_06_mean = np.mean(ar_10000_ce_l_06) / 12

ar_10000_ce_l_08 = np.array([])
ar_10000_ce_l_08_mean = np.mean(ar_10000_ce_l_08) / 12

ar_10000_ce_l_11 = np.array([])
ar_10000_ce_l_11_mean = np.mean(ar_10000_ce_l_11) / 12

ar_10000_ce_l_12 = np.array([])
ar_10000_ce_l_12_mean = np.mean(ar_10000_ce_l_12) / 12

ar_10000_ce_l_15 = np.array([])
ar_10000_ce_l_15_mean = np.mean(ar_10000_ce_l_15) / 12
