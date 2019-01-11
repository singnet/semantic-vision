import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import h5py
import numpy as np
import random
import plot_funs
import time
import matplotlib.pyplot as plt

OUTPUT_DIM = 4*4*128

TARGET_H5 = '/mnt/fileserver/shared/datasets/vg_selected_4decode_sc_128x128.h5'

ITERS = 50

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

def getBatches(dataset):
    random.shuffle(dataset)
    data_len = int(len(dataset)) / int(BATCH_SIZE)
    features_batches = []
    bb_batches = []
    for i in range(data_len):
        temp_batch = dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        temp_f = []
        temp_bb = []
        for record in temp_batch:
            temp_f.append(record[1])
            temp_bb.append(record[0])
        features_batches.append(temp_f)
        bb_batches.append(temp_bb)
    return bb_batches, features_batches

def load_data(th5, t, c):
    with h5py.File(th5, 'r') as f:
        imgs = np.array(f[t + '_images_' + c]) / 255.0
        features = np.array(f[t + '_features'])
    return imgs, features


def e(x):
    if hasattr(e, 'reuse'):
        e.reuse = True
    else:
        e.reuse = False
    c1 = tf.layers.conv2d(
        x, filters=60, kernel_size=[3, 3], padding='valid',
        name='conv1', reuse=e.reuse
    )
    bn1 = tf.layers.batch_normalization(c1, name='bn1', reuse=e.reuse)
    a1 = tf.nn.relu(bn1)
    p1 = tf.layers.max_pooling2d(
        a1, pool_size=[2, 2], strides=[2, 2], padding='valid'
    )

    c2 = tf.layers.conv2d(
        p1, filters=120, kernel_size=[3, 3], padding='valid',
        name='conv2', reuse=e.reuse
    )
    bn2 = tf.layers.batch_normalization(c2, name='bn2', reuse=e.reuse)
    a2 = tf.nn.relu(bn2)
    p2 = tf.layers.max_pooling2d(
        a2, pool_size=[2, 2], strides=[2, 2], padding='valid'
    )

    c3 = tf.layers.conv2d(
        p2, filters=250, kernel_size=[3, 3], padding='valid',
        name='conv3', reuse=e.reuse
    )
    bn3 = tf.layers.batch_normalization(c3, name='bn3', reuse=e.reuse)
    a3 = tf.nn.relu(bn3)
    p3 = tf.layers.max_pooling2d(
        a3, pool_size=[2, 2], strides=[2, 2], padding='valid')
    flat1 = tf.layers.flatten(p3)
    d1 = tf.layers.dense(flat1, 1000, name='dense1', reuse=e.reuse)
    a4 = tf.nn.relu(d1)
    d2 = tf.layers.dense(a4, 5, name='image_z', reuse=e.reuse)

    return d2


def d(z, w_control):
    if hasattr(d, 'reuse'):
        d.reuse = True
    else:
        d.reuse = False
    d1 = tf.layers.dense(z, 4 * 4 * 128, name='d_dense1', reuse=e.reuse)
    a1 = tf.nn.relu(d1)
    a1 = tf.einsum('bi,bij->bj', a1, w_control)

    spatial_z = tf.reshape(a1, [tf.shape(a1)[0], 4, 4, 128])

    c1 = tf.layers.conv2d(
        spatial_z, filters=200, kernel_size=[3, 3], padding='same',
        name='d_conv1', reuse=d.reuse
    )
    bn1 = tf.layers.batch_normalization(c1, name='d_bn1', reuse=d.reuse)
    a2 = tf.nn.relu(bn1)
    ups1 = tf.keras.layers.UpSampling2D((2, 2))(a2)

    c2 = tf.layers.conv2d(
        ups1, filters=100, kernel_size=[3, 3], padding='same',
        name='d_conv2', reuse=d.reuse
    )
    bn2 = tf.layers.batch_normalization(c2, name='d_bn2', reuse=d.reuse)
    a2 = tf.nn.relu(bn2)
    ups2 = tf.keras.layers.UpSampling2D((2, 2))(a2)

    c3 = tf.layers.conv2d(
        ups2, filters=50, kernel_size=[3, 3], padding='same',
        name='d_conv3', reuse=d.reuse
    )
    bn3 = tf.layers.batch_normalization(c3, name='d_bn3', reuse=d.reuse)
    a3 = tf.nn.relu(bn3)
    ups3 = tf.keras.layers.UpSampling2D((8, 8))(a3)

    c4 = tf.layers.conv2d(
        ups3, filters=3, kernel_size=[3, 3], padding='same',
        name='d_conv4', reuse=d.reuse
    )
    a4 = tf.nn.sigmoid(c4)
    return a4

bounding_boxes = tf.placeholder(tf.float32, [None, 128, 128, 3])
rcnn_features = tf.placeholder(tf.float32, [None, 2048])


encoder_features = e(bounding_boxes)

W = tf.constant(np.ones((OUTPUT_DIM, OUTPUT_DIM)), dtype=tf.float32, name="W")
#W = tf.get_variable("W",(OUTPUT_DIM, OUTPUT_DIM))
W = tf.reshape(W, (1, OUTPUT_DIM, OUTPUT_DIM))
W = tf.tile(W, [tf.shape(bounding_boxes)[0], 1, 1])

W_control = tf.layers.dense(encoder_features, 64, activation=tf.nn.relu)
W_control = tf.layers.dense(W_control, OUTPUT_DIM * OUTPUT_DIM, activation=None)
W_control = tf.reshape(W_control, [-1, OUTPUT_DIM, OUTPUT_DIM])

W_rez = W * W_control

pr = d(rcnn_features, W_rez)

r_cost = tf.losses.mean_squared_error(pr, bounding_boxes)

r_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(r_cost)

bbs, features = load_data(TARGET_H5, 'train', 'c')
train_dataset = list(zip(bbs, features))
bbs, features = load_data(TARGET_H5, 'test', 'c')
test_dataset = list(zip(bbs, features))

saver = tf.train.Saver(max_to_keep=20)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

f_train_stat = open("train_log.txt", "w", buffering=1)
f_test_stat = open("test_log.txt", "w", buffering=1)

os.system("mkdir -p figs_rec")

for it in range(ITERS):
    train_bb_batches, train_ft_batches = getBatches(train_dataset)
    start_time = time.time()
    cost = 0
    for i in range(len(train_bb_batches)):
        bb = train_bb_batches[i]
        feat = train_ft_batches[i]
        r_cost_rez, _, res = sess.run([r_cost, r_train_op, pr], feed_dict={bounding_boxes: bb, rcnn_features: feat})
        cost += r_cost_rez

    f_train_stat.write("%i %g\n" % (it, float(cost)/float(len(train_bb_batches))))

    if ((it+1)%10) == 0:
        test_bb_batches, test_ft_batches = getBatches(test_dataset)
        r_cost_rez, samples_test = sess.run([r_cost, pr], feed_dict={bounding_boxes: test_bb_batches[0], rcnn_features: test_ft_batches[0]})
        plot_funs.plot_pair_samples(test_bb_batches[0], samples_test, 'figs_rec/samples_%.6i_unseen.png' % (it))
        f_test_stat.write("%i %g\n" % (it, r_cost_rez))

        train_bb_batches, train_ft_batches = getBatches(train_dataset)
        r_cost_train, samples_train = sess.run([r_cost, pr], feed_dict={bounding_boxes: train_bb_batches[0], rcnn_features: train_ft_batches[0]})
        plot_funs.plot_pair_samples(train_bb_batches[0], samples_train, 'figs_rec/samples_%.6i_seen.png' % (it))

    print("Iteration time:")
    print(it, (time.time() - start_time))

saver.save(sess, 'save/final-model')
