"""
License: Apache-2.0
Author:  Maxim Peterson, max@singularitynet.io
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import capsnet_em as net
from config import cfg, get_coord_add

from PIL import Image
import numpy as np
import os, time, sys
import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_SIZE = 28

sx = 28
sy = 28
sc = 1

IMG_DIM    = (sy, sx, sc)

NCLASSES = 8
BATCH_SIZE = cfg.batch_size

#------- Test options -------------

n_tests = 2

ang_min = -45
ang_max = 46

#ang_min = 46
#ang_max = 315

#ang_min = 180-45
#ang_max = 180+46

#is_only_3_and_4 = True
is_only_3_and_4 = False

#----------------------------------------


ERASE_LINE = '\x1b[2K'

dataset_name = 'mnist'
model_name = 'caps'

def test_model(n_tests, x_test, y_test, ang_min, ang_max):

    # Placeholders for input data and the targets
    x_input = tf.placeholder(tf.float32, (None, *IMG_DIM), name='Input')
    y_target = tf.placeholder(tf.int32, [None, ], name='Target')

    coord_add = get_coord_add(dataset_name )
    sample_batch = tf.identity(x_input)
    batch_labels = tf.identity(y_target)
    batch_x = slim.batch_norm(sample_batch, center=False, is_training=False, trainable=False)
    output, pose_out = net.build_arch(batch_x, coord_add, is_train=True,
                                      num_classes=NCLASSES)
    batch_acc_sum = net.test_accuracy_sum(output, batch_labels)
    batch_pred = net.test_predict(output, batch_labels)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    model_path = cfg.logdir + '/caps/mnist'
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    nImg = x_test.shape[0]
    batch_size = int(cfg.batch_size)
    nBatches = int(nImg / batch_size)

    accuraces = []

    mean_acc = 0
    for n in range(n_tests):
        print('\nTest %d/%d' % (n + 1, n_tests))

        print('-' * 30 + 'Begin: testing' + '-' * 30)
        acc = 0
        k = 0
        xi = np.empty([1, sy, sx, 1])
        x_init = np.empty([1, sy, sx, 1])

        for i in range(nBatches):
            x = x_test[i * batch_size: (i + 1) * batch_size, :, :, :]
            y = y_test[i * batch_size: (i + 1) * batch_size]
            xr = np.empty(x.shape)
            for j in range(x.shape[0]):
                xr[j, :, :, :] = utils.create_inputs_mnist_rot_excl_range(x[j, :, :, :], y[j],
                                                                          ang_min, ang_max)

                k += 1

            batch_acc_v = sess.run(batch_acc_sum, feed_dict={x_input: xr, y_target: y})
            acc += batch_acc_v

            # Just checking what images we are feeding to the network
            if i == 0 and n == 0:
                for j in range(batch_size):
                    if j == 0:
                        xi[0, :, :, :] = xr[0, :, :, :]
                        x_init[0, :, :, :] = x[0, :, :, :]
                    else:
                        xi = np.concatenate([xi, np.expand_dims(xr[j, :, :, :], 0)])
                        x_init = np.concatenate([x_init, np.expand_dims(x[j, :, :, :],0)])
                    # xr = np.concatenate([xr, x_recon])
                    if j == (batch_size - 1):
                        images = utils.combine_images(xi)
                        image = images
                        Image.fromarray(image.astype(np.uint8)).save(cfg.logdir + "/batch_rot.png")

                        images = utils.combine_images(x_init)
                        image = images
                        Image.fromarray(image.astype(np.uint8)).save(cfg.logdir + "/batch_init.png")

            sys.stdout.write(ERASE_LINE)
            sys.stdout.write("\r \r {0}%".format(int(100 * k / nImg)))
            sys.stdout.flush()
            time.sleep(0.001)


        x = x_test[k:, :, :, :]
        y = y_test[k:]

        # duplicate the last sample to adjust the batch size
        n_left = nImg-k
        n_tile = BATCH_SIZE - n_left

        x_tile = np.tile(np.expand_dims(x_test[nImg-1, :, :, :],0), [n_tile, 1, 1, 1])
        y_tile = np.tile(y_test[nImg-1], n_tile)

        x = np.concatenate( (x, x_tile) )
        y = np.concatenate((y, y_tile))

        xr = np.empty(x.shape)
        for j in range(x.shape[0]):
            xr[j, :, :, :] = utils.create_inputs_mnist_rot_excl_range(x[j, :, :, :], y[j],
                                                                      ang_min, ang_max)

        batch_pred_v = sess.run(batch_pred, feed_dict={x_input: xr, y_target: y})
        left_pred = np.asarray(batch_pred_v[:n_left], dtype=np.float32)

        acc += np.sum(left_pred)

        k += n_left

        sys.stdout.write(ERASE_LINE)
        sys.stdout.write("\r \r {0}%".format(str(100)))
        sys.stdout.flush()
        time.sleep(0.001)

        print('\n')
        print('-' * 30 + 'End: testing' + '-' * 30)

        acc_aver = acc / float(y_test.shape[0])

        print('Number of images: {}, Accuracy: {}'.format(k, acc_aver))

        mean_acc += acc_aver
        accuraces.append(acc_aver)

    mean_acc = mean_acc / float(n_tests)

    var_acc = 0
    accuraces = np.array(accuraces)
    for i in range(accuraces.shape[0]):
        var_acc += (accuraces[i] - mean_acc)*(accuraces[i] - mean_acc)

    var_acc /= float(n_tests)

    print('\nTesting is finished!')
    print('Testing options:\nAngles range from {} to {}\tIs only 3 and 4: {}'.format(ang_min, ang_max, is_only_3_and_4))
    print('\nMean testing accuracy for {} runs: {}'.format(n_tests, mean_acc))
    print('Variance of testing accuracy for {} runs: {}'.format(n_tests, var_acc))



# Prepare data
_, (x_test_all, y_test_all) = utils.load_mnist_excluded()

if not is_only_3_and_4:
    test_model(n_tests, x_test_all, y_test_all, ang_min, ang_max)

else:
    nImg = x_test_all.shape[0]
    x_test = np.empty([1, sy, sx, 1], dtype=np.float32)
    y_test = np.empty(1, dtype=np.int32)
    k = 0
    for i in range(nImg):
        y_i = y_test_all[i]
        if (y_i == 3) or (y_i == 4):
            if k == 0:
                x_test[0, :, :, :] = x_test_all[i, :, :, :]
                y_test[0] = y_test_all[i]
            else:
                x_test = np.concatenate([x_test, np.expand_dims(x_test_all[i, :, :, :], 0)])
                yti = np.expand_dims(y_test_all[i], 0)
                y_test = np.concatenate((y_test, yti))

            k += 1

    test_model(n_tests, x_test, y_test, ang_min, ang_max)

