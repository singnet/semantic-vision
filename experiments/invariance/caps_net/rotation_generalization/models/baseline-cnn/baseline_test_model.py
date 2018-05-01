import tensorflow as tf
from PIL import Image
import numpy as np
import os, time, sys, errno
import utils

SAVEDIR = './models/baseline_01/model'
MODEL = 'final-model'


# Target size
sx = 28
sy = 28
sc = 1

NCLASSES = 8

#---------- Test options ----------------
BATCH_SIZE = 120
n_tests = 10

ang_min = -45
ang_max = 46

ang_min = 46
ang_max = 315

ang_min = 180-45
ang_max = 180+46

is_only_3_and_4 = True
#is_only_3_and_4 = False
#-----------------------------------


if not os.path.exists(SAVEDIR):
    print('Error: directory ' + SAVEDIR + " doesn't exist!")
    quit()

ERASE_LINE = '\x1b[2K'


def test_model(n_tests, x_test, y_test, ang_min, ang_max):
    sess = tf.Session()

    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(SAVEDIR + '/' + MODEL + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(SAVEDIR))
    graph = tf.get_default_graph()

    phase_train = graph.get_tensor_by_name('phase_train:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    input = graph.get_tensor_by_name('Basenet/input:0')
    output = graph.get_tensor_by_name('Basenet/output:0')

    nImg = x_test.shape[0]
    batch_size = int(BATCH_SIZE)
    nBatches = int(nImg / batch_size)

    accuraces = []

    mean_acc = 0
    for n in range(n_tests):
        # test_rotated(model=eval_model, data=(x_test, y_test), args=args)
        print('\nTest %d/%d' % (n + 1, n_tests))

        print('-' * 30 + 'Begin: testing rotated all' + '-' * 30)
        acc = 0
        k = 0
        xi = np.empty([1, sy, sx, 1])
        x_init = np.empty([1, sy, sx, 1])

        for i in range(nBatches):
            x = x_test[i * batch_size: (i + 1) * batch_size, :, :, :]
            y = y_test[i * batch_size: (i + 1) * batch_size, :]
            xr = np.empty(x.shape)
            for j in range(x.shape[0]):
                xr[j, :, :, :] = utils.create_inputs_mnist_rot_excl_range(x[j, :, :, :], y[j, :],
                                                                          ang_min, ang_max)
                k += 1

            y_pred = sess.run(output, feed_dict={input: xr,
                                                 phase_train: False,
                                                 keep_prob: 1.0})
            acc += np.sum(np.argmax(y_pred, 1) == np.argmax(y, 1))

            # Just checking what images are we feeding to the network
            if i == 0 and n == 0:
                for j in range(batch_size):
                    if j == 0:
                        xi[0, :, :, :] = xr[0, :, :, :]
                        x_init[0, :, :, :] = x[0, :, :, :]
                    else:
                        xi = np.concatenate([xi, np.expand_dims(xr[j, :, :, :], 0)])
                        x_init = np.concatenate([x_init, np.expand_dims(x[j, :, :, :],0)])

                    if j == (batch_size - 1):
                        images = utils.combine_images(xi)
                        image = images * 255
                        Image.fromarray(image.astype(np.uint8)).save("./batch_rot.png")

                        images = utils.combine_images(x_init)
                        image = images * 255
                        Image.fromarray(image.astype(np.uint8)).save("./batch_init.png")

            sys.stdout.write(ERASE_LINE)
            sys.stdout.write("\r \r {0}%".format(str(100 * k / nImg)))
            sys.stdout.flush()
            time.sleep(0.001)

        x = x_test[k:, :, :, :]
        y = y_test[k:, :]
        xr = np.empty(x.shape)
        for j in range(x.shape[0]):
            xr[j, :, :, :] = utils.create_inputs_mnist_rot_excl_range(x[j, :, :, :], y[j, :],
                                                                      ang_min, ang_max)

        y_pred = sess.run(output, feed_dict={input: xr,
                                             phase_train: False,
                                             keep_prob: 1.0
                                             })
        acc += np.sum(np.argmax(y_pred, 1) == np.argmax(y, 1))

        k += x.shape[0]

        sys.stdout.write(ERASE_LINE)
        sys.stdout.write("\r \r {0}%".format(str(100)))
        sys.stdout.flush()
        time.sleep(0.001)

        print('\n')
        print('-' * 30 + 'End: testing rotated all' + '-' * 30)

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

    print('\nModel: {}'.format(MODEL))
    print('Mean testing accuracy for {} runs: {}'.format(n_tests, mean_acc))
    print('Variance of testing accuracy for {} runs: {}'.format(n_tests, var_acc))



# Prepare data
_, (x_test_all, y_test_all) = utils.load_mnist_excluded()

if not is_only_3_and_4:
    test_model(n_tests, x_test_all, y_test_all, ang_min, ang_max)

else:
    nImg = x_test_all.shape[0]
    x_test = np.empty([1, sy, sx, 1], dtype=np.float32)
    y_test = np.empty([1, NCLASSES])
    k = 0
    for i in range(nImg):
        y_i = y_test_all[i, :]
        y_i = np.argmax(y_i)
        if (y_i == 3) or (y_i == 4):
            if k == 0:
                x_test[0, :, :, :] = x_test_all[i, :, :, :]
                y_test[0, :] = y_test_all[i, :]
            else:
                x_test = np.concatenate([x_test, np.expand_dims(x_test_all[i, :, :, :], 0)])
                y_test = np.concatenate([y_test, np.expand_dims(y_test_all[i, :], 0)])

            k += 1

    test_model(n_tests, x_test, y_test, ang_min, ang_max)




