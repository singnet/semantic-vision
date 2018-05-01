"""
Testing CapsNet with dynamic routing on the rotated MNIST
Author: Maxim Peterson, max@singularitynet.io
"""
import numpy as np
import PIL
from PIL import Image
import utils

import sys, time, os

sx = 28
sy = 28

ERASE_LINE = '\x1b[2K'


def test_model(model, n_tests, x_test, y_test, batch_size, ang_min, ang_max):

    nImg = x_test.shape[0]
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
            y = y_test[i * batch_size: (i + 1) * batch_size, :]
            xr = np.empty(x.shape)
            for j in range(x.shape[0]):
                xr[j, :, :, :] = utils.create_inputs_mnist_rot_excl_range(x[j, :, :, :], y[j, :],
                                                                          ang_min, ang_max)
                k += 1

            y_pred, _ = model.predict(xr)

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
                    # xr = np.concatenate([xr, x_recon])
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

        y_pred, _ = model.predict(xr)
        acc += np.sum(np.argmax(y_pred, 1) == np.argmax(y, 1))

        k += x.shape[0]

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

    print('Mean testing accuracy for {} runs: {}'.format(n_tests, mean_acc))
    print('Variance of testing accuracy for {} runs: {}'.format(n_tests, var_acc))
