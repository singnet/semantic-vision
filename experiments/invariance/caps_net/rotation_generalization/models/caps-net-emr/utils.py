import os
import numpy as np
import tensorflow as tf

from config import cfg

import PIL
from PIL import Image
import math

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

def create_inputs_mnist(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)

def create_inputs_mnist_rot_excl(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    s = tr_x.shape

    if is_train:
        tr_x = tf.convert_to_tensor(tr_x, np.float32)
        tr_y = tf.convert_to_tensor(tr_y, np.int32)

        d6 = tf.fill([1, s[0]], 6)
        d9 = tf.fill([1, s[0]], 9)
        mask_d6 = tf.equal(tr_y, d6)
        mask_d9 = tf.equal(tr_y, d9)
        mask = tf.logical_or(mask_d6, mask_d9)
        mask = tf.squeeze(tf.transpose(mask))
        mask = tf.logical_not(mask)
        tr_y = tf.boolean_mask(tr_y, mask)
        tr_x = tf.boolean_mask(tr_x, mask)


        # tr_x, tr_y = load_mnist_rotated_exclude(cfg.dataset, is_train)
        data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
        x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                      min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

        v3 = tf.fill([1, cfg.batch_size], 3)
        v4 = tf.fill([1, cfg.batch_size], 4)

        mask_v3 = tf.equal(y, v3)
        mask_v4 = tf.equal(y, v4)
        mask = tf.logical_or(mask_v3, mask_v4)
        mask = tf.squeeze(tf.transpose(mask))
        mask_not = tf.logical_not(mask)

        img_excl = tf.boolean_mask(x, mask)
        img_left = tf.boolean_mask(x, mask_not)

        y_excl = tf.boolean_mask(y, mask)
        y_left = tf.boolean_mask(y, mask_not)

        ang_full = tf.random_uniform([cfg.batch_size], minval=0, maxval=2*math.pi)
        ang_full = tf.boolean_mask(ang_full, mask_not)

        ang_excl = tf.random_uniform([cfg.batch_size], minval=-math.pi/4, maxval=math.pi/4)
        ang_excl = tf.boolean_mask(ang_excl, mask)

        img_excl = tf.contrib.image.rotate(img_excl, ang_excl, interpolation='BILINEAR')
        img_left = tf.contrib.image.rotate(img_left, ang_full, interpolation='BILINEAR')

        x = tf.concat([img_left, img_excl], axis=0)
        y = tf.concat([y_left, y_excl], axis=0)

    else:
        tr_x = tf.convert_to_tensor(tr_x, np.float32)
        tr_y = tf.convert_to_tensor(tr_y, np.int32)

        d6 = tf.fill([1, s[0]], 6)
        d9 = tf.fill([1, s[0]], 9)
        mask_d6 = tf.equal(tr_y, d6)
        mask_d9 = tf.equal(tr_y, d9)
        mask = tf.logical_or(mask_d6, mask_d9)
        mask = tf.squeeze(tf.transpose(mask))
        mask = tf.logical_not(mask)
        tr_y = tf.boolean_mask(tr_y, mask)
        tr_x = tf.boolean_mask(tr_x, mask)

        # tr_x, tr_y = load_mnist_rotated_exclude(cfg.dataset, is_train)
        data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
        x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size,
                                      capacity=cfg.batch_size * 64,
                                      min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

        v3 = tf.fill([1, cfg.batch_size], 3)
        v4 = tf.fill([1, cfg.batch_size], 4)

        mask_v3 = tf.equal(y, v3)
        mask_v4 = tf.equal(y, v4)
        mask = tf.logical_or(mask_v3, mask_v4)
        mask = tf.squeeze(tf.transpose(mask))
        mask_not = tf.logical_not(mask)

        img_excl = tf.boolean_mask(x, mask)
        img_left = tf.boolean_mask(x, mask_not)

        y_excl = tf.boolean_mask(y, mask)
        y_left = tf.boolean_mask(y, mask_not)

        ang_full = tf.random_uniform([cfg.batch_size], minval=0, maxval=2 * math.pi)
        ang_full = tf.boolean_mask(ang_full, mask_not)

        # ang_excl = tf.random_uniform([cfg.batch_size], minval=-math.pi / 4, maxval=math.pi / 4)
        # ang_excl = tf.random_uniform([cfg.batch_size], minval=math.pi/4+0.1,
        #                              maxval=2*math.pi-math.pi/4-0.1)
        ang_excl = tf.random_uniform([cfg.batch_size], minval=math.pi -math.pi/4,
                                     maxval=math.pi + math.pi/4)
        ang_excl = tf.boolean_mask(ang_excl, mask)

        img_excl = tf.contrib.image.rotate(img_excl, ang_excl, interpolation='BILINEAR')
        img_left = tf.contrib.image.rotate(img_left, ang_full, interpolation='BILINEAR')

        x = tf.concat([img_left, img_excl], axis=0)
        y = tf.concat([y_left, y_excl], axis=0)

    return (x, y)


def rotate_img(X, a):
    X_rot = np.zeros(X.shape)
    img = np.squeeze(X)
    img = Image.fromarray(img, mode='F')
    img = img.rotate(a, resample=PIL.Image.BILINEAR)
    # img = np.clip(np.array(img) - 1, -1, 1)
    X_rot = np.expand_dims(img, -1)

    return X_rot

def load_mnist_local(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    x_train = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    y_train = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    x_test = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    y_test = loaded[8:].reshape((10000)).astype(np.int32)

    return (x_train, y_train), (x_test, y_test)

def create_inputs_mnist_rot_excl_(x, y):
    # label = np.argmax(y)
    label = y
    if (label==3) or (label==4):
        a = np.random.uniform(-45, 46)
    else:
        a = np.random.uniform(0, 360)

    xr = rotate_img(x, a)

    return xr

def create_inputs_mnist_rot_excl_range(x, y, ang_min, ang_max):
    #label = np.argmax(y)
    label = y
    if (label==3) or (label==4):
        a = np.random.uniform(ang_min, ang_max)
    else:
        a = np.random.uniform(0, 360)

    xr = rotate_img(x, a)

    return xr


def get_random_mnist_batch(x, y, batch_size):
    batch_x = np.zeros([batch_size, x.shape[1], x.shape[2], x.shape[3]])
    batch_y = np.zeros([batch_size])
    for k in range(batch_size):
        i = int(np.random.uniform(0, x.shape[0]))
        xi = x[i, :, :, :]
        yi = y[i]
        batch_x[k, :, :, :] = create_inputs_mnist_rot_excl_(xi, yi)
        batch_y[k] = yi

    return batch_x, batch_y

def load_mnist_excluded(path = './data/mnist'):
    # the data, shuffled and split between train and test sets
    # from keras.datasets import mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()


    (x_train, y_train), (x_test, y_test) = load_mnist_local(path)

    v6_train = np.arange(y_train.size, dtype=np.int32)
    v6_train.fill(6)
    v9_train = np.arange(y_train.size, dtype=np.int32)
    v9_train.fill(9)
    v6_test = np.arange(y_test.size, dtype=np.int32)
    v6_test.fill(6)
    v9_test = np.arange(y_test.size, dtype=np.int32)
    v9_test.fill(9)

    mask_v6_train = np.equal(y_train, v6_train)
    mask_v9_train = np.equal(y_train, v9_train)
    mask_train = np.logical_or(mask_v6_train, mask_v9_train)
    mask_train = np.logical_not(mask_train)

    mask_v6_test= np.equal(y_test, v6_test)
    mask_v9_test = np.equal(y_test, v9_test)
    mask_test = np.logical_or(mask_v6_test, mask_v9_test)
    mask_test = np.logical_not(mask_test)

    x_train = x_train[mask_train]
    y_train = y_train[mask_train]

    x_test = x_test[mask_test]
    y_test = y_test[mask_test]

    # For training we need labels in range [0, N), thus we need to relabel images 7 and 8, because 6 and 9 were deleted
    for i in range(y_train.shape[0]):
        if y_train[i]==7:
            y_train[i] = 6
        if y_train[i]==8:
            y_train[i] = 7

    for i in range(y_test.shape[0]):
        if y_test[i]==7:
            y_test[i] = 6
        if y_test[i]==8:
            y_test[i] = 7

    # x_train = (x_train.reshape(-1, 28, 28, 1).astype('float32') -127.5)/ 127.5
    # x_test = (x_test.reshape(-1, 28, 28, 1).astype('float32') -127.5)/ 127.5
    # x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # y_train = to_categorical(y_train.astype('float32'))
    # y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_mnist(path, is_training):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    # trX = tf.convert_to_tensor(trX, tf.float32)
    # teX = tf.convert_to_tensor(teX, tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image
