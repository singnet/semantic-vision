import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import os, sys
import PIL
from PIL import Image
from keras.utils import to_categorical

def rotate_img(X, a):
    img = np.squeeze(X)
    img = Image.fromarray(img, mode='F')
    X_rot = img.rotate(a, resample=PIL.Image.BILINEAR)
    X_rot = np.expand_dims(X_rot, -1)
    return X_rot

def create_inputs_mnist_rot_excl(x, y):
    label = np.argmax(y)
    if (label==3) or (label==4):
        a = np.random.uniform(-45, 46)
    else:
        a = np.random.uniform(0, 360)

    xr = rotate_img(x, a)

    return xr

def create_inputs_mnist_rot_excl_range(x, y, ang_min, ang_max):
    label = np.argmax(y)
    if (label==3) or (label==4):
        a = np.random.uniform(ang_min, ang_max)
    else:
        a = np.random.uniform(0, 360)

    xr = rotate_img(x, a)

    return xr

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

def load_mnist_excluded(path = './MNIST-data'):
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
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

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



