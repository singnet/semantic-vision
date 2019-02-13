import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist
from tensorflow.contrib.layers import conv2d, conv2d_transpose, layer_norm, fully_connected

import random
import save_images
import time

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings

import seaborn as sns
from tensorflow.python.ops.parallel_for import gradients
from sklearn.decomposition import PCA

from scipy.stats import wasserstein_distance
from scipy.stats import multivariate_normal

BATCH_SIZE = 128
IMG_DIM    = (28, 28, 1)
Z_DIM      = 80
ZC_DIM     = 0
ZU_DIM     = 5

OUTPUT_DIM = int(np.prod(IMG_DIM))
LAMBDA     = 10
ITERS      = 200000
CRITIC_ITER= 5

# leaky relu alpha
leakyrelu_alpha    = 0.1

def lrelu(x):
	return tf.nn.relu(x) - leakyrelu_alpha * tf.nn.relu(-x)

""" Model Definitions """

def generator_tf(x, reuse=True):
	with tf.variable_scope("Generator", reuse=reuse):
		x = tf.identity(x, name="input")
		x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
		x = tf.layers.dense(x, 7 * 7 * 128, activation=tf.nn.relu)
		x = tf.reshape(x, [-1, 7, 7, 128])
		x = tf.layers.conv2d_transpose(x, 64, 4, 2, padding='same', activation=tf.nn.relu)
		x = tf.layers.conv2d_transpose(x, 1, 4, 2, padding='same', activation=tf.nn.tanh)
		x = tf.identity(x, name="output")

		return x


def d_tf(x, reuse=True):
	with tf.variable_scope("Discriminator", reuse=reuse):
		x = tf.identity(x, name="input")
		x = tf.layers.conv2d(x, 64, 4, 2, padding='same', activation=lrelu)
		x = tf.layers.conv2d(x, 128, 4, 2, padding='same', activation=lrelu)
		x = tf.contrib.layers.flatten(x)
		x = tf.layers.dense(x, 1024, activation=lrelu)
		x = tf.layers.dense(x, 1)
		x = tf.identity(x, name="output")
		return x


def q_tf(x, reuse=True):
	with tf.variable_scope("Q", reuse=reuse):
		x = tf.identity(x, name="input")
		x = tf.layers.conv2d(x, 64, 4, 2, padding='same', activation=lrelu)
		x = tf.layers.conv2d(x, 128, 4, 2, padding='same', activation=lrelu)
		x = tf.contrib.layers.flatten(x)
		x = tf.layers.dense(x, 1024, activation=lrelu)
		x = fully_connected(x, ZC_DIM+ZU_DIM, activation_fn=None)
		x = tf.identity(x, name="output")
		return x


def q_cost_tf(z, q):
	z_uni = z[:, : ZU_DIM]
	q_uni = q[:, :  ZU_DIM]

	luni = 0.5 * tf.square(z_uni - q_uni)

	return tf.reduce_mean(luni)


def prepare_mnist_list(X):
	X = (X.astype(np.float32) - 127.5) / 127.5
	X = X[:, :, :, None]
	return list(X)


def random_uc():
	idxs = np.random.randint(ZC_DIM, size=BATCH_SIZE)
	onehot = np.zeros((BATCH_SIZE, ZC_DIM))
	onehot[np.arange(BATCH_SIZE), idxs] = 1
	return onehot


def random_z():
    rez = np.zeros([BATCH_SIZE, Z_DIM])
    rez[:,  : ]       = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM - ZC_DIM))
    return rez


def getZandRec(sess, X):
	data = np.array(random.sample(X, BATCH_SIZE))
	q_z = sess.run(q_on_real_data, feed_dict={real_data: data})
	z = random_z()
	z[:, ZC_DIM:ZC_DIM+ZU_DIM] = q_z[:, ZC_DIM:ZC_DIM+ZU_DIM]
	c_xr = sess.run(fake_data, feed_dict={z_ph: z})
	rec_loss = (np.square(np.reshape(data, (BATCH_SIZE, -1)) - np.reshape(c_xr, (BATCH_SIZE, -1)))).mean(axis=1)
	return rec_loss, data, z[:, ZC_DIM:ZC_DIM+ZU_DIM]

def getZandRec_best(sess, X):
	data = np.array(X)
	q_z = sess.run(q_on_real_data, feed_dict={real_data: data})
	best_loss = 1000
	best_z = random_z()
	best_r = 0
	for i in range(100):
		z = random_z()
		z[:, ZC_DIM:ZC_DIM+ZU_DIM] = q_z[:, ZC_DIM:ZC_DIM+ZU_DIM]
		buf_xr = sess.run(fake_data, feed_dict={z_ph: z})
		rec_loss = (np.square(np.reshape(data, (BATCH_SIZE, -1)) - np.reshape(buf_xr, (BATCH_SIZE, -1)))).mean(axis=1).sum()/float(BATCH_SIZE)
		if(best_loss > rec_loss):
			best_r = buf_xr
			best_loss = rec_loss
			best_z = z
	return best_r, data, best_z[:, ZC_DIM:ZC_DIM+ZU_DIM]

# Prepare Training Data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train_list = prepare_mnist_list(X_train)

idx_train_1 = np.squeeze(np.argwhere(Y_train == 1))
idx_train_2 = np.squeeze(np.argwhere(Y_train == 2))
idx_train_3 = np.squeeze(np.argwhere(Y_train == 3))
idx_train_4 = np.squeeze(np.argwhere(Y_train == 4))
idx_train_5 = np.squeeze(np.argwhere(Y_train == 5))
idx_train_6 = np.squeeze(np.argwhere(Y_train == 6))
idx_train_7 = np.squeeze(np.argwhere(Y_train == 7))
idx_train_8 = np.squeeze(np.argwhere(Y_train == 8))
idx_train_9 = np.squeeze(np.argwhere(Y_train == 9))
idx_train_0 = np.squeeze(np.argwhere(Y_train == 0))

X_1 = X_train[idx_train_1]
X_2 = X_train[idx_train_2]
X_3 = X_train[idx_train_3]
X_4 = X_train[idx_train_4]
X_5 = X_train[idx_train_5]
X_6 = X_train[idx_train_6]
X_7 = X_train[idx_train_7]
X_8 = X_train[idx_train_8]
X_9 = X_train[idx_train_9]
X_0 = X_train[idx_train_0]

X_1 = prepare_mnist_list(X_1)
X_2 = prepare_mnist_list(X_2)
X_3 = prepare_mnist_list(X_3)
X_4 = prepare_mnist_list(X_4)
X_5 = prepare_mnist_list(X_5)
X_6 = prepare_mnist_list(X_6)
X_7 = prepare_mnist_list(X_7)
X_8 = prepare_mnist_list(X_8)
X_9 = prepare_mnist_list(X_9)
X_0 = prepare_mnist_list(X_0)

# Initialize Models

real_data = tf.placeholder(tf.float32, (None, *IMG_DIM))
z_ph = tf.placeholder(tf.float32, (None, Z_DIM))

fake_data = generator_tf(z_ph, reuse=False)

d_on_real_data = d_tf(real_data, reuse=False)
d_on_fake_data = d_tf(fake_data)
q_on_fake_data = q_tf(fake_data, reuse=False)
q_on_real_data = q_tf(real_data)

alpha = tf.random_uniform(shape=[tf.shape(fake_data)[0], 1, 1, 1], minval=0., maxval=1.)
interpolates = real_data + alpha * (fake_data - real_data)

gradients = tf.gradients(d_tf(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

q_cost = q_cost_tf(z_ph, q_on_fake_data)
g_cost = -tf.reduce_mean(d_on_fake_data)
d_cost = tf.reduce_mean(d_on_fake_data) - tf.reduce_mean(d_on_real_data) + LAMBDA * gradient_penalty

g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
q_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q')

g_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(g_cost, var_list=g_param)
d_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(d_cost, var_list=d_param)
q_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(q_cost, var_list=q_param + g_param)

saver = tf.train.Saver(max_to_keep=20)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_name = "./save/final-model"
new_saver = tf.train.import_meta_graph(model_name + ".meta")
new_saver.restore(sess, model_name)
# graph = tf.get_default_graph()

z_0_batch = []
z_1_batch = []
z_2_batch = []
z_3_batch = []
z_4_batch = []
z_5_batch = []
z_6_batch = []
z_7_batch = []
z_8_batch = []
z_9_batch = []
rec_0_batch = []
rec_1_batch = []
rec_2_batch = []
rec_3_batch = []
rec_4_batch = []
rec_5_batch = []
rec_6_batch = []
rec_7_batch = []
rec_8_batch = []
rec_9_batch = []


for i in range(0, 10000, 500):
	rec_loss_1, data_1, z_1 = getZandRec(sess, X_1)
	rec_loss_2, data_2, z_2 = getZandRec(sess, X_2)
	rec_loss_3, data_3, z_3 = getZandRec(sess, X_3)
	rec_loss_4, data_4, z_4 = getZandRec(sess, X_4)
	rec_loss_5, data_5, z_5 = getZandRec(sess, X_5)
	rec_loss_6, data_6, z_6 = getZandRec(sess, X_6)
	rec_loss_7, data_7, z_7 = getZandRec(sess, X_7)
	rec_loss_8, data_8, z_8 = getZandRec(sess, X_8)
	rec_loss_9, data_9, z_9 = getZandRec(sess, X_9)
	rec_loss_0, data_0, z_0 = getZandRec(sess, X_0)

	z_0_batch.append(z_0)
	z_1_batch.append(z_1)
	z_2_batch.append(z_2)
	z_3_batch.append(z_3)
	z_4_batch.append(z_4)
	z_5_batch.append(z_5)
	z_6_batch.append(z_6)
	z_7_batch.append(z_7)
	z_8_batch.append(z_8)
	z_9_batch.append(z_9)
	rec_0_batch.append(rec_loss_0)
	rec_1_batch.append(rec_loss_1)
	rec_2_batch.append(rec_loss_2)
	rec_3_batch.append(rec_loss_3)
	rec_4_batch.append(rec_loss_4)
	rec_5_batch.append(rec_loss_5)
	rec_6_batch.append(rec_loss_6)
	rec_7_batch.append(rec_loss_7)
	rec_8_batch.append(rec_loss_8)
	rec_9_batch.append(rec_loss_9)


z_0_batch = np.vstack(z_0_batch)
z_1_batch = np.vstack(z_1_batch)
z_2_batch = np.vstack(z_2_batch)
z_3_batch = np.vstack(z_3_batch)
z_4_batch = np.vstack(z_4_batch)
z_5_batch = np.vstack(z_5_batch)
z_6_batch = np.vstack(z_6_batch)
z_7_batch = np.vstack(z_7_batch)
z_8_batch = np.vstack(z_8_batch)
z_9_batch = np.vstack(z_9_batch)
rec_0_batch = np.vstack(rec_0_batch)
rec_1_batch = np.vstack(rec_1_batch)
rec_2_batch = np.vstack(rec_2_batch)
rec_3_batch = np.vstack(rec_3_batch)
rec_4_batch = np.vstack(rec_4_batch)
rec_5_batch = np.vstack(rec_5_batch)
rec_6_batch = np.vstack(rec_6_batch)
rec_7_batch = np.vstack(rec_7_batch)
rec_8_batch = np.vstack(rec_8_batch)
rec_9_batch = np.vstack(rec_9_batch)


print('Done!\n')
plt.clf()
sns.kdeplot(z_0_batch.flatten(), shade=True)
sns.kdeplot(z_1_batch.flatten(), shade=True)
sns.kdeplot(z_2_batch.flatten(), shade=True)
sns.kdeplot(z_3_batch.flatten(), shade=True)
sns.kdeplot(z_4_batch.flatten(), shade=True)
sns.kdeplot(z_5_batch.flatten(), shade=True)
sns.kdeplot(z_6_batch.flatten(), shade=True)
sns.kdeplot(z_7_batch.flatten(), shade=True)
sns.kdeplot(z_8_batch.flatten(), shade=True)
sns.kdeplot(z_9_batch.flatten(), shade=True)
plt.legend(['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9'])
plt.title('Z Distributions')
plt.savefig('Z_Distributions.png')
print('fig Z_Distributions.png saved\n')

plt.clf()
sns.kdeplot(rec_0_batch.flatten(), shade=True)
sns.kdeplot(rec_1_batch.flatten(), shade=True)
sns.kdeplot(rec_2_batch.flatten(), shade=True)
sns.kdeplot(rec_3_batch.flatten(), shade=True)
sns.kdeplot(rec_4_batch.flatten(), shade=True)
sns.kdeplot(rec_5_batch.flatten(), shade=True)
sns.kdeplot(rec_6_batch.flatten(), shade=True)
sns.kdeplot(rec_7_batch.flatten(), shade=True)
sns.kdeplot(rec_8_batch.flatten(), shade=True)
sns.kdeplot(rec_9_batch.flatten(), shade=True)
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.title('Loss')
plt.savefig('Loss.png')
print('fig Loss.png saved\n')

data = np.array(random.sample(X_train_list, BATCH_SIZE))

data_z = sess.run(q_on_real_data, feed_dict={real_data: data})
z = random_z()
z[:, :ZU_DIM] = data_z
rec = sess.run(fake_data, feed_dict={z_ph: z})
border = np.ones((128, 28, 3, 1))
plt.clf()
save_images.save_images(np.concatenate((data, rec, border), axis=2), 'Reconstruction.png')

rec, _, _ = getZandRec_best(sess, data)

save_images.save_images(np.concatenate((data, rec, border), axis=2), 'Reconstruction_best.png')
