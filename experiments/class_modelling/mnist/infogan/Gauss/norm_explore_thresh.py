import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
import argparse

import random

from tensorflow.contrib.layers import conv2d, conv2d_transpose, layer_norm, fully_connected

from save_images import save_images


def density_threshold(test, fake, bins=100, step=0.01):
    _, te = np.histogram(test, bins=bins)
    _, fe = np.histogram(fake, bins=bins)
    th = []
    r = np.arange(min(te[0], fe[0]), max(te[-1], fe[-1]), step)
    for cth in r:
        th.append((
            np.argwhere(test < cth).size,
            np.argwhere(test > cth).size,
            np.argwhere(fake < cth).size,
            np.argwhere(fake > cth).size,
        ))
    th = np.array(th)
    return th, r


BATCH_SIZE = 500
IMG_DIM    = (28, 28, 1)
Z_DIM      = 120
ZC_DIM     = 0
ZU_DIM     = 100

OUTPUT_DIM = int(np.prod(IMG_DIM))
LAMBDA     = 10
ITERS      = 200000
CRITIC_ITER= 5

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
    rez[:, :] = np.random.normal(0, 1, size=(BATCH_SIZE, Z_DIM - ZC_DIM))
    return rez


def getZandRec(sess, X):
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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = ((x_train) / 255.0).reshape((-1, 28, 28, 1))
x_test = ((x_test) / 255.0).reshape((-1, 28, 28, 1))

train_idx_real = np.argwhere(y_train == 3).flatten()
train_idx_fake = np.argwhere(y_train == 5).flatten()
test_idx_real = np.argwhere(y_test == 3).flatten()
test_idx_fake = np.argwhere(y_test == 5).flatten()

xz_train, xz_test = [], []
xr_train, xr_test = [], []
batch_size = BATCH_SIZE

for i in range(0, x_train.shape[0], batch_size):
	c_xr, daata, c_xz = getZandRec(sess, x_train[i:i+batch_size])
	xz_train.append(c_xz)
	xr_train.append(c_xr)
xz_train = np.vstack(xz_train)
xr_train = np.vstack(xr_train)

batch_size = BATCH_SIZE
for i in range(0, x_test.shape[0], batch_size):
	c_xr, daata, c_xz = getZandRec(sess, x_test[i:i + batch_size])
	xz_test.append(c_xz)
	xr_test.append(c_xr)
xz_test = np.vstack(xz_test)
xr_test = np.vstack(xr_test)

xr_train_loss = np.mean(np.square(x_train - xr_train), (1, 2, 3))
xr_test_loss = np.mean(np.square(x_test - xr_test), (1, 2, 3))

save_images(x_test[test_idx_fake][:25], 'Results/Fake_orig.png').astype(np.int)
save_images(xr_test[test_idx_fake][:25], 'Results/Fake_reconstruct.png').astype(np.int)

r_test_loss = xr_test_loss[test_idx_real]
r_fake_loss = xr_test_loss[test_idx_fake]

th_rloss, te_rloss = density_threshold(r_test_loss, r_fake_loss, step=0.0015)

ths_rloss = th_rloss[:, 1] + th_rloss[:, 2]
thv_rloss = te_rloss[np.argmin(ths_rloss)]
print('threshold value:', thv_rloss)
plt.plot(te_rloss, ths_rloss)
plt.title('threshold criterion')
plt.savefig("Results/Threshold_reconstruction.png")
plt.clf()

sns.distplot(r_test_loss)
sns.distplot(r_fake_loss)
yb = plt.ylim()
plt.plot([thv_rloss, thv_rloss], [yb[0], yb[1]])
plt.ylim(yb)
plt.legend(['threshold', '3s (test)', '5s (fake)'])
plt.title('InfoGAN reconstruction error (z size = {})'.format(ZU_DIM))
plt.savefig("Results/Division_by_rec_threshold.png")
plt.clf()

print('Threshold value: {0:3.3f}'.format(thv_rloss))
pth = np.argwhere(r_test_loss < thv_rloss).size
nth = np.argwhere(r_fake_loss < thv_rloss).size
print('Positives behind threshold:', '{:3.1f} %'.format((pth / r_test_loss.size)*100), pth, '/', r_test_loss.size)
print('Negatives behind threshold:', '{:3.1f} %'.format((nth / r_fake_loss.size)*100), nth, '/', r_fake_loss.size)
fail_score = (((1 - (pth / r_test_loss.size))*100) + ((nth / r_fake_loss.size)*100)) / 2
print('Fails score:', '{:3.1f} %'.format(fail_score))

sns.distplot(xz_test[test_idx_real].flatten())
sns.distplot(xz_test[test_idx_fake].flatten())
plt.legend(['3s (test)', '5s (fake)'])
plt.title('InfoGAN.z distribution (z size = {})'.format(ZU_DIM))
plt.savefig("Results/Z_distr.png")
plt.clf()

axz_test = np.sum(np.square(xz_test[test_idx_real]), 1)
axz_fake = np.sum(np.square(xz_test[test_idx_fake]), 1)
sns.distplot(axz_test)
sns.distplot(axz_fake)
plt.legend(['3s (test)', '5s (fake)'])
plt.title('InfoGAN.z likelihood (z size = {})'.format(ZU_DIM))
plt.savefig("Results/z_likelihood.png")
plt.clf()

th_zll, te_zll = density_threshold(axz_test, axz_fake, step=0.05)
th_c = th_zll[:, 1] + th_zll[:, 2]
thv_zll = te_zll[np.argmin(th_c)]
print('threshold value:', thv_zll)
plt.plot(te_zll, th_c)
plt.title('threshold criterion')
plt.savefig("Results/threshold_for_likelihood.png")
plt.clf()

sns.distplot(axz_test)
sns.distplot(axz_fake)
yb = plt.ylim()
plt.plot([thv_zll, thv_zll], [yb[0], yb[1]])
plt.ylim(yb)
plt.legend(['threshold', '3s (test)', '5s (fake)'])
plt.title('InfoGAN.z likelihood (z size = {})'.format(ZU_DIM))
plt.savefig("Results/division_by_likelihood.png")
plt.clf()

print('Threshold value: {0:3.3f}'.format(thv_zll))
pth = np.argwhere(axz_test < thv_zll).size
nth = np.argwhere(axz_fake < thv_zll).size
print('Positives behind threshold:', '{:3.1f} %'.format((pth / axz_test.size)*100), pth, '/', axz_test.size)
print('Negatives behind threshold:', '{:3.1f} %'.format((nth / axz_fake.size)*100), nth, '/', axz_fake.size)
fail_score = (((1 - (pth / axz_test.size))*100) + ((nth / axz_fake.size)*100)) / 2
print('Fails score:', '{:3.1f} %'.format(fail_score))
