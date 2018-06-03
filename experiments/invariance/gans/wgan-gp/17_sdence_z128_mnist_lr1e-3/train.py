#!/usr/bin/env python
# SingularityNet, Saint-Petersburg research laboratory
# Corresponding Author, Sergey Rodionov, email sergey@singularitynet.io 
# code is based on  https://github.com/igul222/improved_wgan_training

import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist
from tensorflow.contrib.layers import conv2d, conv2d_transpose, layer_norm, fully_connected

import random
import save_images
import os
import time
""" Parameters """

BATCH_SIZE = 128
IMG_DIM    = (28, 28, 1)
Z_DIM      = 128
OUTPUT_DIM = int(np.prod(IMG_DIM))
LAMBDA     = 10
ITERS      = 20001
CRITIC_ITER= 5

# leaky relu alpha
leakyrelu_alpha    = 0.2


def lrelu(x):
    return tf.nn.relu(x) - leakyrelu_alpha * tf.nn.relu(-x)
""" Model Definitions """

def generator_tf(x, reuse = False):
    with tf.variable_scope("Generator", reuse = reuse):
        x = fully_connected(x, 1024, activation_fn=tf.nn.relu)
        x = fully_connected(x, 1024, activation_fn=tf.nn.relu)
        x = fully_connected(x, OUTPUT_DIM , activation_fn=tf.nn.tanh)
        x = tf.reshape(x, [-1, *IMG_DIM])
        return x

def discriminator_tf(x, reuse = False):
    with tf.variable_scope("Discriminator", reuse = reuse):
        x = tf.contrib.layers.flatten(x)
        x = fully_connected(x, 1024, activation_fn=lrelu)
        x = fully_connected(x, 1024, activation_fn=lrelu)
        x = fully_connected(x, 1 , activation_fn=None)
        return x
    
def prepare_mnist_list(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return list(X)


def train():
    # Prepare Training Data
    (X_train, _), (X_test, _) = mnist.load_data()
    
    X_train_list = prepare_mnist_list(X_train)
    X_test_list  = prepare_mnist_list(X_test)
    
    # Initialize Models
    
    real_data = tf.placeholder(tf.float32, (BATCH_SIZE, *IMG_DIM))
    z_ph      = tf.placeholder(tf.float32, (BATCH_SIZE,  Z_DIM))
    z_random  = tf.random_uniform(shape=(BATCH_SIZE,  Z_DIM), minval=0, maxval=1)
        
    fake_data      = generator_tf(z_random)
    fake_data_ph   = generator_tf(z_ph, reuse = True)
    
    d_on_fake_data = discriminator_tf(fake_data)    
    d_on_real_data = discriminator_tf(real_data, reuse = True)
    
    alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
    interpolates      = real_data + alpha * (fake_data - real_data)
    
    gradients        = tf.gradients(discriminator_tf(interpolates, reuse=True), [interpolates])[0]
    slopes           = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)

    g_cost  = -tf.reduce_mean(d_on_fake_data)
    d_cost  =  tf.reduce_mean(d_on_fake_data) - tf.reduce_mean(d_on_real_data) + LAMBDA * gradient_penalty
    
    g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
    g_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(g_cost, var_list=g_param)
    d_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(d_cost, var_list=d_param)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    fix_z = np.random.rand(BATCH_SIZE, Z_DIM)
    
    f_train_stat = open("train_log.txt", "w", buffering = 1);
    f_test_stat  = open("test_log.txt",  "w", buffering = 1);
    os.system("mkdir -p figs");
    for it in range(ITERS):
        if (it % 100 == 0):
            samples = sess.run([fake_data_ph], feed_dict={z_ph: fix_z})
            save_images.save_images(np.squeeze(samples),'figs/samples_%.6i.png'%(it))
            
            data = np.array(random.sample(X_test_list, BATCH_SIZE))
            g_cost_rez, d_cost_rez = sess.run([g_cost, d_cost], feed_dict={real_data: data})
            f_test_stat.write("%i %g %g\n"%(it, g_cost_rez, d_cost_rez))
        
        start_time = time.time()
        for i in range(CRITIC_ITER):
            
            data = np.array(random.sample(X_train_list, BATCH_SIZE))
            d_cost_rez, _ = sess.run( [d_cost, d_train_op], feed_dict={real_data: data})
        
        g_cost_rez, _ = sess.run([g_cost, g_train_op])
        f_train_stat.write("%i %g %g\n"%(it, g_cost_rez, d_cost_rez))
        print(it, (time.time() - start_time ))
        
            

train()
