#!/usr/bin/env python
# SingularityNet, Saint-Petersburg research laboratory
# Corresponding Author, Sergey Rodionov, email sergey@singularitynet.io

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
Z_DIM      = 80
ZC_DIM     = 8
ZU_DIM     = 0

OUTPUT_DIM = int(np.prod(IMG_DIM))
LAMBDA     = 10
ITERS      = 100000
CRITIC_ITER= 5
LEARNING_RATE = 1e-3
# leaky relu alpha
leakyrelu_alpha    = 0.1


def lrelu(x):
    return tf.nn.relu(x) - leakyrelu_alpha * tf.nn.relu(-x)
""" Model Definitions """

def generator_tf(x, reuse = True):
    with tf.variable_scope("Generator", reuse = reuse):
        x = tf.identity(x, name="input")
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x = tf.layers.dense(x, 7 * 7 * 128 , activation=tf.nn.relu)
        x = tf.reshape(x, [-1, 7, 7, 128])
        x = tf.layers.conv2d_transpose(x, 64, 4, 2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 1, 4, 2,  padding='same', activation=tf.nn.tanh)
        x = tf.identity(x, name="output")
                
        return x

def d_tf(x, reuse = True):
    with tf.variable_scope("Discriminator", reuse = reuse):
        x = tf.identity(x, name="input")
        x = tf.layers.conv2d(x,     64,  4, 2, padding='same', activation=lrelu)
        x = tf.layers.conv2d(x,     128, 4, 2, padding='same', activation=lrelu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=lrelu)
        x = tf.layers.dense(x, 1)
        x = tf.identity(x, name="output")
        return x

def q_tf(x, reuse = True):
    with tf.variable_scope("Q", reuse = reuse):
        x = tf.identity(x, name="input")
        x = tf.layers.conv2d(x,     64,  4, 2, padding='same', activation=lrelu)
        x = tf.layers.conv2d(x,     128, 4, 2, padding='same', activation=lrelu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=lrelu)        
        x = fully_connected(x, ZC_DIM + ZU_DIM , activation_fn=None)
        x = tf.identity(x, name="output")
        return x    

def q_cost_tf(z, q):
    
    # categorical part
    z_cat = z[:, : ZC_DIM]
    q_cat = q[:, : ZC_DIM]
    lcat = tf.nn.softmax_cross_entropy_with_logits(labels=z_cat, logits=q_cat)
    
    if (ZU_DIM == 0):
        return  tf.reduce_mean(lcat);
    
    z_uni = z[:, ZC_DIM: ZC_DIM + ZU_DIM]
    q_uni = q[:, ZC_DIM: ZC_DIM + ZU_DIM]
    luni  = 0.5 * tf.square(z_uni - q_uni);
        
    return tf.reduce_mean(lcat) + tf.reduce_mean(luni);
    
def prepare_mnist_list(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return list(X)

def random_uc():
    idxs = np.random.randint(ZC_DIM, size=BATCH_SIZE)
    onehot = np.zeros((BATCH_SIZE, ZC_DIM))
    onehot[np.arange(BATCH_SIZE), idxs] = 1
    return onehot            
    

def random_z():
    rez = np.zeros([BATCH_SIZE, Z_DIM])
    rez[:,        : ZC_DIM] = random_uc()
    rez[:, ZC_DIM : ]       = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM - ZC_DIM))        
    return rez;

def train():
    # Prepare Training Data
    (X_train, _), (X_test, _) = mnist.load_data()
    
    X_train_list = prepare_mnist_list(X_train)
    X_test_list  = prepare_mnist_list(X_test)
    
    # Initialize Models
    
    real_data = tf.placeholder(tf.float32, (None, *IMG_DIM))
    z_ph      = tf.placeholder(tf.float32, (None,  Z_DIM))
        
    fake_data      = generator_tf(z_ph, reuse = False)    
    
    
    d_on_real_data  = d_tf(real_data, reuse = False)
    d_on_fake_data  = d_tf(fake_data)
    q_on_fake_data  = q_tf(fake_data, reuse = False)
    
    
    alpha = tf.random_uniform(shape=[tf.shape(fake_data)[0], 1, 1, 1], minval=0., maxval=1.)
    interpolates      = real_data + alpha * (fake_data - real_data)
            
    
    gradients        = tf.gradients(d_tf(interpolates), [interpolates])[0]
    slopes           = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
    
    q_cost  = q_cost_tf(z_ph, q_on_fake_data)
    g_cost  = -tf.reduce_mean(d_on_fake_data)
    d_cost  =  tf.reduce_mean(d_on_fake_data) - tf.reduce_mean(d_on_real_data) + LAMBDA * gradient_penalty

    
    g_param     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    d_param     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    q_param     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q')
    
    
    g_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(g_cost, var_list=g_param)
    d_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(d_cost, var_list=d_param)
    q_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(q_cost, var_list=q_param + g_param)
    
    saver = tf.train.Saver(max_to_keep=20)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    fix_z = random_z()
    
    f_train_stat = open("train_log.txt", "w", buffering = 1);
    f_test_stat  = open("test_log.txt",  "w", buffering = 1);
    os.system("mkdir -p figs");
    for it in range(ITERS):        
        start_time = time.time()
        for i in range(CRITIC_ITER):
            
            data = np.array(random.sample(X_train_list, BATCH_SIZE))
            d_cost_rez, _ = sess.run( [d_cost, d_train_op], feed_dict={real_data: data, z_ph: random_z()})
            

        g_cost_rez, q_cost_rez, _, _ = sess.run([g_cost, q_cost, g_train_op, q_train_op], feed_dict={z_ph: random_z()})
        
        f_train_stat.write("%i %g %g %g\n"%(it, g_cost_rez, d_cost_rez, q_cost_rez))
        print(it, (time.time() - start_time ))
        
        if ((it + 1) % 100 == 0):
            samples = sess.run([fake_data], feed_dict={z_ph: fix_z})
            save_images.save_images(np.squeeze(samples),'figs/samples_%.6i.png'%(it))
            
            data = np.array(random.sample(X_test_list, BATCH_SIZE))
            g_cost_rez, d_cost_rez, q_cost_rez = sess.run([g_cost, d_cost, q_cost], 
                                                          feed_dict={real_data: data, z_ph: random_z()})
            f_test_stat.write("%i %g %g %g\n"%(it, g_cost_rez, d_cost_rez, q_cost_rez))

        if ((it + 1) % 20000 == 0):
            saver.save(sess, 'save/model', global_step=it)
        
    saver.save(sess, 'save/final-model')
        

train()
