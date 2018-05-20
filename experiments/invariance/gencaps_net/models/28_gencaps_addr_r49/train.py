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
import  rotatepair_batch
import plot_funs
""" Parameters """

BATCH_SIZE = 128
IMG_DIM    = (28, 28, 1)
Z_DIM      = 10

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

def encoder_tf(x, reuse = True):
    with tf.variable_scope("Encoder", reuse = reuse):
        x = tf.identity(x, name="input")
        x = tf.layers.conv2d(x,     64,  4, 2, padding='same', activation=lrelu)
        x = tf.layers.conv2d(x,     128, 4, 2, padding='same', activation=lrelu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=lrelu)
        x = fully_connected(x, Z_DIM , activation_fn=None)
        x = tf.identity(x, name="output")
        return x    

# Decoder == Reconstructor
def reconstructor_tf(x, angles, reuse = True):
    with tf.variable_scope("Reconstructor", reuse = reuse):
        aco   = tf.cos(angles*math.pi)
        asi   = tf.sin(angles*math.pi)
        acosi = tf.stack((aco,asi),axis=-1)
        x = tf.concat((x, acosi), axis=1)
        x = tf.identity(x, name="input")
        NCAP   = 128
        LENCAP = 4
        NDECAP = 7*7*128
        
        x      = tf.layers.dense(x, 1024, activation=tf.nn.relu)
        x_mcap = tf.layers.dense(x, NCAP * LENCAP, activation=tf.nn.relu)
        x_mcap = tf.reshape(x_mcap, (-1, NCAP, LENCAP))
        # all acap is positive because of relu
        x_acap = tf.layers.dense(x, NCAP,          activation=tf.nn.relu)
        
        
        NI = tf.get_variable("NI",(1, LENCAP, NDECAP ))
        NI = tf.tile(NI, [tf.shape(x_acap)[0], 1, 1])
        
        # BS, NCAP, NDECAP
        MNI = tf.matmul(x_mcap, NI)
        
        #BS, NCAP, NDECAP
        MNI = tf.nn.softmax(MNI, 2)
        
        MNI *= tf.expand_dims(x_acap, -1)
        
        x = tf.reduce_sum(MNI, axis=1)
        x = tf.nn.relu(x)
        
        x = tf.reshape(x, [-1, 7, 7, 128])
        x = tf.layers.conv2d_transpose(x, 64, 4, 2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 1,  4, 2,  padding='same', activation=tf.nn.tanh)
        x = tf.identity(x, name="output")                
        return x
    
def prepare_mnist(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return X

def random_z():
    return  np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM))

def get_batch_only_Xb(X, Y):
    idx = np.random.randint(len(X), size=BATCH_SIZE)
    return X[idx], Y[idx]

def get_batch(X,Y):
    a = np.random.uniform(-1, 1, size=(BATCH_SIZE,))
    Xb,Yb  = get_batch_only_Xb(X, Y)
    sel_idx = (Yb == 4) | (Yb == 9)  
    a[sel_idx] /= 4
    Xb_rot = rotatepair_batch.rotate_batch(Xb, a*180)
    return a, Xb, Xb_rot 


def  plot_pair_samples(X1, X2, save_path):
    X1 = np.squeeze(X1)
    X2 = np.squeeze(X2)
    sh     = list(X1.shape)
    sh[0] += X2.shape[0]
    X = np.zeros(sh)
    
    X[0::2] = X1
    X[1::2] = X2
    plot_funs.plot_img_1D_given_2D(X, 8, 32, save_path)
    

def train():
    # Prepare Training Data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    X_train = prepare_mnist(X_train)
    X_test  = prepare_mnist(X_test)
    
    # Initialize Models
    real_data     = tf.placeholder(tf.float32, (None, *IMG_DIM))
    real_data_rot = tf.placeholder(tf.float32, (None, *IMG_DIM))
    angles_tf     = tf.placeholder(tf.float32, (None,))
    
    real_z        = tf.placeholder(tf.float32, (None,  Z_DIM))
    
    encoded_data  = encoder_tf(real_data,                      reuse = False)
    rec_data      = reconstructor_tf(encoded_data,  angles_tf, reuse = False) 
    rec_real_z    = reconstructor_tf(real_z,        angles_tf, reuse = True)
    

    # reconstruction loss ( decoder cost)
    r_cost = tf.losses.mean_squared_error(real_data_rot, rec_data) 
    
    r_param  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,   scope='Reconstructor')
    e_param  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,   scope='Encoder')
        
    r_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(r_cost, var_list=e_param + r_param)
    
    saver = tf.train.Saver(max_to_keep=20)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    fix_z = random_z()
    
    f_train_stat = open("train_log.txt", "w", buffering = 1);
    f_test_stat  = open("test_log.txt",  "w", buffering = 1);
    os.system("mkdir -p figs figs_rec");
    for it in range(ITERS):        
        start_time = time.time()
        # first reconstruction phase
        angles, Xb, Xb_rot = get_batch(X_train, Y_train)        
        r_cost_rez, _ = sess.run( [r_cost, r_train_op], feed_dict={real_data: Xb, real_data_rot: Xb_rot, angles_tf : angles})        
        f_train_stat.write("%i %g\n"%(it, r_cost_rez))
        print(it, (time.time() - start_time ))
        
        if ((it + 1) % 500 == 0):
            
            angles, Xb, Xb_rot = get_batch(X_train, Y_train)
            samples = sess.run([rec_data], feed_dict={real_data: Xb, real_data_rot: Xb_rot, angles_tf : angles})
            plot_pair_samples(Xb_rot, samples, 'figs_rec/samples_%.6i_seen.png'%(it))
            
            angles, Xb, Xb_rot = get_batch(X_test, Y_test)
            samples = sess.run([rec_data], feed_dict={real_data: Xb, real_data_rot: Xb_rot, angles_tf : angles})
            plot_pair_samples(Xb_rot, samples, 'figs_rec/samples_%.6i_unseen.png'%(it))
                        
            
            samples = sess.run([rec_real_z], feed_dict={real_z: fix_z, angles_tf : angles})
            save_images.save_images(np.squeeze(samples),'figs/samples_%.6i.png'%(it))
            
            
            r_cost_rez = sess.run([r_cost], feed_dict={real_data: Xb, real_data_rot: Xb_rot, angles_tf : angles, real_z: random_z()})
            f_test_stat.write("%i %g\n"%(it, r_cost_rez[0]))

        if ((it + 1) % 10000 == 0):
            saver.save(sess, 'save/model', global_step=it)
        
    saver.save(sess, 'save/final-model')
        

train()
