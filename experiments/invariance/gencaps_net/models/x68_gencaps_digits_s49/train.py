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
from matrix_distance_tf import matrix_distance_2_tf

import random
import save_images
import os
import time
import batch
import plot_funs
import caps
""" Parameters """

BATCH_SIZE = 128
IMG_DIM    = (28, 28, 1)
IMG_DIM_OUT= (68, 68, 1)

Z_DIM      = 20
MAX_SHIFT = 20                                                                                                      
SHIFT_49  = 5




OUTPUT_DIM = int(np.prod(IMG_DIM))
LAMBDA     = 10
ITERS      = 100000
CRITIC_ITER= 5
LEARNING_RATE = 1e-3
# leaky relu alpha
leakyrelu_alpha    = 0.1

is_aae = False

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
def reconstructor_tf(x, shifts, reuse = True):
    with tf.variable_scope("Reconstructor", reuse = reuse):        
        x = tf.concat((x, shifts), axis=1)
        x = tf.identity(x, name="input")
        CAP_LEN = 10
        NCAP    = 20 
        NCATEGORICAL  = 10
        NMAPS  = 32
        SMAP   = 17 # size of map
        
        x_categorical = tf.nn.softmax(x[:,  :NCATEGORICAL])
        x_pose        = x[:,NCATEGORICAL:]
        
       
        x_firstcaps = tf.layers.dense(x_pose, CAP_LEN * NCATEGORICAL * NCAP )
        x_firstcaps = tf.reshape(x_firstcaps, (-1, NCAP, CAP_LEN, NCATEGORICAL))
        x_firstcaps = tf.einsum('bcln,bn->bcl', x_firstcaps, x_categorical)
        
        cap_r = []
        cap_a = []
        for i in range(NCAP):
            x = x_firstcaps[:,i,:]
            x_r = tf.layers.dense(x, 128, activation=tf.nn.relu)
            x_r = tf.layers.dense(x_r, 2, activation=None)
            x_a = tf.layers.dense(x, 128, activation=tf.nn.relu)
            x_a = tf.layers.dense(x_a, NMAPS, activation=tf.nn.sigmoid)
            cap_r.append(x_r)
            cap_a.append(x_a)
        cap_r = tf.stack(cap_r, axis=1)
        cap_a = tf.stack(cap_a, axis=1)
        
        # [-1,1] - > [0,SMAP-1] (to index space)                                                                                    
        cap_r = (cap_r + 1) * (SMAP - 1) / 2.0    
        
        
        x = caps.caps_tf(cap_r, cap_a, SMAP)
        x = tf.layers.conv2d_transpose(x, 32, 4, 2, padding='same',  activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 1,  4, 2,  padding='same', activation=tf.nn.tanh)
        x = tf.identity(x, name="output")                
        return x

def discriminator_tf(x, reuse = True):
    with tf.variable_scope("Discriminator", reuse = reuse):
        x = tf.identity(x, name="input")
        x = fully_connected(x, 1024, activation_fn=lrelu)
        x = fully_connected(x, 1024, activation_fn=lrelu)
        x = fully_connected(x, 1 , activation_fn=None)
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
    real_data_rot = tf.placeholder(tf.float32, (None, *IMG_DIM_OUT))
    shifts_tf     = tf.placeholder(tf.float32, (None,2))
    
    
    encoded_data  = encoder_tf(real_data,                      reuse = False)
    rec_data      = reconstructor_tf(encoded_data,  shifts_tf, reuse = False) 
    
    real_z        = tf.placeholder(tf.float32, (None,  Z_DIM))
    rec_real_z    = reconstructor_tf(real_z,        shifts_tf, reuse = True)

    # reconstruction loss ( decoder cost)
    r_cost = tf.losses.mean_squared_error(real_data_rot, rec_data)     
    
    if (is_aae):
    
    
        # for our discriminator
        # encoded_data is a fake_z
        fake_z = encoded_data
        d_on_real_data  = discriminator_tf(real_z, reuse = False)
        d_on_fake_data  = discriminator_tf(fake_z, reuse = True)
    
    
        alpha = tf.random_uniform(shape=[tf.shape(fake_z)[0], 1, 1, 1], minval=0., maxval=1.)
        interpolates      = real_z + alpha * (fake_z - real_z)
            
        
        gradients        = tf.gradients(discriminator_tf(interpolates, reuse=True), [interpolates])[0]
        slopes           = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)
    
        # "generator" loss (it is also our encoder)
        e_cost  = -tf.reduce_mean(d_on_fake_data)
        
        # discriminator loss 
        d_cost  =  tf.reduce_mean(d_on_fake_data) - tf.reduce_mean(d_on_real_data) + LAMBDA * gradient_penalty
        d_param  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,   scope='Discriminator')
        e_param  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,   scope='Encoder')
        
        d_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(d_cost, var_list=d_param)
        e_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(e_cost, var_list=e_param)

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
        shifts, Xb, Xb_rot = batch.get_batch_49(X_train, Y_train, BATCH_SIZE, MAX_SHIFT, SHIFT_49)                
        r_cost_rez, _ = sess.run( [r_cost, r_train_op], feed_dict={real_data: Xb, real_data_rot: Xb_rot, shifts_tf : shifts})
        
        if (is_aae):
            # second regularization phase (first udate discriminator next update generator(encoder))
            for i in range(CRITIC_ITER):
                Xb,_ = get_batch_only_Xb(X_train, Y_train)
                d_cost_rez, _ = sess.run( [d_cost, d_train_op], feed_dict={real_data: Xb, real_z: random_z()})
            e_cost_rez, _ = sess.run([e_cost, e_train_op], feed_dict={real_data: Xb})
            f_train_stat.write("%i %g %g %g\n"%(it, r_cost_rez, d_cost_rez, e_cost_rez))
        else:
            f_train_stat.write("%i %g\n"%(it, r_cost_rez))
            
        print(it, (time.time() - start_time ))
        
        if ((it + 1) % 500 == 0):
            
            shifts, Xb, Xb_rot = batch.get_batch_49(X_train, Y_train, BATCH_SIZE, MAX_SHIFT, SHIFT_49)
            samples = sess.run([rec_data], feed_dict={real_data: Xb, real_data_rot: Xb_rot, shifts_tf : shifts})
            plot_pair_samples(Xb_rot, samples, 'figs_rec/samples_%.6i_seen.png'%(it))
            
            shifts, Xb, Xb_rot = batch.get_batch_49(X_test, Y_test, BATCH_SIZE, MAX_SHIFT, SHIFT_49)
            samples = sess.run([rec_data], feed_dict={real_data: Xb, real_data_rot: Xb_rot, shifts_tf : shifts})
            plot_pair_samples(Xb_rot, samples, 'figs_rec/samples_%.6i_unseen.png'%(it))
                        
            
            samples = sess.run([rec_real_z], feed_dict={real_z: fix_z, shifts_tf : shifts})
            save_images.save_images(np.squeeze(samples),'figs/samples_%.6i.png'%(it))
            
            
            
                                                          
            if (is_aae):
                r_cost_rez, d_cost_rez, e_cost_rez = sess.run([r_cost, d_cost, e_cost], 
                feed_dict={real_data: Xb, real_data_rot: Xb_rot, shifts_tf : shifts, real_z: random_z()})
            else:
                r_cost_rez = sess.run(r_cost, feed_dict={real_data: Xb, real_data_rot: Xb_rot, shifts_tf : shifts})                
                f_test_stat.write("%i %g\n"%(it, r_cost_rez))

        if ((it + 1) % 10000 == 0):
            saver.save(sess, 'save/model', global_step=it)
        
    saver.save(sess, 'save/final-model')
        

train()
