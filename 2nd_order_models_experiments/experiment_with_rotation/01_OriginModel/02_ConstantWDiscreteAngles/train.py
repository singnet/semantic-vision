import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist
from tensorflow.contrib.layers import conv2d, conv2d_transpose, layer_norm, fully_connected

import random
import os
import time
import  rotatepair_batch
import plot_funs
""" Parameters """

BATCH_SIZE = 128
IMG_DIM    = (28, 28, 1)

OUTPUT_DIM = int(np.prod(IMG_DIM))
ITERS      = 100000
LEARNING_RATE = 1e-3

""" Model Definitions """
    
def prepare_mnist(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return X


def get_batch_only_Xb(X, Y):
    idx = np.random.randint(len(X), size=BATCH_SIZE)
    return X[idx], Y[idx]

def get_batch(X,Y):
    a = np.random.uniform(-1, 1, size=(BATCH_SIZE,))
    a = np.random.randint(-5, 5, size=(BATCH_SIZE,))
    a = np.divide(a, 5)
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
    real_data     = tf.placeholder(tf.float32, (None, *IMG_DIM), name="input_img")
    real_data_rot = tf.placeholder(tf.float32, (None, *IMG_DIM))
    angles_tf     = tf.placeholder(tf.float32, (None,), name="input_angles")
    
    # convert angles to acosi
    aco   = tf.cos(angles_tf*math.pi)
    asi   = tf.sin(angles_tf*math.pi)
    acosi = tf.stack((aco,asi),axis=-1)

    W = tf.constant(np.ones((OUTPUT_DIM, OUTPUT_DIM)), dtype=tf.float32, name="W")
    W = tf.reshape(W, (1, OUTPUT_DIM, OUTPUT_DIM))
    W = tf.tile(W, [tf.shape(real_data)[0], 1, 1])
    
    W_control = tf.layers.dense(acosi, 64, activation=tf.nn.relu)
    W_control = tf.layers.dense(W_control, OUTPUT_DIM * OUTPUT_DIM, activation=None)
    W_control = tf.reshape(W_control, [-1, OUTPUT_DIM, OUTPUT_DIM])
    
    W_control = tf.nn.softmax(W_control, dim=1)
    
    W_rez = W * W_control
    
    x = tf.contrib.layers.flatten(real_data)
    
    x = tf.einsum('bi,bij->bj', x, W_rez)
    rec_data = tf.reshape(x, [-1, *IMG_DIM])
    rec_data = tf.identity(rec_data, "output_img")
    
    r_cost = tf.losses.mean_squared_error(rec_data, real_data_rot)     
    
    r_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(r_cost)
    
    saver = tf.train.Saver(max_to_keep=20)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    f_train_stat = open("train_log.txt", "w", buffering = 1);
    f_test_stat  = open("test_log.txt",  "w", buffering = 1);
    
    os.system("mkdir -p figs_rec");
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
                        
            
            r_cost_rez = sess.run(r_cost, feed_dict={real_data: Xb, real_data_rot: Xb_rot, angles_tf : angles})                
            f_test_stat.write("%i %g\n"%(it, r_cost_rez))

        if ((it + 1) % 10000 == 0):
            saver.save(sess, 'save/model', global_step=it)
        
    saver.save(sess, 'save/final-model')
        

train()
