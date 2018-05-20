#!/usr/bin/env python
# SingularityNet, Saint-Petersburg research laboratory
# Corresponding Author, Sergey Rodionov, email sergey@singularitynet.io 

import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist

import random
import os
import time
from matrix_distance import matrix_distance_np_fast

from scipy.misc import imsave
import rotatepair_batch

BATCH_SIZE = 20
Z_DIM       = 20



def save_plot_matrix(X, save_path):
    # [-1, 1] -> [0,255]
    X = (np.abs(127.5 * X +  127.4999)).astype('uint8')
    
    nh, nw = X.shape[:2]
    
    h, w = X[0,0].shape
    print(nh,nw,h,w)
    img = np.zeros((h*nh, w*nw))
                            
    for j in range(nh):
        for i in range(nw):
            img[j*h : j*h+h, i*w:i*w+w] = X[j,i]
    imsave(save_path, img)    
        
def prepare_mnist_list(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return list(X)

X_test_digits = []
for i in range(10):
    _, (X_test, Y_test) = mnist.load_data()                                                                                                  
    X_test = X_test[Y_test == i]                                                                                                         
    X_test_list  = prepare_mnist_list(X_test) 
    X_test_digits.append(X_test_list)

def get_sample_test_digit(digit, BATCH_SIZE):
    
#    _, (X_test, Y_test) = mnist.load_data()
#    X_test = X_test[Y_test == digit]
#    X_test_list  = prepare_mnist_list(X_test)
#    data = random.sample(X_test_list, BATCH_SIZE)
#    data = random.sample(X_test_digits[digit], BATCH_SIZE)
    data = X_test_digits[digit][:BATCH_SIZE]
    return data


def plot_for_model(model, tag):
    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         

    
    input_rec  = graph.get_tensor_by_name('Reconstructor/input:0')
    output_rec = graph.get_tensor_by_name('Reconstructor/output:0')
    input_enc  = graph.get_tensor_by_name('Encoder/input:0')
    output_enc = graph.get_tensor_by_name('Encoder/output:0')
    
    for v in np.arange(-2,2,0.5):
        z  = np.zeros((Z_DIM,Z_DIM))
        for z_pos in range(Z_DIM):
            z[z_pos,z_pos] = v
    
        #    print(z)
        csamples = []
        for a in np.arange(-1,1,0.1):
            angles = np.ones((Z_DIM,1)) * a                 
            aco = np.cos(angles * math.pi)
            asi = np.sin(angles * math.pi)
            za = np.concatenate((z, aco, asi), axis=1)
            samples = sess.run([output_rec], feed_dict={input_rec: za})
            csamples.append(np.squeeze(samples))
        
            
        csamples = np.squeeze(csamples)
        save_plot_matrix(csamples, 'rec_%s_%g.png'%(tag,v))
        
    sess.close() 

#for idx in np.arange(9999,2000000,10000):
for idx in (99999,):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        plot_for_model(model, str(idx))

