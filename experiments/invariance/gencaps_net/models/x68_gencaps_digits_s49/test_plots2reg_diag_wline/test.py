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
import batch

BATCH_SIZE = 20
Z_DIM       = 10

MAX_SHIFT = 20                                                                                                                             
SHIFT_49  = 5


def save_plot_matrix(X, save_path):
    # [-1, 1] -> [0,255]
    X = (np.abs(127.5 * X +  127.4999)).astype('uint8')
    
    nh, nw = X.shape[:2]
    
    h, w = X[0,0].shape
    print(nh,nw,h,w)
    img = np.zeros((h*nh, w*nw + nw//2)) + 255
                            
    for j in range(nh):
        for i in range(nw):
            delta_line = i//2
            img[j*h : j*h+h, i*w + delta_line:i*w+w + delta_line] = X[j,i]
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
    
    for digit in range(10):
        data = get_sample_test_digit(digit,  BATCH_SIZE)
        data = np.array(data)
        z  = sess.run([output_enc], feed_dict={input_enc: data})
        z  = z[0]
        csamples = []
        creal    = []
        for s in np.arange(-20,20,2):
            # diagonal shifts
            shifts = np.ones((BATCH_SIZE,2), dtype=np.int) * s
            
            Xb_shift = batch.shift_batch(data, shifts, MAX_SHIFT)
            shifts = shifts / MAX_SHIFT
            za = np.concatenate((z, shifts), axis=1)
            samples = sess.run([output_rec], feed_dict={input_rec: za})
            csamples.append(np.squeeze(samples))
            creal.append(Xb_shift)
        csamples = np.squeeze(csamples)
        creal    = np.squeeze(creal)
        sh = list(csamples.shape)
        sh[1] *= 2
        cmerge = np.zeros(sh)
        cmerge[:,0::2] = creal
        cmerge[:,1::2] = csamples    
        save_plot_matrix(cmerge, 'rec_%s_%i.png'%(tag,digit))
    
    
    sess.close() 

#for idx in np.arange(9999,2000000,10000):
for idx in (99999,):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        plot_for_model(model, str(idx))

