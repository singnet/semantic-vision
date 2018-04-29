#!/usr/bin/env python


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
Z_DIM       = 10



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
        
def prepare_mnist(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return X

X_test_digits = []
for i in range(10):
    _, (X_test, Y_test) = mnist.load_data()                                                                                                  
    X_test = X_test[Y_test == i]                                                                                                         
    X_test_list  = prepare_mnist(X_test) 
    X_test_digits.append(X_test_list)



def print_for_model_digit(model, tag):
    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         

    
    input_rec  = graph.get_tensor_by_name('Reconstructor/input:0')
    output_rec = graph.get_tensor_by_name('Reconstructor/output:0')
    input_enc  = graph.get_tensor_by_name('Encoder/input:0')
    output_enc = graph.get_tensor_by_name('Encoder/output:0')
    
    #    r_cost = tf.losses.mean_squared_error(real_data_rot, output_enc)
    
    for digit in range(10):
        data = X_test_digits[digit]
        z  = sess.run([output_enc], feed_dict={input_enc: data})
        z  = z[0]
        
        outf = open("score_%s_%i.txt"%(tag, digit), 'w')
        for a in np.arange(-1,1,0.01):
            angles = np.ones((len(data),1)) * a                 
            Xb_rot = rotatepair_batch.rotate_batch(data, angles[:,0] * 180)        
            aco = np.cos(angles * math.pi)
            asi = np.sin(angles * math.pi)
            za = np.concatenate((z, aco, asi), axis=1)
            samples = sess.run([output_rec], feed_dict={input_rec: za})
        
            score = np.mean(np.square(Xb_rot - samples))
            outf.write("%g %g\n"%(a, score))
            print(tag, digit, a, score)        
    
    sess.close() 

for idx in np.arange(9999,2000000,10000):
#for idx in (99999,):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        print_for_model_digit(model, str(idx))
            
