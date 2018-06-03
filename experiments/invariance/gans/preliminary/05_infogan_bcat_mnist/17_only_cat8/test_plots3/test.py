#!/usr/bin/env python
# SingularityNet, Saint-Petersburg research laboratory
# Corresponding Author, Sergey Rodionov, email sergey@singularitynet.io 

import argparse

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist

import random

import os
import time
from scipy.misc import imsave                                                                                                                


BATCH_SIZE = 20
Z_DIM      = 80
ZC_DIM     = 8

                
     
    
def random_z_zeroc():
    rez = np.zeros([BATCH_SIZE, Z_DIM])
    rez[:, ZC_DIM : ]       = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM - ZC_DIM))
    return rez;
    
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

def plot_for_model(model, tag):
    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         
    input = graph.get_tensor_by_name('Generator/input:0')
    output = graph.get_tensor_by_name('Generator/output:0')
    
    
    plot_matrix = np.zeros((BATCH_SIZE, ZC_DIM, 28, 28))
    
    # over all combinations
    z = random_z_zeroc()

    for i in range(ZC_DIM):
        
        z[:,i] = 1
        samples = sess.run([output], feed_dict={input: z})
        plot_matrix[:, i, :, :] = np.squeeze(samples)
        z[:,i] = 0
    
    save_plot_matrix(plot_matrix,'samples_%s.png'%(tag))                                                                       
            
    sess.close() 

for idx in np.arange(9999,2000000,10000):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        plot_for_model(model, str(idx))
                                
