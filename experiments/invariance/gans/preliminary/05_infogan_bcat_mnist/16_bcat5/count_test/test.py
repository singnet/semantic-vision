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

BATCH_SIZE = 128
Z_DIM      = 80
ZBC_DIM    = 5
     
def prepare_mnist(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return X

def random_z():
    rez = np.zeros([BATCH_SIZE, Z_DIM])
    rez[:,        : ZBC_DIM] = np.random.randint(2, size=[BATCH_SIZE, ZBC_DIM]) # random 0 or 1 (binary categorical)
    rez[:, ZBC_DIM : ]       = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_DIM - ZBC_DIM))
    return rez;
                
def plot_for_model(model, tag):
    # Prepare Training Data
    (X_train, _), _= mnist.load_data()
    
    X_train= prepare_mnist(X_train)
                    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         
    input = graph.get_tensor_by_name('Q/input:0')
    output = graph.get_tensor_by_name('Q/output:0')

    output = tf.nn.sigmoid(output)

    # count number of different combinations
    
    rez = np.zeros((2**ZBC_DIM),dtype=np.int)
    
    # we can calculate for entire training set
    all_feats = sess.run(output, feed_dict={input: X_train})
    
    for f in all_feats:
        
        f = f[0:ZBC_DIM]>0.5
    
        idx_str_binary = "";
        for p in f:
            idx_str_binary += str(int(p))
        
        idx = int(idx_str_binary, 2)
        rez[idx] += 1
    
    print(tag, " ".join(list(map(str,rez))))
    sess.close() 

for idx in np.arange(9999,2000000,10000):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        plot_for_model(model, str(idx))
                                
