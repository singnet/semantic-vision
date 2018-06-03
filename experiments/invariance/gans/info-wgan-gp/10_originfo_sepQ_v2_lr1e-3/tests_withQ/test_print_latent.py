#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import os
import time

Z_DIM      = 80
ZC_DIM     = 10
ZU_DIM     = 2
     
def prepare_mnist(X):
    X = (X.astype(np.float32) - 127.5)/127.5
    X = X[:, :, :, None]
    return X

def print_for_model(model):
    # Prepare Training Data
    _, (X_test,_)= mnist.load_data()
    
    X_test= prepare_mnist(X_test)
                    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         
    input = graph.get_tensor_by_name('Q/input:0')
    output = graph.get_tensor_by_name('Q/output:0')
    
    output_cat = output[:,:ZC_DIM]
    output_cat_softmax = tf.nn.softmax(output_cat)
    output_uniform = output[:, ZC_DIM: ZC_DIM + ZU_DIM]
    
    for X in X_test:
        cat, cat_softmax, uniform = sess.run([output_cat, output_cat_softmax, output_uniform],
                                            feed_dict={input: np.array([X])})
        print(cat[0], cat_softmax[0], uniform[0])
    sess.close() 

print_for_model("../save/final-model")
