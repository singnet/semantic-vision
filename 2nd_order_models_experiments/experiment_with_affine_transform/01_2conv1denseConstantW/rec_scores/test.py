import argparse
from PIL import Image

import numpy as np
import math

import tensorflow as tf
from keras.datasets import mnist

import random
import os
import time

from scipy.misc import imsave
import rotatepair_batch

SMALL_BATCH_SIZE = 32
PARAM_SIZE = 6


def small_batch_run(sess, feed_dict, full_batch_size, out_tf, small_batch_size):
    
    N_split = full_batch_size//small_batch_size + 1
    for k in feed_dict:
        feed_dict[k] = np.array_split(feed_dict[k], N_split)
    out_np_all = []
    for i in range(N_split):
        feed_dict_batch = dict()
        for k in feed_dict:
            feed_dict_batch[k] = feed_dict[k][i]
        out_np_batch  = sess.run(out_tf, feed_dict=feed_dict_batch)          
        out_np_all.append(out_np_batch)
                                
    out_np = np.concatenate(out_np_all, axis=0)
                                    
    return out_np
                                    


def save_plot_matrix(X, save_path):
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
    
    input_img     = graph.get_tensor_by_name('input_img:0')
    input_parameters  = graph.get_tensor_by_name('input_parameters:0')
    output_img = graph.get_tensor_by_name('output_img:0')
    for digit in range(10):
        Xb = X_test_digits[digit]
        batch_size = len(Xb)
        outf = open("score_%s_%i.txt"%(tag, digit), 'w')
        for affineParam in np.arange(-1,1,0.01):
            affineParamArray = np.ones(batch_size) * affineParam
            affineParamArrayVec = np.stack(arrays=[affineParamArray,affineParamArray], axis=-1)
            Xb_rot, output = rotatepair_batch.AffineTransform(Xb, affineParamArrayVec, affineParamArray, affineParamArrayVec, affineParamArrayVec)
            if PARAM_SIZE == 7:
                samples = small_batch_run(sess, {input_img: Xb, input_parameters:np.stack(arrays=[affineParamArray,affineParamArray,affineParamArray,affineParamArray,affineParamArray,affineParamArray,affineParamArray], axis=-1)}, batch_size, output_img, SMALL_BATCH_SIZE)
            else:
                samples = small_batch_run(sess, {input_img: Xb, input_parameters: output}, batch_size,
                                          output_img, SMALL_BATCH_SIZE)
                        
            score = np.mean(np.square(Xb_rot - samples))
            outf.write("%g %g\n"%(affineParam, score))
            print(tag, digit, affineParam, score)
    
    sess.close() 

for idx in (99999,):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        print_for_model_digit(model, str(idx))