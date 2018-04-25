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

BATCH_SIZE = 20
Z_DIM       = 10



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
    data = X_test_digits[digit][:BATCH_SIZE]
    return data


def plot_for_model(model, tag):
    
    sess = tf.Session()  
    new_saver = tf.train.import_meta_graph(model + ".meta")
    new_saver.restore(sess, model)  
    graph = tf.get_default_graph()                         

    
    input_img     = graph.get_tensor_by_name('input_img:0')
    input_angles  = graph.get_tensor_by_name('input_angles:0')
    output_img = graph.get_tensor_by_name('output_img:0')
    
    for digit in range(10):
        Xb = get_sample_test_digit(digit,  BATCH_SIZE)
        Xb = np.array(Xb)
        csamples = []
        creal    = []
        for a in np.arange(-1,1,0.1):
            angles = np.ones(BATCH_SIZE) * a                 
            Xb_rot = rotatepair_batch.rotate_batch(Xb, angles * 180)        
            samples = sess.run(output_img, feed_dict={input_img: Xb, input_angles:angles})
            csamples.append(np.squeeze(samples))
            creal.append(Xb_rot)
        csamples = np.squeeze(csamples)
        creal    = np.squeeze(creal)
        sh = list(csamples.shape)
        sh[1] *= 2
        cmerge = np.zeros(sh)
        cmerge[:,0::2] = creal
        cmerge[:,1::2] = csamples    
        save_plot_matrix(cmerge, 'rec_%s_%i.png'%(tag,digit))
    
    
    sess.close() 

for idx in np.arange(9999,2000000,10000):
    model = os.path.join('../save/', 'model-%i'%(idx))
    if (os.path.isfile(model + ".meta")):
        print(model)
        plot_for_model(model, str(idx))

