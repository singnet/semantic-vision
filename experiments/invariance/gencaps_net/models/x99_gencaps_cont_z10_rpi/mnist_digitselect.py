#!/usr/bin/env python

import numpy as np
from keras.datasets import mnist


def mnist_digitselect(digits):
    digits = sorted(set(digits))

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))
    
    sel = Y == digits[0]
    for d in digits:
        sel = np.logical_or(sel, Y == d)
        
    X = X[sel]
    Y = Y[sel]
    
    # reorder labels
    
    label_map = dict()
    for i,d in enumerate(digits):
        label_map[d] = i
        
    for i in range(len(Y)):
        Y[i] = label_map[Y[i]]
        
    return (X, Y)

def mnist_digitselect_inverse(digits):
    inverse_digits = set(range(10)) - set(digits)
    print(inverse_digits);
    return mnist_digitselect(inverse_digits)