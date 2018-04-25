import PIL
from PIL import Image
import numpy as np
import random

    
    
def rotate_batch(X, a):
    
    X_rot = np.zeros(X.shape)
    
    for i in range (len(X)):        
        img      = Image.fromarray(np.squeeze(X[i] + 1),mode='F')
        img      = img.rotate(a[i],resample=PIL.Image.BILINEAR)
        img      = np.clip(np.array(img) - 1, -1, 1)        
        X_rot[i] = np.expand_dims(img, -1)
    return X_rot