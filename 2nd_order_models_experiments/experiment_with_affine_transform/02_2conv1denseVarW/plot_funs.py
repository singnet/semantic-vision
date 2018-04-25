import numpy as np
from scipy.misc import imsave


# input is 2D matrix of 2D images)
def plot_img_2D_matrix(X, save_path):
    if (len(X.shape) == 5 and X.shape[4] == 1):
        X = X[:,:,:,:,0]
    X = (np.abs(127.5 * np.array(X) +  127.4999)).astype('uint8')
    nh, nw = X.shape[:2]
    
    h, w = X[0,0].shape
    print(nh,nw,h,w)
    img = np.zeros((h*nh, w*nw))
    
    for j in range(nh):
        for i in range(nw):
            img[j*h : j*h+h, i*w:i*w+w] = X[j,i]
    imsave(save_path, img)

# input is 1D vector of images (nh * nw must be equal to len(X))
def plot_img_1D_given_2D(X, nh, nw, save_path):
    assert len(X) == nh * nw , "Bad size"
    
    X2 = np.zeros((nh,nw, *X.shape[1:]))
    for i in range(nh):
        for j in range(nw):
            X2[i,j] = X[i * nw + j]
    plot_img_2D_matrix(X2, save_path)