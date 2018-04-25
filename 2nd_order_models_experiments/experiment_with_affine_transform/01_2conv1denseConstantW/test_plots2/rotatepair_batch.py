import PIL
from PIL import Image
from PIL import ImageTransform
import numpy as np
import random
import math

    
    
def rotate_batch(X, a):
    
    X_rot = np.zeros(X.shape)
    
    for i in range (len(X)):        
        img      = Image.fromarray(np.squeeze(X[i] + 1),mode='F')
        img      = img.rotate(a[i],resample=PIL.Image.BILINEAR)
        img      = np.clip(np.array(img) - 1, -1, 1)        
        X_rot[i] = np.expand_dims(img, -1)
    return X_rot

def AffineTransform(X, shift, angle, shear, scale):
    bs, w, h, ch = X.shape
    X_rot = np.zeros(X.shape)
    params = np.zeros(shape=[bs, 6])
    for i in range(len(X)):
        image = Image.fromarray(np.squeeze(X[i] + 1), mode='F')
        ang = angle[i]*math.pi
        shearx, sheary = shear[i] / 5 + 0.3   # 0.1:0.5
        scalex, scaley = scale[i] / 4 + 1.15  # 0.9:1.4
        shiftx, shifty = shift[i] * 5         # -5:5
        ssin = np.sin(ang)
        ccos = np.cos(ang)
        RotMat = np.identity(3)
        RotMat[0, 0] = ccos
        RotMat[0, 1] = ssin
        RotMat[0, 2] = w/2 - w/2 * ccos - h/2 * ssin
        RotMat[1, 0] = -1 * ssin
        RotMat[1, 1] = ccos
        RotMat[1, 2] = h/2 + h/2 * ssin - w/2 * ccos

        ShearMat = np.identity(3)
        ShearMat[0, 1] = shearx
        ShearMat[1, 0] = sheary

        ScaleMat = np.identity(3)
        ScaleMat[0, 0] = scalex
        ScaleMat[1, 1] = scaley

        ShiftMat = np.identity(3)
        ShiftMat[0, 2] = shiftx
        ShiftMat[1, 2] = shifty

        AffineMat = np.matmul(np.matmul(RotMat, ShearMat), np.matmul(ScaleMat, ShiftMat))

        a, b, c = AffineMat[0]
        d, e, f = AffineMat[1]
        image = image.transform((w, h), Image.AFFINE, (a,b,c,d,e,f), resample=Image.BILINEAR)
        params[i] = np.stack(arrays=[a,b,c,d,e,f])
        image = np.clip(np.array(image) - 1, -1, 1)
        X_rot[i] = np.expand_dims(image, -1)
    return X_rot, params