
#!/usr/bin/env python


import numpy as np
import tensorflow as tf

""" Parameters """


# simple indexes (in index space!!!)
def fixed_address(s):
    NI = np.zeros((s, s, 2))
    NI[:,:,0] = np.reshape(range(s), (-1, 1))
    NI[:,:,1] = np.reshape(range(s), (1 ,-1))    
    return NI

# cap_r (batch_size, NCAP, 2)
# cap_r should be in INDEX SPACE!
# cap_a  (batch_size, NCAP, NMAPS)
# SMAP - size of output maps
# output: (batch_size, SMAP, SMAP, NMAPS)
def caps_tf(cap_r, cap_a, SMAP):
    
    NI = fixed_address(SMAP)     
    NI = np.reshape(NI, (1,1, SMAP,SMAP, 2))    
    NI = tf.constant(NI, dtype=cap_r.dtype, name="NI")
    
    # cap_r - (batch_size, NCAP, 2) -> (batch_size, NCAP, 1, 1, 2)
    cap_r = tf.reshape(cap_r, (tf.shape(cap_r)[0], tf.shape(cap_r)[1], 1, 1, 2) )
    #relu(1 - |dx| ) * relu(1 - |dy|)  
    x = tf.nn.relu(1 - tf.abs(NI - cap_r))
    x = tf.reduce_prod(x, axis=-1)       # (1-dx) * (1-dy)
        
    # x (batch_size, NCAP, SMAP, SMAP)
    # a  (batch_size, NCAP, NMAPS) 
    x  = tf.einsum('abcd,abf->acdf', x, cap_a)
    return x

# only for test
def caps_np(cap_r, cap_a, SMAP):
    rez = np.zeros((cap_a.shape[0], SMAP, SMAP, cap_a.shape[-1]))
    for b in range(cap_a.shape[0]):
        for m in range(cap_a.shape[-1]):
            for c in range(cap_r.shape[1]):
                x =  cap_r[b, c, 0]
                y =  cap_r[b, c, 1]
                s = 0
                for i in range(SMAP):
                    for j in range(SMAP):
                        dx = 1 - np.abs(i - x)
                        dy = 1 - np.abs(j - y)
                        w = np.clip(dx,0, None) * np.clip(dy, 0, None)
                        rez[b,i,j,m] += w * cap_a[b, c, m]
                        s += w
                if (x > 0 and x <= SMAP -1 and y > 0 and y <= SMAP -1  and (not np.allclose(s,1))):  
                    print("Error s", x, y, SMAP, s - 1)
                    exit(0)
                if (s > 1.0000001 and s < 0):
                    print("Error s2", s)
                    exit(0)
    return rez
                        
