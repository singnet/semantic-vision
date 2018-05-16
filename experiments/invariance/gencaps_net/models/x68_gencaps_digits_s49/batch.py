
#!/usr/bin/env python

import numpy as np


def get_batch_only_Xb(X, Y, BATCH_SIZE):
    idx = np.random.randint(len(X), size=BATCH_SIZE)
    return X[idx], Y[idx]

def shift_batch(Xb, shifts, MAX_SHIFT):
    s1 = Xb[0].shape[0]
    s2 = Xb[0].shape[1]
    
    sh = (len(Xb), s1 + MAX_SHIFT * 2, s2 + MAX_SHIFT * 2, 1)
    Xb_shift = -1 * np.ones(sh)
    
    for i in range(len(Xb)):
        start1 = shifts[i,0]  + MAX_SHIFT
        start2 = shifts[i,1]  + MAX_SHIFT
        Xb_shift[i, start1 : start1 + s1 , start2 : start2 + s2] = Xb[i]
    return Xb_shift

def get_batch_49(X,Y, BATCH_SIZE, MAX_SHIFT, SHIFT_49):
    shifts  = np.random.randint(-MAX_SHIFT, MAX_SHIFT, size=(BATCH_SIZE,2))
    Xb,Yb   = get_batch_only_Xb(X, Y, BATCH_SIZE)
    sel_idx = (Yb == 4) | (Yb == 9)  
    shifts[sel_idx] = np.random.randint(-SHIFT_49, SHIFT_49, size=(np.sum(sel_idx), 2) )
    Xb_sh = shift_batch(Xb, shifts, MAX_SHIFT)
    # print(list(zip(Yb, shifts)))
    return shifts / MAX_SHIFT, Xb, Xb_sh 
