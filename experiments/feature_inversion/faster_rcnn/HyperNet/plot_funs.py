#!/usr/bin/env python

import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

def plot_pair_samples(imgs, rimgs, filename):
    N = len(imgs)
    fig, axs = plt.subplots(2, N, figsize=(10, 10))
    for i in range(N):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(rimgs[i])
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    # plt.subplots_adjust(top = 0.1, bottom=0.01, hspace=0.1, wspace=0.01)
    plt.savefig(filename, format='png')