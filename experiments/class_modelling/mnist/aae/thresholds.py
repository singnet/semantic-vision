"""
Created on Wed Jan 23 12:31:48 2019

@author: ahab
"""
from __future__ import print_function
import argparse
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from model import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
import time
import random
from torch.autograd.gradcheck import zero_gradients
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score
import seaborn as sns
from mnist import mnist_n

title_size = 16
axis_title_size = 14
ticks_size = 18

power = 2.0

device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def density_threshold(test, fake, bins=100, step=0.01):
    _, te = np.histogram(test, bins=bins)
    _, fe = np.histogram(fake, bins=bins)
    th = []
    r = np.arange(min(te[0], fe[0]), max(te[-1], fe[-1]), step)
    for cth in r:
        th.append((
            np.argwhere(test < cth).size,
            np.argwhere(test > cth).size,
            np.argwhere(fake < cth).size,
            np.argwhere(fake > cth).size,
        ))
    th = np.array(th)
    th1_arg = np.argmin(th[:, 1] + th[:, 2])
    #th2_arg = np.argmin(th[:, 0] + th[:, 3])
    thv1 = r[th1_arg]
    return thv1, np.where(test < thv1)[0].size / test.size, np.where(fake > thv1)[0].size / fake.size #, r[th2_arg]


def f1_threshold(test, fake, bins=100, step=0.01):
    #_, te = np.histogram(test, bins=bins)
    #_, fe = np.histogram(fake, bins=bins)
    te = [np.min(test), np.max(test)]
    fe = [np.min(fake), np.max(fake)]
    th = []
    #'''
    #precision = pth / (pth + (1 - nth))
    #recall = pth / (pth + (1 - pth))
    #f1 = (2*(recall*precision)) / (recall + precision)
    #'''
    r = np.arange(min(te[0], fe[0]), max(te[-1], fe[-1]), step)
    for cth in r:
        tp = np.where(test < cth)[0].size / test.size
        fn = np.where(test > cth)[0].size / test.size
        fp = np.where(fake < cth)[0].size / fake.size
        if tp == 0.0: #  and fp == 0.0
            th.append(-np.inf)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2*(recall*precision)) / (recall + precision)
        th.append(f1)
        '''
        th.append((
            np.argwhere(test < cth).size,
            np.argwhere(test > cth).size,
            np.argwhere(fake < cth).size,
            np.argwhere(fake > cth).size,
        ))
        '''
    th = np.array(th)
    th1_arg = np.argmax(th)
    #th2_arg = np.argmin(th[:, 0] + th[:, 3])
    thv1 = r[th1_arg]
    return thv1, np.max(th)


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def extract_batch_(data, it, batch_size):
    x = data[it * batch_size:(it + 1) * batch_size]
    return x


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def GetF1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def main(inliner_classes):
    batch_size = 64
    mnist_train = []
    mnist_valid = []
    z_size = 32

    #def shuffle_in_unison(a, b):
    #    assert len(a) == len(b)
    #    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    #    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    #    permutation = np.random.permutation(len(a))
    #    for old_index, new_index in enumerate(permutation):
    #        shuffled_a[new_index] = a[old_index]
    #        shuffled_b[new_index] = b[old_index]
    #    return shuffled_a, shuffled_b

    #outlier_classes = []
    #for i in range(total_classes):
    #    if i not in inliner_classes:
    #        outlier_classes.append(i)

    #for i in range(folds):
    #    if i != folding_id:
    #        with open('data_fold_%d.pkl' % i, 'rb') as pkl:
    #            fold = pickle.load(pkl)
    #        if len(mnist_valid) == 0:
    #            mnist_valid = fold
    #        else:
    #            mnist_train += fold

    #with open('data_fold_%d.pkl' % folding_id, 'rb') as pkl:
    #    mnist_test = pickle.load(pkl)

    #keep only train classes
    #mnist_train = [x for x in mnist_train if x[0] in inliner_classes]
    mnist_train, _ = mnist_n(inliner_classes, True)

    #random.seed(0)
    #random.shuffle(mnist_train)

    #def list_of_pairs_to_numpy(l):
    #    return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    #print("Train set size:", len(mnist_train))

    #mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    

    G = Generator(z_size).to(device)
    E = Encoder(z_size).to(device)
    setup(E)
    setup(G)
    G.eval()
    E.eval()

    G.load_state_dict(torch.load("Gmodel.pkl"))
    E.load_state_dict(torch.load("Emodel.pkl"))

    sample = torch.randn(64, z_size).to(device)
    sample = G(sample.view(-1, z_size, 1, 1)).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    if True:
        zlist = []
        rlist = []
        mzlist = []

        for it in range(0, mnist_train.shape[0], batch_size):
            #x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            x = setup(Variable(mnist_train[it:it+batch_size], requires_grad=True))
            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            z = z.cpu().detach().numpy()

            for i in range(x.shape[0]):
                distance = np.sum(np.power(recon_batch[i].flatten() - x[i].flatten(), power))
                rlist.append(distance)

            zlist.append(z)
            mzlist.append(np.mean(np.square(z), 1))

        data = {}
        data['rlist'] = np.array(rlist)
        data['zlist'] = zlist
        data['mzlist'] = np.array(mzlist)

        with open('data.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)

    with open('data.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    rlist = data['rlist']
    zlist = data['zlist']
    
    #print(data['mzlist'].shape)
    '''
    sns.distplot(data['mzlist'].flatten())
    plt.savefig('mnist_d%d_z_dist.png' % inliner_classes[0])
    plt.clf()
    plt.cla()
    plt.close()
    
    sns.distplot(data['rlist'].flatten())
    plt.savefig('mnist_d%d_x_rec.png' % inliner_classes[0])
    plt.clf()
    plt.cla()
    plt.close()
    '''
    '''
    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    plt.plot(bin_edges[1:], counts, linewidth=2)
    plt.xlabel(r"Distance, $\left \|\| I - \hat{I} \right \|\|$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('mnist_d%d_randomsearch.pdf' % inliner_classes[0])
    plt.savefig('mnist_d%d_randomsearch.eps' % inliner_classes[0])
    plt.clf()
    plt.cla()
    plt.close()
    '''
    return data['rlist']

def main2(inliner_classes, outlier_classes=False):
    batch_size = 64
    mnist_train = []
    mnist_valid = []
    z_size = 32
    
    if outlier_classes:
        in_mnist_train, _ = mnist_n(inliner_classes, True)
        in_mnist_train_size = in_mnist_train.shape[0]
        in_mnist_train_chunk = in_mnist_train_size // 9
        mnist_train = torch.from_numpy(np.vstack([mnist_n(i, True)[0][:in_mnist_train_chunk] for i in range(10) if i != inliner_classes]))
        #print(mnist_train.shape, in_mnist_train.shape)
        #exit()
    else:
        mnist_train, _ = mnist_n(inliner_classes, True)
    
    G = Generator(z_size).to(device)
    E = Encoder(z_size).to(device)
    setup(E)
    setup(G)
    G.eval()
    E.eval()

    G.load_state_dict(torch.load("Gmodel.pkl"))
    E.load_state_dict(torch.load("Emodel.pkl"))

    sample = torch.randn(64, z_size).to(device)
    sample = G(sample.view(-1, z_size, 1, 1)).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    if True:
        zlist = []
        rlist = []
        mzlist = []

        for it in range(0, mnist_train.shape[0], batch_size):
            #x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, 32 * 32).data, requires_grad=True)
            x = setup(Variable(mnist_train[it:it+batch_size], requires_grad=True))
            z = E(x.view(-1, 1, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            z = z.cpu().detach().numpy()

            for i in range(x.shape[0]):
                distance = np.sum(np.power(recon_batch[i].flatten() - x[i].flatten(), power))
                rlist.append(distance)

            zlist.append(z)
            mzlist.append(np.mean(np.square(z), 1))

        data = {}
        data['rlist'] = np.array(rlist)
        data['zlist'] = zlist
        data['mzlist'] = np.array(mzlist)

        with open('data.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)

    with open('data.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    rlist = data['rlist']
    zlist = data['zlist']

    return data['rlist']


if __name__ == '__main__':
    #main(0, [0], 10)
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--ntrain', help='digit index', type=int)
    #parser.add_argument('--seed', help='random seed', type=int)
    #args = parser.parse_args()
    
    #Z_SIZE = args.zsize
    #NID = args.ntrain
    #np.random.seed(args.seed)
    
    #mnist = mnist_n(NID)
    #mnist_test_size = mnist.x_test.shape[0]
    #mnist_test_chunk = mnist_test_size // 9
    #omnist = np.vstack([mnist_n(i).next_test_batch(mnist_test_chunk) for i in range(10) if i != NID])
    
    #Fclass = 5
    T = 0
    r3 = main2(T, False) #main(3)
    r5 = main2(T, True) #main(Fclass)
    
    thv, rec_pth, rec_nth = density_threshold(r3, r5)
    f1thv, f1v = f1_threshold(r3, r5)
    fail = (((1 - rec_nth) + (1 - rec_pth))*100) / 2
    
    rrange = (int(min(r3.min(), r5.min())), int(max(r3.max(), r5.max())))
    r3b, r3e = np.histogram(r3, bins=100, range=rrange)
    r5b, _ = np.histogram(r5, bins=100, range=rrange)
    rnb = r3b + r5b
    r3e = np.array([(r3e[i]+r3e[i+1]) / 2 for i in range(len(r3e)-1)])
    
    p1 = (r3e[np.argmax(r3b)], np.max(r3b))
    p2 = (r3e[np.argmax(r5b)], np.max(r5b))
    
    start_idx = np.argmax(r3b)
    end_idx = np.argmax(r5b)
    
    #d1, d2 = [], []
    crit = []
    for i in range(start_idx, end_idx):
        #d.append(np.abs(np.abs(i - p1[1]) - np.abs(i - p2[1])))
        
        s1 = np.sum(r3e[start_idx:i+1]*rnb[start_idx:i+1])
        s2 = np.sum(r3e[i:end_idx]*rnb[i:end_idx])
        crit.append(s1 + s2)
        #d1.append(s1) #np.abs(i - p1[1]))
        #d2.append(s2) #np.abs(i - p2[1]))
    
    uth = r3e[start_idx+np.argmin(crit)]
    
    upth = np.argwhere(r3 > uth).size / r3.size # false negative
    unth = np.argwhere(r5 < uth).size / r5.size # false positive
    ufail = ((upth + unth)*100) / 2
    
    sns.distplot(r3)
    sns.distplot(r5)
    
    yb = plt.ylim()
    plt.plot([thv, thv], [yb[0], yb[1]], 'r-')
    plt.plot([f1thv, f1thv], [yb[0], yb[1]], 'k--')
    plt.plot([uth, uth], [yb[0], yb[1]], 'm--')
    plt.ylim(yb)
    plt.title('AAE{} reconstruction error, (z size = 32)\nfail = {:3.1f} %, F1 = {:1.2f}, bimode fail = {:3.2f} %'.format(T, fail, f1v, ufail))
    
    plt.legend([
            'threshold',
            'F1 threshold',
            'bimode threshold',
            '{}s {:3.1f} %'.format(T, rec_pth*100),
            'others {:3.1f} %'.format(rec_nth*100),
    ])
    
    plt.savefig(f'mnist_d{T}_rec_th.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    
    
    
    #plt.plot(r3e[:-1], r3b)
    #plt.plot(r3e[:-1], r5b)
    
    
    #crit = np.array(crit)
    
    #plt.plot(r3e, rnb)
    #plt.plot(r3e[start_idx:end_idx], d1)
    #plt.plot(r3e[start_idx:end_idx], d2)
    #plt.plot(r3e[start_idx:end_idx], crit) # r3e[start_idx:end_idx]
    #plt.scatter(*p1)
    #plt.scatter(*p2)
    #plt.savefig(f'mnist_rn_d{T}_rec_th.png')
    #plt.clf()
    #plt.cla()
    #plt.close()
    
    ''' #'''
    
    

