import warnings
warnings.filterwarnings('ignore')

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace


def next_batch(xs, batch_size):
        idx = np.random.randint(0, xs.shape[0], batch_size)
        x_batch = xs[idx]
        return x_batch


def mnist_n(n):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = ((x_train - 127.5) / 127.5).reshape((-1, 784))
    x_test = ((x_test - 127.5) / 127.5).reshape((-1, 784))
    
    if n is not None:
        train_target_idxs = np.argwhere(y_train == n).flatten()
        test_target_idxs = np.argwhere(y_test == n).flatten()
        
        x_train = x_train[train_target_idxs]
        x_test = x_test[test_target_idxs]
    return SimpleNamespace(
            x_train=x_train, x_test=x_test,
            next_train_batch=lambda batch_size: next_batch(x_train, batch_size),
            next_test_batch=lambda batch_size: next_batch(x_test, batch_size)
    )


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zsize', help='dimension of PCA', type=int)
    parser.add_argument('--ntrain', help='digit index', type=int)
    parser.add_argument('--seed', help='random seed', type=int)
    args = parser.parse_args()
    
    Z_SIZE = args.zsize
    NID = args.ntrain
    np.random.seed(args.seed)
    
    mnist = mnist_n(NID)
    mnist_test_size = mnist.x_test.shape[0]
    mnist_test_chunk = mnist_test_size // 9
    omnist = np.vstack([mnist_n(i).next_test_batch(mnist_test_chunk) for i in range(10) if i != NID])
    
    pca = PCA(Z_SIZE, whiten=True)
    pca.fit(mnist.x_train) # fit on trainset
    
    z_train = pca.transform(mnist.x_train)
    z_test = pca.transform(mnist.x_test)
    oz_test = pca.transform(omnist)
    
    r_train = pca.inverse_transform(z_train)
    r_test = pca.inverse_transform(z_test)
    or_test = pca.inverse_transform(oz_test)
    
    # scores
    z_train_score = np.mean(np.square(z_train), 1)
    z_test_score = np.mean(np.square(z_test), 1)
    oz_test_score = np.mean(np.square(oz_test), 1)
    
    r_train_score = np.mean(np.square(r_train - mnist.x_train), 1)
    r_test_score = np.mean(np.square(r_test - mnist.x_test), 1)
    or_test_score = np.mean(np.square(or_test - omnist), 1)
    
    # reconstruction error
    r_f1thv, r_f1score = f1_threshold(r_test_score, or_test_score)
    
    thv, pth, nth = density_threshold(r_test_score, or_test_score)
    rec_fail_score = (((1 - nth) + (1 - pth))*100) / 2
    
    precision = pth / (pth + (1 - nth))
    recall = pth / (pth + (1 - pth))
    rec_f1 = (2*(recall*precision)) / (recall + precision)
    
    sns.distplot(r_train_score)
    sns.distplot(r_test_score)
    sns.distplot(or_test_score)
    yb = plt.ylim()
    plt.plot([thv, thv], [yb[0], yb[1]], 'r-')
    plt.plot([r_f1thv, r_f1thv], [yb[0], yb[1]], 'k--')
    plt.ylim(yb)
    plt.title('PCA{} z{} reconstrustion error, fails {:3.1f} %'.format(NID, Z_SIZE, rec_fail_score))
    plt.legend(['test threshold', 'F1 threshold', f'{NID}s train', f'{NID}s test', 'others test'])
    plt.xlabel('loss')
    plt.savefig(f'pca{NID}_z{Z_SIZE}_s{args.seed}_rec.png')
    plt.cla()
    plt.clf()
    plt.close()
    
    # likelihood error
    lk_f1thv, lk_f1score = f1_threshold(z_test_score, oz_test_score)
    
    thv, pth, nth = density_threshold(z_test_score, oz_test_score)
    lk_fail_score = (((1 - nth) + (1 - pth))*100) / 2
    
    precision = pth / (pth + (1 - nth))
    recall = pth / (pth + (1 - pth))
    lk_f1 = (2*(recall*precision)) / (recall + precision)
    
    sns.distplot(z_train_score)
    sns.distplot(z_test_score)
    sns.distplot(oz_test_score)
    yb = plt.ylim()
    plt.plot([thv, thv], [yb[0], yb[1]], 'r-')
    plt.plot([lk_f1thv, lk_f1thv], [yb[0], yb[1]], 'k--')
    plt.ylim(yb)
    plt.title('PCA{} z{} likelihood, fails {:3.1f} %'.format(NID, Z_SIZE, lk_fail_score))
    plt.legend(['test threshold', 'F1 threshold', f'{NID}s train', f'{NID}s test', 'others test'])
    plt.xlabel('loss')
    plt.savefig(f'pca{NID}_z{Z_SIZE}_s{args.seed}_lk.png')
    plt.cla()
    plt.clf()
    plt.close()
    
    loss_data = np.vstack((
            np.hstack((z_test_score.reshape(-1, 1), r_test_score.reshape(-1, 1))),
            np.hstack((oz_test_score.reshape(-1, 1), or_test_score.reshape(-1, 1)))))
    loss_label = np.concatenate((np.ones((z_test_score.size,)), np.zeros((oz_test_score.size,))))
    
    clf = LogisticRegression().fit(loss_data, loss_label)
    
    with open(f'logit_z{Z_SIZE}.txt', 'a') as f:
        f.write(f'{clf.score(loss_data, loss_label)}\n')
    
    with open(f'metrics_z{Z_SIZE}.txt', 'a') as f:
        f.write(f'{rec_fail_score} {lk_fail_score} {r_f1score} {lk_f1score}\n') # {lk_f1}
    ''' #'''
