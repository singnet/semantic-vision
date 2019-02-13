import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F

from save_images import save_images
from solver import Solver
from utils import str2bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return th, r

parser = argparse.ArgumentParser(description='toy Beta-VAE')

parser.add_argument('--train', default=False, type=str2bool, help='train or traverse')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')

parser.add_argument('--z_dim', default=20, type=int, help='dimension of the representation z')
parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

parser.add_argument('--dset_dir', default='../data', type=str, help='dataset directory')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
#parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

parser.add_argument('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = ((x_train) / 255.0).reshape((-1, 28, 28, 1))
x_test = ((x_test) / 255.0).reshape((-1, 28, 28, 1))


train_idx_real = np.argwhere(y_train == 3).flatten()
train_idx_fake = np.argwhere(y_train == 5).flatten()
test_idx_real = np.argwhere(y_test == 3).flatten()
test_idx_fake = np.argwhere(y_test == 5).flatten()

net = Solver(args, x_train)

xz_train, xz_test = [], []
xr_train, xr_test = [], []
batch_size = 500

for i in range(0, x_train.shape[0], batch_size):
    x_in = torch.from_numpy(np.transpose(x_train[i:i+batch_size], axes=(0, 3, 1, 2))).float().to(device)
    c_xr, c_xz, _ = net.net(x_in)
    xz_train.append(c_xz.cpu().detach().numpy())
    xr_train.append(np.transpose(F.sigmoid(c_xr).cpu().detach().numpy(), axes=(0, 2, 3, 1)))
xz_train = np.vstack(xz_train)
xr_train = np.vstack(xr_train)

batch_size = 500
for i in range(0, x_test.shape[0], batch_size):
    x_in = torch.from_numpy(np.transpose(x_test[i:i + batch_size], axes=(0, 3, 1, 2))).float().to(device)
    c_xr, c_xz, _ = net.net(x_in)
    xz_test.append(c_xz.cpu().detach().numpy())
    xr_test.append(np.transpose(F.sigmoid(c_xr).cpu().detach().numpy(), axes=(0, 2, 3, 1)))
xz_test = np.vstack(xz_test)
xr_test = np.vstack(xr_test)

xr_train_loss = np.mean(np.square(x_train - xr_train), (1, 2, 3))
xr_test_loss = np.mean(np.square(x_test - xr_test), (1, 2, 3))

save_images(x_test[test_idx_fake][:25], 'Results/Fake_orig.png').astype(np.int)
save_images(xr_test[test_idx_fake][:25], 'Results/Fake_reconstruct.png').astype(np.int)

r_test_loss = xr_test_loss[test_idx_real]
r_fake_loss = xr_test_loss[test_idx_fake]

th_rloss, te_rloss = density_threshold(r_test_loss, r_fake_loss, step=0.0015)

ths_rloss = th_rloss[:, 1] + th_rloss[:, 2]
thv_rloss = te_rloss[np.argmin(ths_rloss)]
print('threshold value:', thv_rloss)
plt.plot(te_rloss, ths_rloss)
plt.title('threshold criterion')
plt.savefig("Results/Threshold_reconstruction.png")
plt.clf()

sns.distplot(r_test_loss)
sns.distplot(r_fake_loss)
yb = plt.ylim()
plt.plot([thv_rloss, thv_rloss], [yb[0], yb[1]])
plt.ylim(yb)
plt.legend(['threshold', '3s (test)', '5s (fake)'])
plt.title('VAE reconstruction error (z size = {})'.format(net.z_dim))
plt.savefig("Results/Division_by_rec_threshold.png")
plt.clf()

print('Threshold value: {0:3.3f}'.format(thv_rloss))
pth = np.argwhere(r_test_loss < thv_rloss).size
nth = np.argwhere(r_fake_loss < thv_rloss).size
print('Positives behind threshold:', '{:3.1f} %'.format((pth / r_test_loss.size)*100), pth, '/', r_test_loss.size)
print('Negatives behind threshold:', '{:3.1f} %'.format((nth / r_fake_loss.size)*100), nth, '/', r_fake_loss.size)
fail_score = (((1 - (pth / r_test_loss.size))*100) + ((nth / r_fake_loss.size)*100)) / 2
print('Fails score:', '{:3.1f} %'.format(fail_score))

sns.distplot(xz_test[test_idx_real].flatten())
sns.distplot(xz_test[test_idx_fake].flatten())
plt.legend(['3s (test)', '5s (fake)'])
plt.title('VAE.z distribution (z size = {})'.format(net.z_dim))
plt.savefig("Results/Z_distr.png")
plt.clf()

axz_test = np.sum(np.square(xz_test[test_idx_real]), 1)
axz_fake = np.sum(np.square(xz_test[test_idx_fake]), 1)
sns.distplot(axz_test)
sns.distplot(axz_fake)
plt.legend(['3s (test)', '5s (fake)'])
plt.title('VAE.z likelihood (z size = {})'.format(net.z_dim))
plt.savefig("Results/z_likelihood.png")
plt.clf()

th_zll, te_zll = density_threshold(axz_test, axz_fake, step=0.05)
th_c = th_zll[:, 1] + th_zll[:, 2]
thv_zll = te_zll[np.argmin(th_c)]
print('threshold value:', thv_zll)
plt.plot(te_zll, th_c)
plt.title('threshold criterion')
plt.savefig("Results/threshold_for_likelihood.png")
plt.clf()

sns.distplot(axz_test)
sns.distplot(axz_fake)
yb = plt.ylim()
plt.plot([thv_zll, thv_zll], [yb[0], yb[1]])
plt.ylim(yb)
plt.legend(['threshold', '3s (test)', '5s (fake)'])
plt.title('VAE.z likelihood (z size = {})'.format(net.z_dim))
plt.savefig("Results/division_by_likelihood.png")
plt.clf()

print('Threshold value: {0:3.3f}'.format(thv_zll))
pth = np.argwhere(axz_test < thv_zll).size
nth = np.argwhere(axz_fake < thv_zll).size
print('Positives behind threshold:', '{:3.1f} %'.format((pth / axz_test.size)*100), pth, '/', axz_test.size)
print('Negatives behind threshold:', '{:3.1f} %'.format((nth / axz_fake.size)*100), nth, '/', axz_fake.size)
fail_score = (((1 - (pth / axz_test.size))*100) + ((nth / axz_fake.size)*100)) / 2
print('Fails score:', '{:3.1f} %'.format(fail_score))
