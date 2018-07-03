import os
import time
import argparse
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
from models import build_model_A3x2

import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = './data/saved_models/A3x2_1280_LeakyReLU_Adamax_D0.3_DL0.1_DG0.2_DW0.4_DC0.5_w0_SD9731_initializer_kaiming_normal/model.pth'
MODEL_PATH = './saved_models/A3x2_1280_LeakyReLU_Adamax_D0.3_DL0.1_DG0.2_DW0.4_DC0.5_w0_SD9731_initializer_kaiming_normal/model.pth'
BATCH_SIZE = 512

def weights_init_ku(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data, a=0.01)

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1280) # they used 1024
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_L', type=float, default=0.1)
    parser.add_argument('--dropout_G', type=float, default=0.2)
    parser.add_argument('--dropout_W', type=float, default=0.4)
    parser.add_argument('--dropout_C', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='PReLU, ReLU, LeakyReLU, Tanh, Hardtanh, Sigmoid, RReLU, ELU, SELU')
    parser.add_argument('--norm', type=str, default='weight', help='weight, batch, layer, none')
    parser.add_argument('--model', type=str, default='A3x2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adamax', help='Adam, Adamax, Adadelta, RMSprop')
    parser.add_argument('--initializer', type=str, default='kaiming_normal')
    parser.add_argument('--seed', type=int, default=9731, help='random seed')
    args = parser.parse_args()
    return args

def evaluate(model, dataloader):
    score = 0
    V_loss = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        a = Variable(a, volatile=True).cuda()
        pred = model(v, b, q, None)
        loss = instance_bce_with_logits(pred, a)
        V_loss += loss.data[0] * v.size(0)
        batch_score = compute_score_with_logits(pred, a.data).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    V_loss /= len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    return score, upper_bound, V_loss


args = parse_args()
dictionary = Dictionary.load_from_file('./data/dictionary.pkl')

# test_dset = VQAFeatureDataset('test', dictionary)
eval_dset = VQAFeatureDataset('val', dictionary)
model = build_model_A3x2(eval_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
model = model.cuda()
model.w_emb.init_embedding('data/glove6b_init_300d.npy')
model = nn.DataParallel(model).cuda()

# model.apply(weights_init_ku)

ckpt = torch.load(MODEL_PATH)
model.load_state_dict(ckpt)

eval_loader  = DataLoader(eval_dset, BATCH_SIZE, shuffle=True, num_workers=4)
# test_loader  = DataLoader(test_dset, BATCH_SIZE, shuffle=True, num_workers=4)

model.train(False)

eval_score, bound, V_loss = evaluate(model, eval_loader)
print('\teval loss: %.3f, score: %.3f (%.3f)' % (V_loss, 100 * eval_score, 100 * bound))

# test_score, bound, V_loss = evaluate(model, test_loader)
# print('\ttest loss: %.3f, score: %.3f (%.3f)' % (V_loss, 100 * test_score, 100 * bound))




