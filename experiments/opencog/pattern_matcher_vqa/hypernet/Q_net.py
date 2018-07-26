import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm
import numpy as np

def get_norm(norm):
    no_norm = lambda x, dim: x
    if norm == 'weight':
        norm_layer = weight_norm
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = no_norm
    else:
        print("Invalid Normalization")
        raise Exception("Invalid Normalization")
    return norm_layer


def get_act(act):
    if act == 'ReLU':
        act_layer = nn.ReLU
    elif act == 'LeakyReLU':
        act_layer = nn.LeakyReLU
    elif act == 'PReLU':
        act_layer = nn.PReLU
    elif act == 'RReLU':
        act_layer = nn.RReLU
    elif act == 'ELU':
        act_layer = nn.ELU
    elif act == 'SELU':
        act_layer = nn.SELU
    elif act == 'Tanh':
        act_layer = nn.Tanh
    elif act == 'Hardtanh':
        act_layer = nn.Hardtanh
    elif act == 'Sigmoid':
        act_layer = nn.Sigmoid
    else:
        print("Invalid activation function")
        raise Exception("Invalid activation function")
    return act_layer

class WEmbed(nn.Module):
    def __init__(self, ntoken, emb_dim, dropout):
        super(WEmbed, self).__init__()

        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, word):
        emb = self.emb(word)
        emb = self.dropout(emb)
        return emb


class QFCNet(nn.Module):
    def __init__(self, dims, inter_layers_num, dropout, norm, act):
        super(QFCNet, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)
        layers = []

        for i in range(inter_layers_num-1):
            layers.append(norm_layer(nn.Linear(dims[i], dims[i+1]), dim=None))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))


        self.fc = nn.Sequential(*layers)

    def forward(self, wemb):
        return self.fc(wemb)
