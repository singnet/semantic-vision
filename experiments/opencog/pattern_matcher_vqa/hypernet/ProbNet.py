import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm

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


class ProbNet(nn.Module):
    def __init__(self, dims, inter_layers_num, dropout, norm, act):
        super(ProbNet, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        layers = []

        for i in range(inter_layers_num-3):
            layers.append(norm_layer(nn.Linear(dims[i], dims[i+1]), dim=None))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))

        layers.append(norm_layer(nn.Linear(dims[inter_layers_num-3], dims[inter_layers_num-2]), dim=None))
        layers.append(act_layer())
        layers.append(norm_layer(nn.Linear(dims[inter_layers_num-2], dims[inter_layers_num-1]), dim=None))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, joint):
        return self.fc(joint)
