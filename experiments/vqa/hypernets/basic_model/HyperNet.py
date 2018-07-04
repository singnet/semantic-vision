import torch.nn as nn
import torch
import math


class HNet(nn.Module):
    def __init__(self, control_dims, main_dims):
        super(HNet, self).__init__()
        q_in_dim = control_dims[0]
        inter_control_dim = control_dims[1]
        control_mat_size = control_dims[2]
        
        v_in_dim = main_dims[0]
        out_dim = main_dims[1]
        
        #control part
        hyper_control_layers = []
        hyper_control_layers.append(nn.Linear(q_in_dim, inter_control_dim))
        hyper_control_layers.append(nn.ReLU())
        hyper_control_layers.append(nn.Linear(inter_control_dim, control_mat_size*control_mat_size))
        self.hyper_control = nn.Sequential(*hyper_control_layers)

        #left half of main network
        hyper_main_layers_in = []
        hyper_main_layers_in.append(nn.Linear(v_in_dim, control_mat_size))
        hyper_main_layers_in.append(nn.ReLU())
        self.hyper_main_in = nn.Sequential(*hyper_main_layers_in)

        #right part of main network
        hyper_main_layers_out = []
        hyper_main_layers_out.append(nn.Linear(control_mat_size, out_dim))
        hyper_main_layers_out.append(nn.ReLU())
        self.hyper_main_out = nn.Sequential(*hyper_main_layers_out)

    def forward(self, q, v):
        control_mat = self.hyper_control(q)
        c_m_size = list(control_mat.size())
        left_main_part = self.hyper_main_in(v)
        control_mat = control_mat.reshape((-1, int(math.sqrt(c_m_size[1])), int(math.sqrt(c_m_size[1]))))
        left_main_part = torch.einsum('bj,bjk->bk', (left_main_part, control_mat))
        return self.hyper_main_out(left_main_part)
