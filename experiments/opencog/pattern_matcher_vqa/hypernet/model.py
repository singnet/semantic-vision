import torch.nn as nn
import torch
from .Q_net import QFCNet, WEmbed
from .F_net import VFCNet
from .ProbNet import ProbNet

class Model(nn.Module):
    def __init__(self, w_embed, q_fc_net, v_fc_net, prob_net):
        super(Model, self).__init__()
        self.w_embed = w_embed
        self.q_fc_net = q_fc_net
        self.v_fc_net = v_fc_net
        self.prob_net = prob_net

    def forward(self, word, bb_feature):

        w_emb = self.w_embed(word)

        q_fced = self.q_fc_net(w_emb)
        v_fced = self.v_fc_net(bb_feature)

        joint = q_fced * v_fced
        probability = self.prob_net(joint)
        return probability

def build_baseline_model(voc_size, q_dims, v_dims, prob_dims, inter_dim_nums, network_params):
    q_inter_dims = inter_dim_nums[0]
    v_inter_dims = inter_dim_nums[1]
    prob_inter_dims = inter_dim_nums[2]

    dropout = network_params[0]
    norm = network_params[1]
    activation = network_params[2]

    w_embed = WEmbed(voc_size, q_dims[0], dropout)
    q_fc_net = QFCNet(q_dims, q_inter_dims, dropout, norm, activation)
    v_fc_net = VFCNet(v_dims, v_inter_dims, dropout, norm, activation)
    prob_net = ProbNet(prob_dims, prob_inter_dims, dropout, norm, activation)
    return Model(w_embed, q_fc_net, v_fc_net, prob_net)
