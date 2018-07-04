import torch.nn as nn

class HNetControl(nn.Module):
    def __init__(self, dims):
        super(HNetControl, self).__init__()
        q_in_dim = dims[0]
        inter_dim = dims[1]
        out_dim = dims[2]
        hyper_control_layers = []
        hyper_control_layers.append(nn.Linear(q_in_dim, inter_dim))
        hyper_control_layers.append(nn.ReLU())
        hyper_control_layers.append(nn.Linear(inter_dim, out_dim*out_dim))
        self.main = nn.Sequential(*hyper_control_layers)

    def forward(self, x):
        return self.main(x)

class HNetMain(nn.Module):
    def __init__(self):
        super(HNetMain, self).__init__()
        hyper_main_layers = []
        hyper_main_layers.append(nn.ReLU())
        self.main = nn.Sequential(*hyper_main_layers)

    def forward(self, x):
        return self.main(x)
