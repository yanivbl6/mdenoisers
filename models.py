import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_act(output_net, new_norm):
    output_net_shape = output_net.shape
    output_net = output_net.view(output_net.size(0), -1)
    frac = new_norm / (torch.norm(output_net, dim=1))
    output_net = output_net * (1 - torch.nn.functional.relu(1 - frac)).unsqueeze(1)
    return output_net.view(output_net_shape)


class NormAct(nn.Module):
    def __init__(self):
        super(NormAct, self).__init__()  # init the base class

    def forward(self, output_net, new_norm):
        return norm_act(output_net, new_norm)


class dbl_FC(nn.Module):
    def __init__(self, input_size, layers, new_norm, use_output_activation=False):
        self.use_output_activation = use_output_activation
        self.new_norm = new_norm
        d_size = input_size * input_size
        super(dbl_FC, self).__init__()
        self.depth = layers
        self.linops = nn.ModuleList([nn.Linear(d_size, d_size, bias=False) for _ in range(layers)])

        for i, linop in enumerate(self.linops):
            torch.nn.init.eye_(self.linops[i].weight)

        self.normact = NormAct()

    def forward(self, input):
        output = input.view(input.size(0), -1)
        masks = []

        for k in range(self.depth - 1):
            output = self.linops[k](output)
            output = F.relu(output)
            mask = (output != 0.0).to(dtype=torch.int)
            masks.append(mask.clone().detach())

        output = self.linops[-1](output)

        if self.use_output_activation:
            return self.normact(output.view(input.shape), self.new_norm)
        else:
            return output.view(input.shape)