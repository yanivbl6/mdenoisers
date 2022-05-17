import torch
import torch.nn as nn
import torch.nn.functional as F
import basicblock as B


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


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR', new_norm=1 ,use_output_activation=False):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.use_output_activation = use_output_activation
        self.new_norm = new_norm
        self.normact = NormAct()
    def forward(self, x):
        n = self.model(x)
        if self.use_output_activation:
            return self.normact(x-n, self.new_norm)
        else:
            return x-n



class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, new_norm=1 ,use_output_activation=False):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)
        self.use_output_activation = use_output_activation
        self.new_norm = new_norm
        self.normact = NormAct()

    def forward(self, x):
        n = self.model(x)
        if self.use_output_activation:
            return self.normact(x - n, self.new_norm)
        else:
            return x - n
