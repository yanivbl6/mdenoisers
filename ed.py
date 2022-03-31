'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn

import math
from collections import OrderedDict
import torch.nn.functional as F

import numpy as np

import torch.nn.utils.spectral_norm as spectral_norm



def str2act(txt, param= None):
    return {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "none": nn.Sequential() , "lrelu": nn.LeakyReLU(param), "selu": nn.SELU() }[txt.lower()]



class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
   
    def forward(self, x):
        return x.view(x.shape[0],-1) 

def init_layer(op):
    stdv = 1. / math.sqrt(op.weight.size(1)) 
    op.weight.data.uniform_(-stdv, stdv) 
    if op.bias is not None: 
        op.bias.data.fill_(0.0)


class ED_Conv(nn.Module):
    def __init__(self, input_size, layers, tW, tD, skips, act, widening, lamb = 0.0, K = 3, input_channels = 1, strides = 1):
        super(ED_Conv,self).__init__()

        self.act = act


        self.input_size  = input_size
        self.layers  = layers
        self.tD  = tD
        self.tW  = tW
        self.skips = skips
        self.lamb = lamb
        self.use_batchnorm  = False
        layer_specs = [1] + [ 16*widening**i for i in range(layers)]


        layer_specs = layer_specs[0:self.layers+1]

        self.encoders = nn.ModuleList()
        num_strides = 0
        ins = input_size 
        while ins % strides == 0:
            num_strides = num_strides + 1
            ins = ins//strides

        strides_l = np.asarray([(l+1)*num_strides/self.layers for l in range(self.layers)  ]) 
        strides_l = strides_l.round()
        strides_l[1:] = strides_l[1:] - strides_l[:-1]


        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], rounding_needed  = True, strides= strides if strides_l[0] == 1.0 else 1)
        op = nn.Sequential(pad, conv)

        ##op = nn.Linear(layer_specs[0], layer_specs[1], bias = True)
        ##init_layer(op)




        self.encoders.append(op)
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn

            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, strides= strides if strides_l[i+1] == 1.0 else 1)
            if not pad is None:
                d['pad'] = pad

            conv.weight.data.fill_(0.0)

            for j in range(last_ch):
                if j < ch_out:
                    conv.weight.data[j,j,1,1] = 1.0


            d['conv'] = conv

            if self.use_batchnorm:
                d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))

            ##linop = nn.Linear(last_ch, ch_out, bias = True)
            ##init_layer(linop)
            ##d['linear'] = linop

            encoder_block = nn.Sequential(d)

            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()

        for i,ch_out in enumerate(layer_specs[1:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2) 

            conv = self._gen_deconv(last_ch, ch_out , gain = gain, strides= strides if strides_l[self.layers-1-i] == 1.0 else 1, k = 3 if strides == 1 or (not strides_l[self.layers-1-i] == 1.0) else 4)
            conv.weight.data.fill_(0.0)

            if not self.skips:
                for j in range(last_ch):
                    if j < ch_out:
                        conv.weight.data[j,j,1,1] = 1.0



            d['conv'] = conv

            # # if i < self.num_dropout and self.droprate > 0.0:
            # #     d['dropout'] = nn.Dropout(self.droprate)

            if self.use_batchnorm and i < self.layers-1:
                d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            ##linop = nn.Linear(last_ch, ch_out, bias = True)
            ##init_layer(linop)
            ##d['linear'] = linop


            decoder_block = nn.Sequential(d)

            self.decoders.append(decoder_block)
            last_ch = ch_out 

        self.flat = Flatten()


    def forward(self, x):
        src = x
        src_shape = x.shape

        ##x = x.view(src_shape[0],-1) 

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            if self.skips:
                x = x + encoders_output[-(i+2)]

        x = self.decoders[-1](x) 
        x = x + src
        ##x = x.view(src_shape)
        return x

    def _gen_conv(self, in_ch,  out_ch, strides = 2, kernel_size = (3,3), gain = math.sqrt(2), rounding_needed= False):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        ky,kx = kernel_size
        p1x = (kx-1)//2
        p2x = kx-1 - p1x
        p1y = (ky-1)//2
        p2y = ky-1 - p1y

        if rounding_needed:
            pad_counts = (p1x,p1x,p1y , p1y)
            pad = torch.nn.ReplicationPad2d(pad_counts)
        else:
            pad = None

        if pad is None:
            conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides, padding = (p1y, p1x) )
        else:
            conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides , padding=0)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv, pad 

    def _gen_deconv(self, in_ch,  out_ch, strides = 2, k = 4, gain = math.sqrt(2), p = 1 ):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]

        conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size= (k,k), stride = strides, padding_mode='zeros',padding = (p,p), dilation  = 1)

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv


class ED_FC(nn.Module):
    def __init__(self, input_size, layers, tW, tD, skips, act, widening, lamb = 0.0):
        super(ED_FC,self).__init__()

        self.act = act


        self.input_size  = input_size
        self.layers  = layers
        self.tD  = tD
        self.tW  = tW
        self.skips = skips
        self.lamb = lamb

        layer_specs = [ (input_size*input_size)*widening**i for i in range(layers+1)]


        layer_specs = layer_specs[0:self.layers+1]


        self.encoders = nn.ModuleList()

        ##op, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], convGlu = self.convGlu, rounding_needed  = True)
        ##op = nn.Sequential(pad, conv)

        op = nn.Linear(layer_specs[0], layer_specs[1], bias = True)
        init_layer(op)

        
        self.encoders.append(op)
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            # gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            # gain = gain / math.sqrt(2)  ## for naive signal propagation with residual w/o bn
            # conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, convGlu = self.convGlu, kernel_size = self.k_xy)
            # if not pad is None:
            #     d['pad'] = pad
            # d['conv'] = conv

            # if self.use_batchnorm:
            #     d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            linop = nn.Linear(last_ch, ch_out, bias = True)
            init_layer(linop)

            
            d['linear'] = linop
            encoder_block = nn.Sequential(d)

            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()

        for i,ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act,self.lamb)
            # gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            # gain = gain / math.sqrt(2) 

            # if i == len(layer_specs)-2:
            #      kernel_size = 5
            #      ch_out = 2
            # conv = self._gen_deconv(last_ch, ch_out , gain = gain, k= kernel_size)
            # d['conv'] = conv

            # # if i < self.num_dropout and self.droprate > 0.0:
            # #     d['dropout'] = nn.Dropout(self.droprate)

            # if self.use_batchnorm and i < self.n_layers-1:
            #     d['bn']  = nn.BatchNorm2d(ch_out)

            print("last_ch: %d, ch_out: %d " %(last_ch, ch_out))
            linop = nn.Linear(last_ch, ch_out, bias = True)
            init_layer(linop)
            ##d['linear'] = linop

            if self.skips or (not last_ch == ch_out):
                op.weight.data.fill_(0.0)
            else:
                op.weight.data = torch.eye(int(last_ch))

            if op.bias is not None: 
                op.bias.data.fill_(0.0)


            d['linear'] = linop
            decoder_block = nn.Sequential(d)

            self.decoders.append(decoder_block)
            last_ch = ch_out 

        self.flat = Flatten()


    def forward(self, x):
        

        src_shape = x.shape

        x = x.view(src_shape[0],-1) 

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)
            if self.skips:
                x = x + encoders_output[-(i+2)]

        x = self.decoders[-1](x) 

        x = x.view(src_shape)
        return x


def norm_act(input_net, output_net):
    input_net_shape = input_net.shape
    input_net = input_net.view(input_net.size(0), -1)
    output_net = output_net.view(output_net.size(0), -1)
    frac = (torch.norm(output_net, dim=1)**2)/((input_net*output_net).sum(dim=1))
    output_net = output_net * (1 - torch.nn.functional.relu(1-frac)).unsqueeze(1)
    return output_net.view(input_net_shape)


class NormAct(nn.Module):
    def __init__(self):
        super(NormAct, self).__init__() # init the base class
        
    def forward(self, input_net, output_net):
        return norm_act(input_net, output_net)

class dbl_FC(nn.Module):
    def __init__(self, input_size, layers, dbl = True, spectralNorm = False, use_output_activation = False):
        self.use_output_activation = use_output_activation
        if self.use_output_activation:
            self.normact = lambda x, y: norm_act(x, y)
        else:
            self.normact = lambda x, y: y
        d_size = input_size*input_size
        super(dbl_FC, self).__init__()
        self.depth = layers

        
         

        if spectralNorm:
            self.linops = nn.ModuleList([spectral_norm(nn.Linear(d_size, d_size, bias = False)) for _ in range(layers) ])
        else:
            self.linops = nn.ModuleList([nn.Linear(d_size, d_size, bias = False) for _ in range(layers)])
            
        for i,linop in enumerate(self.linops):
            #import pdb; pdb.set_trace()
            #linop.weight.data = torch.eye(int(d_size)) 
            torch.nn.init.eye_(self.linops[i].weight)


        ##self.weights = nn.ParameterList([nn.Parameter(torch.eye(int(d_size)) * (0.001 if i==0 else 1) ) for i in range(layers)])


        ##self.weights = nn.ParameterList([nn.Parameter(torch.randn(int(d_size),int(d_size)) *  1e-1  / math.sqrt(input_size)  ) for i in range(layers)])

        self.dbl = dbl

    def forward(self, input):
        output = input.view(input.size(0), -1)
        output_T = output.clone()
        masks = []
        
        for k in range(self.depth-1):

            output = self.linops[k](output)
            output = F.relu(output)
            mask = (output != 0.0).to(dtype = torch.int)
            masks.append(mask.clone().detach())

        output = self.linops[-1](output)


        if self.dbl:
            for k in range(self.depth-2,-1,-1):

                W  = self.linops[k].weight
                output_T = F.linear(output_T, torch.transpose(W,0,1))
                output_T = output_T * masks.pop()

            W = self.linops[0].weight
            output_T = F.linear(output_T, torch.transpose(W,0,1))
            return self.normact(input ,(output/2 + output_T/2).view(input.shape))
        else:
            return self.normact(input, (output).view(input.shape))


# test()


class dbl_FC2(nn.Module):
    def __init__(self, input_size, layers, dbl = True, spectralNorm = True):


        d_size = input_size*input_size
        super(dbl_FC2, self).__init__()
        self.depth = layers

        
        self.weights = nn.ParameterList([nn.Parameter(torch.eye(int(d_size))) for i in range(layers)])
        
        ##self.weights = nn.ParameterList([nn.Parameter(torch.randn(int(d_size),int(d_size)) *  1e-1  / math.sqrt(input_size)  ) for i in range(layers)])

        self.dbl = dbl

    def forward(self, input):
        output = input.view(input.size(0), -1)
        output_T = output.clone()
        masks = []
        for k in range(self.depth-1):

            
            output = F.linear(output, self.weights[k])
            output = F.relu(output)
            mask = (output != 0.0).to(dtype = torch.int)
            masks.append(mask.clone().detach())

        output = F.linear(output, self.weights[-1])
        if self.dbl:
            for k in range(self.depth-2,-1,-1):
                output_T = F.linear(output_T, torch.transpose(self.weights[k],0,1))
                output_T = output_T * masks.pop()

            output_T = F.linear(output_T, torch.transpose(self.weights[0],0,1))
            return (output/2 + output_T/2).view(input.shape)
        else:
            return (output).view(input.shape)


# test()
