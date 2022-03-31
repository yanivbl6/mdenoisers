import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from torch.utils.data import random_split
from torch.distributions.beta import Beta

import numpy as np
import scipy


import torch.nn as nn
import math
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

import random

def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True

            
class dbl_FC(nn.Module):
    def __init__(self, input_size, layers, dbl = True, spectralNorm = False):


        d_size = input_size*input_size
        super(dbl_FC, self).__init__()
        self.depth = layers

        
        

        if spectralNorm:
            self.linops = nn.ModuleList([spectral_norm(nn.Linear(d_size, d_size, bias = False)) for _ in range(layers) ])
        else:
            self.linops = nn.ModuleList([nn.Linear(d_size, d_size, bias = False)])
            
        for i,linop in enumerate(self.linops):
            linop.weight.data = torch.eye(int(d_size)) * (0.001 if i==0 else 1)
            
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
            return input- (output + output_T).view(input.shape)
        else:
            return input- (output).view(input.shape)
        
            
class dbl_FC_old(nn.Module):
    def __init__(self, input_size, layers, dbl = True, spectralNorm = False):


        d_size = input_size*input_size
        super(dbl_FC_old, self).__init__()
        self.depth = layers
        
        ##import pdb; pdb.set_trace()
        self.weights = nn.ParameterList([nn.Parameter(torch.eye(int(d_size)) * (0.001 if i==0 else 1) ) for i in range(layers)])
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
            return (input - (output + output_T).view(input.shape))
        else:
            return (input - (output).view(input.shape))

        
        
def auto_name(args):
    txt = f"{args.dataset}_{args.arch}_{args.layers}_{args.widening}_"


    if args.tW:
        txt = txt + "tW_"
    if args.tD:
        txt = txt + "tD_"
    if args.single:
        txt = txt + "single_"


    loglr = round(np.log(args.lr)/np.log(10))
    lead = math.ceil(args.lr / (10**loglr))

    lr = f"{lead}E{loglr}"
    txt = txt + f"LR{lr}_"

    if args.strides == 1:
        txt = txt + "strides-%d" % args.strides

    if not args.skips:
        txt = txt + "no-skips_"

    if args.spectral_norm:
        txt = txt + "SpectralNorm_"

    if args.adveserial:
        txt = txt + "Adveserial_"


    ##if not args.optim == "sgd":
    txt = txt + f"{args.optim}_"

    txt = txt + f"sigma{args.sigma}_"

    if len(args.name) > 0:
        txt = txt + args.name
    else:
        txt = txt[:-1]

    return txt

class Arguments():
    def __init__(self,arch, single, layers, dataset, sigma, widening, no_cuda, seed, spectral_norm, adveserial):
        self.arch = arch
        self.tW = False
        self.tD = False
        self.skips = False
        self.strides = 1
        self.single = single
        self.layers = layers
        self.dataset = dataset
        self.arch = arch
        self.sigma = sigma
        self.widening  = widening
        self.no_cuda = no_cuda
        self.seed = seed
        self.spectral_norm = spectral_norm
        self.adveserial = adveserial
        sigma_idx = np.argmin((np.logspace(0,-2,10)-sigma)**2)
        self.name = f"NR-{sigma_idx}-10"


def get_model(sigma = 1.0 , src_dir = "logs", arch = "dbl-fc", dataset = 'mnist', single = False, widening = 1, layers = 8, no_cuda = False, seed =1, spectral_norm = False, new_version =None, adveserial ):
    
    if new_version is None:
        new_version = spectral_norm
    

    args = Arguments(arch, single, layers, dataset, sigma, widening, no_cuda, seed, spectral_norm, adveserial)


    set_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    aname = auto_name(args)
    save_path = os.path.join("./logs", aname + "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == "cifar10":
        dataset = 'cifar10'
        input_size = 32
        input_channels = 3
    else:
        dataset = 'mnist'
        input_size = 28
        input_channels = 1

    # create model
    args.arch = args.arch.lower()
    if args.arch == "dbl-fc":
        if new_version:
            model = dbl_FC(input_size, args.layers, (not args.single), spectralNorm = args.spectral_norm)
        else:
            model = dbl_FC_old(input_size, args.layers, (not args.single), args.spectral_norm)

    else:
        raise ValueError("wrong arch: %f" % args.arch)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    
    re = True
    retxt = ""
    loaded = False
    for retxt in ["","1000","500","200","100"]:
        try:
            load_path = os.path.join(f"./{src_dir}", aname + "/")
            
            print("loading %s" % (load_path + f"ED{retxt}.pt"))

            
            state_dict = torch.load(load_path + f"ED{retxt}.pt")
            model.load_state_dict(state_dict)
            model.eval()
            re = False
            print("loaded %s" % (load_path + f"ED{retxt}.pt"))
            loaded = True
            break
        except:
            pass
    assert(loaded)
    
    _ = model(torch.randn([16,28,28], device = device));

    return model



