import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import random_split
import utils
import math




from ed import *
from torch.distributions.beta import Beta

import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
#from tensorboard_logger import configure, log_value

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

##from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--aug-lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--no-nesterov', dest='nesterov', default=True,  action='store_false', help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq_train', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--print-freq_test', default=20, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dir', type=str, default="TMP", help='Results dir name')
parser.add_argument('--optim', type=str, default="sgd", help='optimizer')



parser.add_argument('--cont', action='store_true', default=False,
                    help='just load checkpoint')

parser.add_argument('--cont-src', default="", type=str,
                    help='source to load weights from')

parser.add_argument('--name', type=str, default="", help='Name of event file')

parser.add_argument('--arch', default="ED-FC", type=str,
                    help='architecture')

parser.add_argument('-D', '--tD', action='store_true', default=False,
                    help='transposable activation')

parser.add_argument('-W', '--tW', action='store_true', default=False,
                    help='transposable weights')

parser.add_argument('--single',  action='store_true', default=False,
                    help='single version of transposed net')

parser.add_argument('--old',  action='store_true', default=False,
                    help='use old version')

parser.add_argument('--spectral-norm',  action='store_true', default=False,
                    help='Spectral Normalization')

parser.add_argument('--use-output-activation',  action='store_true', default=False,
                    help='use output activation')

parser.add_argument('--no-skips', dest= 'skips', action='store_false', default=True,
                    help='use skips')

parser.add_argument('--no-cosine', dest= 'cosine', action='store_false', default=True,
                    help='dont use cosine annealing')


parser.add_argument('--adveserial', action='store_true', default=False,
                    help='adveserial')

parser.add_argument('--layers', '-L', default=4, type=int,
                    help='number of layers')


parser.add_argument('--start-epoch', default=0, type=int,
                    help='epoch to start from')

parser.add_argument('-w', '--widening', default=2, type=int,
                    help='widening factor')

parser.add_argument('--strides', default=1, type=int,
                    help='strides')

parser.add_argument('-a', '--act', default="relu", type=str,
                    help='activation')

parser.add_argument('--dataset', default="mnist", type=str,
                    help='activation')


parser.add_argument('--noise-up', default=1, type=float, help='Noise level Up (default: 1.00)')
parser.add_argument('--noise-down', default=0.01, type=float, help='Noise level Down (default: 0.01)')



parser.set_defaults(augment=True)


def parse_txt(cmd):
    return parser.parse_args(cmd.replace("  "," ").split(" "))

def count_parameters(model):
    return None
    # table = PrettyTable(["Modules", "Parameters"])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
    #     param = parameter.numel()
    #     table.add_row([name, param])
    #     total_params+=param
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params

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

    if args.use_output_activation:
        txt = txt + "OutputActivation_"

    ##if not args.optim == "sgd":
    txt = txt + f"{args.optim}_"

    txt = txt + f"{args.noise_down}-{args.noise_up}_"

    if len(args.name) > 0:
        txt = txt + args.name
    else:
        txt = txt[:-1]

        


    return txt


def get_model(args):


    utils.set_seed(args.seed)


    print(args.noise_down)
    print(args.noise_up)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading code




    print('==> Preparing data..')

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.dataset == "cifar10":


        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        dataset = 'cifar10'
        input_size = 32
        input_channels = 3
    else:
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

        dataset = 'mnist'
        input_size = 28
        input_channels = 1

    transform=transforms.Compose([
        transforms.ToTensor()])
        ##normalize
        ##])

    aname = auto_name(args)
    save_path = os.path.join("/results/logs", aname + "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ds_train = datasets.__dict__[dataset.upper()](root='../data', train=True, download=True, transform=transform)
    ds_val = datasets.__dict__[dataset.upper()](root='../data', train=False, transform=transform)


    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, **kwargs)



    # create model
    args.arch = args.arch.lower()
    if args.arch == "ed-fc":
        model = ED_FC(input_size, args.layers, args.tW, args.tD, args.skips, args.act, args.widening, lamb = 0.0)
    elif args.arch == "ed-conv":
        model = ED_Conv(input_size, args.layers, args.tW, args.tD, args.skips, args.act, args.widening, lamb = 0.0, input_channels = input_channels, strides = args.strides)
    elif args.arch == "dbl-fc":
        if args.old:
            model = dbl_FC2(input_size, args.layers, (not args.single))
        else:
            model = dbl_FC(input_size, args.layers, (not args.single), args.spectral_norm, args.use_output_activation)

    else:
        raise ValueError("wrong arch: %f" % args.arch)
    # get the number of model parameters
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    count_parameters(model)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    if args.cont:
        if len(args.cont_src)  == 0:
            re = True
            retxt = ""
            for retxt in ["","500","200","100"]:
                try:
                    load_path = os.path.join("./logs", aname + "/")
                    print("loading %s" % (load_path + f"ED{retxt}.pt"))

                    state_dict = torch.load(load_path + f"ED{retxt}.pt")
                    model.load_state_dict(state_dict)
                    model.eval()
                    re = False
                    print("loaded %s" % (load_path + f"ED{retxt}.pt"))
                    break
                except:
                    pass
        else:
            load_path = args.cont_src
            state_dict = torch.load(load_path)
            model.load_state_dict(state_dict)
            model.eval()
            re = False
            print("loaded %s" % load_path)
                

    return model, dl_train, dl_val, device, aname, save_path , input_size

def main():
    global args
    args = parser.parse_args()
    
    model, dl_train, dl_val, device, aname, save_path, input_size = get_model(args)

    # define loss function (criterion) and optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cosine learning rate
    ##scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ((len(train_loaders)*args.levels)//args.M)*args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dl_train))*args.epochs)


    print(f"name: {aname}")
    writer = SummaryWriter(log_dir="/results/runs/%s" % aname, comment=str(args))

    zpack = None
    if args.adveserial:
        z = torch.randn(1,input_size,input_size, device = device)
        zvec = torch.autograd.Variable(z / z.norm() , requires_grad = True)
        if args.optim == "sgd":
            zoptimizer = torch.optim.SGD([zvec], args.lr,
                                        momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
        elif args.optim == "adam":
            zoptimizer = torch.optim.Adam([zvec], lr=args.lr)

        zpack  = (zvec, zoptimizer)

    loss_lst = []
    test_loss_lst = []
    noise_range = (args.noise_down, args.noise_up)
    for epoch in range(args.start_epoch,args.epochs):

        train_loss = train(dl_train, model, noise_range  , optimizer, scheduler, epoch, device, writer, zpack)
        test_loss = test(dl_val, model, noise_range, device, epoch , writer)
        loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)
        if epoch in [100,200,500,1000]:
            if args.save_model:
                torch.save(model.state_dict(), save_path + f"/ED{epoch}.pt")


    print('Finished training!')
    print('Train Loss: %.3f | Test Loss: %.3f' % (train_loss, test_loss))



    loss_lst = np.asarray(loss_lst)
    test_loss_lst = np.asarray(test_loss_lst)
    np.save(save_path + '/train_loss.npy', loss_lst)
    np.save(save_path + '/test_loss.npy', test_loss_lst)
    epochs = np.arange(args.epochs)

    plt.switch_backend('agg')

    plt.figure()
    plt.plot(epochs, test_loss_lst, label='Test Loss')
    plt.plot(epochs, loss_lst, label='Train Loss')
    plt.legend()
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Number of epochs')
    plt.savefig(save_path + "/accuracy.png", bbox_inches='tight')
    plt.figure()
    plt.plot(epochs, loss_lst, label='loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of epochs')
    plt.savefig(save_path + "/loss.png", bbox_inches='tight')

    if args.save_model:
        torch.save(model.state_dict(), save_path + "/ED.pt")


def train(train_loader, model, noise_range, optimizer, scheduler, epoch, device, writer=None, zpack = None):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    total = 0

    if not zpack is None:
        zvec, zoptimizer = zpack
        ztrain_loss = 0

    second_loss = False
    ##import pdb; pdb.set_trace()
    for i, (input, _) in tqdm(enumerate(train_loader), total = len(train_loader)):


        if not zpack is None:
            zoptimizer.zero_grad()
            zout = model(zvec/zvec.norm())
            zloss = -zout.norm()
            zloss.backward()
            zoptimizer.step()
            ztrain_loss += zloss.item()
            second_loss = (zout.detach().norm().item() >= 1)
        
        optimizer.zero_grad()
        input = input.to(device)
        input = input / torch.sqrt((input**2).sum(dim=[1,2,3], keepdim= True))
        sigma = torch.rand([input.size(0),1,1,1],device = device)*(noise_range[1]-noise_range[0]) + noise_range[0] 

        noise = torch.randn(input.shape, device = device) *sigma
        output = model((input + noise)/sigma**2)



        loss = ((output - input)**2).mean()

        if second_loss:
            zout2 = model(zvec/zvec.norm())
            zloss2 = zout2.norm()
            loss = loss + zloss2


        loss.backward()
        train_loss += loss.item()

        total += input.size(0)

        optimizer.step()
        if args.cosine:
            scheduler.step()

    if writer is not None:
        writer.add_scalar('train_loss', train_loss/(i+1), epoch)
        if not zpack is None:
            writer.add_scalar('ztrain_loss', ztrain_loss/(i+1), epoch)


    return train_loss/(i+1)

def test(test_loader, model, noise_range, device, epoch, writer=None):
    """Perform test on the test set"""
    # switch to evaluate mode
    model.eval()
    test_loss = 0
    total = 0
    for i, (input, _) in tqdm(enumerate(test_loader), total = len(test_loader)):
        input = input.to(device)

        # compute output
        with torch.no_grad():

            input = input / torch.sqrt((input**2).sum(dim=[1,2,3], keepdim= True))
            sigma = torch.rand([input.size(0),1,1,1],device = device)*(noise_range[1]-noise_range[0]) + noise_range[0] 
            noise = torch.randn(input.shape, device = device) *sigma
            output = model(((input + noise)/sigma)**2)
            loss = ((output - input)**2).mean()



        test_loss += loss.item()

        total += input.size(0)
        # if i % args.print_freq_test == 0:
        #     print('Test Loss: %.3f' % (test_loss / (i + 1)))

    if writer is not None:
        writer.add_scalar('test_loss', test_loss/(i+1), epoch)
    
    print('Test Loss: %.5f' % (test_loss / (i + 1)))

    return test_loss / (i+1)



if __name__ == '__main__':
    main()
