import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import argparse
import logging
import sys
# import dill

parser = argparse.ArgumentParser(description='MNIST Denosing')
parser.add_argument('--device', default='cuda:0', help='device assignment ("cpu:0" or "cuda:0"/"cuda:1")')
parser.add_argument('--batch-size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--epochs', default=1000, type=int, help='epochs (default: 1000)')
parser.add_argument('--lr', default=5e-1, type=float, help='lr (default: 5e-1)')
parser.add_argument('--seed', default=-0, type=int, help='random seed (default: random)')
parser.add_argument('--file-name', default='test', help='save file name (default: test)')
parser.add_argument('--noise-up', default=1, type=float, help='Noise level Up (default: 1.00)')
parser.add_argument('--noise-down', default=0.01, type=float, help='Noise level Down (default: 0.01)')

parser.add_argument('--m', default=6, type=int, help='Number of layers (default: 6)') ##
parser.add_argument('--i', default='y', help='Identity initialization (y or n)')
parser.add_argument('--cont', action='store_true', default=False,
                    help='just load checkpoint')



def parse_txt2(cmd):
    return parser.parse_args(cmd.split(" "))


def get_model2(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parameters
    dev = args.device

    # New values

    m = args.m
    d_vec = 784*np.ones(m+1, dtype=int)
    ITERS = args.epochs  # 4000
    batch_size = args.batch_size
    lr = args.lr
    momentum = 0
    log_interval = 100
    
    Global_filename = 'globalsave'+args.file_name+'.pkl'
    filename = 'cp'+ args.file_name+'.pt'
    log_filename = 'log'+ args.file_name+'.txt'
    N_points = 1000
    epsilon = 1e-6
    delta = 1
    identity = args.i

    # Network architecture
    class MyNet(nn.Module):
        def __init__(self, m, d_vec):
            super(MyNet, self).__init__()
            self.depth = m
            self.hidden = nn.ModuleList()
            for k in range(m):
                self.hidden.append(nn.Linear(d_vec[k], d_vec[k+1], bias=False))
                if identity == 'y':
                    self.hidden[k].weight.data = torch.eye(int(d_vec[k]))
                    #self.hidden[k].bias.data = torch.zeros(d_vec[k+1])

        def forward(self, input):
            output = input.view(input.size(0), -1)
            for k in range(self.depth-1):
                output = F.relu(self.hidden[k](output))
            output = self.hidden[self.depth-1](output)
            return output

    # Training data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=False)

    dataMean = torch.zeros(28, 28)
    for data, target in train_loader:
        dataMean = dataMean+data.sum(dim=0).squeeze()
    dataMean = dataMean/len(train_loader.dataset)
    dataMean = torch.zeros(dataMean.size())
    dataMean = dataMean.to(dev)

    # Testing data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=False)


    # -------------------------- Main script -------------------------- #

    # Initialization
    myModel = MyNet(m, d_vec)
    myModel.to(dev)

    save_path = "./net2/"+filename

    if args.cont:
        load_path = save_path
        state_dict = torch.load(save_path)
        myModel.load_state_dict(state_dict)
        myModel.eval()

    return myModel, train_loader, test_loader, dev, args.file_name, save_path, (args.noise_down, args.noise_up)





if __name__ == '__main__':
    DBUG = 0

    plt.switch_backend('Qt5Agg')

    #  Seed
    args = parser.parse_args()
    myModel, train_loader, test_loader, device, args.file_name, save_path, noise_range = get_model2(args)


    # Training function
    def train(model, device, train_loader, optimizer, epoch, noise_range):
        model.train()
        train_loss = 0
        print('m = '+str(m) + ', lr = ' + str(lr) + ', dev = ' + dev)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data - data-dataMean.view(1, 1, 28, 28)
            data = data / torch.sqrt((data**2).sum(dim=[1,2], keepdim= True))

            optimizer.zero_grad()
            noise_std = torch.rand()*(noise_range[1]-noise_range[0]) + noise_range[0] 
            noise = torch.normal(0, noise_std,  size=data.size(), device=device)
            output = model((data+noise)/noise_std**2)
            loss = F.mse_loss(output, data.view(data.size(0), -1))
            train_loss += loss*len(data)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        print('Average train loss value: ', train_loss.item()/len(train_loader.dataset))
        return train_loss.item()/len(train_loader.dataset)

    # Testing function
    def test(model, device, test_loader, noise_range):
        model.eval()
        err = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data - data-dataMean.view(1, 1, 28, 28)
                data = data / torch.sqrt((data**2).sum(dim=[1,2], keepdim= True))
                noise_std = torch.rand()*(noise_range[1]-noise_range[0]) + noise_range[0] 
                noise = torch.normal(0, noise_std, size=data.size(), device=device)
                output = model((data+noise)/noise_std**2)
                err += F.mse_loss(output, data.view(data.size(0), -1))*len(data)

        if DBUG:
            test_image = data[1]
            test_image = test_image.to(dev)
            noise = torch.normal(0, noise_std, size=test_image.size(), device=device)
            out_image = model(test_image.view(1, -1)-dataMean.view(1, -1) + noise.view(1, -1))+dataMean.view(1, -1)
            noise_image = noise+test_image
            plt.figure("Examples")
            plt.subplot(1, 3, 1)
            plt.title("Original image")
            plt.imshow(test_image.view(28, 28).cpu().data.numpy())
            plt.subplot(1, 3, 2)
            plt.title("Noisy image")
            plt.imshow(noise_image.view(28, 28).cpu().data.numpy())
            plt.subplot(1, 3, 3)
            plt.imshow(out_image.view(28, 28).cpu().data.numpy())
            plt.title("Denoised image")

        print('\nTest set: {:.6f}\n'.format(err.item()/len(test_loader.dataset)))
        return err.item()/len(test_loader.dataset)


    myGD = optim.SGD(myModel.parameters(), lr=lr, momentum=momentum)
    TrainingError = np.zeros((1, ITERS))
    TestError = np.zeros((1, ITERS))


    if DBUG:
        plt.ion()
        fig = plt.figure()
        Training_line = plt.plot(0, 0, '-')[0]
        Test_line = plt.plot(0, 0, '-')[0]
        ax = plt.gca()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. epoch")

    # Training procedure
    t = time.time()
    for epoch in range(0, ITERS):
        # if epoch == 130:
        #     for g in myGD.param_groups:
        #         g['lr'] = 10e-2
        TrainingError[0, epoch] = train(myModel, dev, train_loader, myGD, epoch, noise_range)
        LayerNorm = np.zeros((1, m))
        for ii in range(m):
            LayerNorm[0, ii] = myModel.hidden[ii](torch.eye(784, device=dev)).cpu().detach().norm().numpy()
        print(LayerNorm)

        TestError[0, epoch] = test(myModel, dev, test_loader, noise_range)

        if DBUG:
            Training_line.set_data(np.arange(epoch+1), TrainingError[0, 0:epoch+1])
            Test_line.set_data(np.arange(epoch+1), TestError[0, 0:epoch+1])
            ax.set_xlim(auto=False, xmin=0, xmax=epoch+1)
            ax.set_ylim(auto=False, ymin=0, ymax=1.1*TrainingError.max())
            plt.pause(0.001)

    elapsed = time.time() - t
    print(elapsed)


    # -------------------------- Lambda max -------------------------- #
    TrainData = train_loader.dataset.data.float()/255
    TrainData = TrainData.to(dev)
    noise_std = torch.rand()*(noise_range[1]-noise_range[0]) + noise_range[0] 

    noise = torch.normal(0, noise_std, size=TrainData.size(), device=dev)
    NoisyTrainData = TrainData+noise
    NoisyTrainData.requires_grad = False
    TrainData.requires_grad = False
    v = torch.randn(size=(m, 784, 784), device=dev)
    v = v/v.norm(p='fro')
    v.requires_grad = False
    myModel.train()
    delta_v = torch.tensor(1, device=dev)

    del train_loader, myGD, test_loader, noise
    torch.cuda.empty_cache()
    n_episodes = 100
    delta_examples = round(60000/n_episodes)
    myFlag = True
    full_grad = torch.zeros(size=(m, 784, 784), device=dev)
    threshold = 0.035

    while delta_v > epsilon:
        cumm_v = torch.zeros(size=(m, 784, 784), device=dev)
        for ii in range(n_episodes):
            currentNoisyData = NoisyTrainData[ii*delta_examples:(ii+1)*delta_examples, :, :]
            currentData = TrainData[ii*delta_examples:(ii+1)*delta_examples, :, :]
            output = myModel(currentNoisyData)
            loss = F.mse_loss(output, currentData.view(currentData.size(0), -1))*delta_examples/TrainData.size(0)
            grd = torch.autograd.grad(loss, myModel.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            temp = torch.zeros(size=[1], device=dev)
            for jj in range(m):
                temp += torch.tensordot(a=grd[jj], b=v[jj], dims=[[0, 1], [0, 1]])
                if myFlag:
                    full_grad[jj] += grd[jj].detach().clone()

            newV_tuple = torch.autograd.grad(temp, myModel.parameters(), retain_graph=False, create_graph=False, allow_unused=True)
            for jj in range(m):
                cumm_v[jj] += newV_tuple[jj].detach()
        oldV = v.clone()
        lambda_max = cumm_v.norm(p='fro')
        v = cumm_v.clone()/lambda_max
        delta_v = torch.norm(v-oldV, p='fro')
        print(lambda_max)
        del newV_tuple, temp, loss, grd, output, oldV, cumm_v
        myFlag = False
        torch.cuda.empty_cache()
        if TrainingError[0, -1] > threshold:
            delta_v = epsilon/2


    # Show results
    print(lambda_max)

    # Get gradient norm
    # output = myModel(NoisyTrainData)
    # loss = F.mse_loss(output, TrainData.view(TrainData.size(0), -1))
    # grd = torch.autograd.grad(loss, myModel.parameters(), retain_graph=False, create_graph=False, allow_unused=True)
    GradNorm = full_grad.norm(p='fro')
    # for ii in range(m):
    #     GradNorm += grd[ii].norm().pow(2)
    # del grd
    # torch.cuda.empty_cache()

    # Save work space variables
    # dill.dump_session(Global_filename) # To restore the session just type dill.load_session(Global_filename)

    
    torch.save(myModel.state_dict(),  save_path )

    logging.basicConfig(level=logging.DEBUG, filename='./logs/'+log_filename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Batch size = ' + str(batch_size))
    logging.info('Epochs = ' + str(ITERS))
    logging.info('Number of layers = ' + str(m))
    logging.info('Learning rate = ' + str(lr))
    logging.info('Momentum = ' + str(momentum))
    logging.info('Noise STD = ' + str(noise_std))
    logging.info('Seed = ' + str(seed))
    logging.info('Train loss = ' + str(TrainingError[0, -1]))
    logging.info('Test loss = ' + str(TestError[0, -1]))
    logging.info('Gradient norm = ' + str(GradNorm.item()))
    logging.info('Lambda max = ' + str(lambda_max.item()))

    if TrainingError[0, -1] > threshold:
        sys.exit(1)
