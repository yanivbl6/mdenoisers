import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import argparse
import os
from models import dbl_FC
import utils
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def get_noise(data, min_sigma, max_sigma, device):
    noise = torch.randn_like(data, device=device)
    n = noise.shape[0]
    #noise_tensor_array = (max_sigma - min_sigma) * torch.rand(n, device=device) + min_sigma
    noise_tensor_array = 0.5 * torch.ones(size=(n,), device=device) + 0.2 * torch.randint(low=0, high=2, size=(n,), device=device)
    for i in range(n):
        noise.data[i] = noise.data[i] * noise_tensor_array[i];
    return noise, noise_tensor_array

def save_analysis_plot_denoiser_residual(model, sigma_min, sigma_max, save_path, criterion, testloader, device):
    print('==> Saving analysis figure..')
    model.eval()
    sigma_arr = np.linspace(sigma_min, sigma_max, num=20)
    sigma_lst = list(sigma_arr)
    res_norm_lst = []
    noise_norm_lst = []
    mse_lst = []
    with torch.no_grad():
        for sigma in sigma_lst:
            res_norm = []
            noise_norm = []
            test_loss = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                noise = torch.randn_like(inputs, device=device) * sigma
                noisy_inputs = inputs + noise
                outputs = model(noisy_inputs / (2 * (sigma ** 2)))
                res_norm.append(torch.norm(noisy_inputs-outputs, dim=1).cpu().detach().numpy())
                noise_norm.append(torch.norm(noise, dim=1).cpu().detach().numpy())
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
            mse_lst.append(test_loss / (batch_idx + 1))
            res_norm = np.asarray(res_norm)
            noise_norm = np.asarray(noise_norm)
            res_norm_lst.append(np.mean(res_norm))
            noise_norm_lst.append(np.mean(noise_norm))
    res_norm_arr = np.asarray(res_norm_lst)
    noise_norm_arr = np.asarray(noise_norm_lst)
    mse_arr = np.asarray(mse_lst)
    np.save(save_path + '/res_norm_arr.npy',  res_norm_arr)
    np.save(save_path + '/noise_norm_arr.npy', noise_norm_arr)
    np.save(save_path + '/mse_arr.npy', mse_arr)
    sigma_arr = np.asarray(sigma_lst)
    plt.figure()
    plt.plot(sigma_arr, res_norm_arr, label='Residual', marker='o',)
    plt.plot(sigma_arr, noise_norm_arr, label='Noise', linestyle='--')
    plt.legend()
    plt.ylabel('Norms')
    plt.xlabel('Noise std')
    plt.savefig(save_path + "/residual_of_denoiser.png", bbox_inches='tight')
    plt.figure()
    plt.plot(sigma_arr, mse_arr, label='Test MSE')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Noise std')
    plt.savefig(save_path + "/MSE.png", bbox_inches='tight')


def save_loss_figure(train_loss_lst, test_loss_lst, save_path, epochs):
    print('==> Saving loss figure..')
    train_loss_lst = np.asarray(train_loss_lst)
    test_loss_lst = np.asarray(test_loss_lst)
    np.save(save_path + '/train_loss.npy', train_loss_lst)
    np.save(save_path + '/test_loss.npy', test_loss_lst)
    epochs_arr = np.linspace(0, epochs-1, num=epochs)
    plt.figure()
    plt.plot(epochs_arr, train_loss_lst, label='Train loss')
    plt.plot(epochs_arr, test_loss_lst, label='Test loss')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Number of epochs')
    plt.savefig(save_path + "/loss.png", bbox_inches='tight')


def train(model, sigma, epoch, criterion, optimizer, trainloader, device, use_multiple_noise_level = False):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        if use_multiple_noise_level:
            #sigma_i =sigma + 0.2*torch.randint(low=0, high=2, size=(1,),device=device)
            #noise = torch.randn_like(inputs, device=device)*(sigma_i)
            noise, noise_tensor_array = get_noise(inputs, 0, 1, device)
        else:
            sigma_i = sigma
            noise = torch.randn_like(inputs, device=device)*sigma_i
        noisy_inputs = inputs + noise
        optimizer.zero_grad()
        if use_multiple_noise_level:
            print(inputs.size(2))
            print(noise_tensor_array.unsqueeze(3).size())
            outputs = model(torch.div(noisy_inputs, 2 * torch.square(noise_tensor_array).unsqueeze(3).repeat(1, 1, inputs.size(2), inputs.size(3))))
        else:
            outputs = model(noisy_inputs/(2*(sigma_i**2)))
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Loss: %.3f '
                  % (train_loss/(batch_idx+1)))
    return train_loss/(batch_idx+1)


def test(model, sigma, criterion, testloader, device, use_multiple_noise_level = False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            if use_multiple_noise_level:
                noise = torch.randn_like(inputs, device=device) * sigma
                noisy_inputs = inputs + noise
                outputs = model(noisy_inputs / (2 * (sigma ** 2)))
                loss = criterion(outputs, inputs)
                test_loss += 0.5*loss.item()
                noise = torch.randn_like(inputs, device=device) * (sigma + 0.2)
                noisy_inputs = inputs + noise
                outputs = model(noisy_inputs / (2 * (sigma ** 2)))
                loss = criterion(outputs, inputs)
                test_loss += 0.5 * loss.item()
            else:
                noise = torch.randn_like(inputs, device=device) * sigma
                noisy_inputs = inputs + noise
                outputs = model(noisy_inputs/(2*(sigma**2)))
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
            if batch_idx % 100 == 0:
                print('Test Loss: %.3f '
                      % (test_loss / (batch_idx + 1)))
    return test_loss/(batch_idx+1)


def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of spherical image denoiser')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs per task (default: 200)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--use-output-activation', action='store_true', default=False,
                        help='use output activation')
    parser.add_argument('--layers', '-L', default=8, type=int,
                        help='number of layers')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--results-dir', type=str, default="TMP",
                        help='Results dir name')
    parser.add_argument('--sigma', default=0.5, type=float, help='noise std')
    parser.add_argument('--sigma_min', default=0.2, type=float, help='analysis noise std min')
    parser.add_argument('--sigma_max', default=1, type=float, help='analysis noise std max')
    parser.add_argument('--use-multiple-noise-level', action='store_true', default=False,
                        help='use multiple noise level')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    utils.set_seed(args.seed)

    save_path = os.path.join("./logs", str(args.results_dir) + "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if use_cuda else "cpu")

    print('==> Preparing data..')
    dataset = 'mnist'
    input_size = 28
    input_channels = 1
    new_norm = 27.66902 #mean of the norm of mnist images

    lambd = lambda x: norm_change(x, new_norm)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           transforms.Lambda(lambd)
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambd)
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print('==> Building model..')
    model = dbl_FC(input_size, args.layers, new_norm, args.use_output_activation).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    train_loss_lst = []
    test_loss_lst = []
    for epoch in range(args.epochs):
        train_loss = train(model, args.sigma, epoch, criterion, optimizer, train_loader, device, args.use_multiple_noise_level)
        test_loss = test(model, args.sigma, criterion, test_loader, device)
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)

    if args.save_model:
        torch.save(model.state_dict(), save_path + "/mnist_denoiser.pt")

    save_loss_figure(train_loss_lst, test_loss_lst, save_path, args.epochs)
    save_analysis_plot_denoiser_residual(model, args.sigma_min, args.sigma_max, save_path, criterion, test_loader, device)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))