import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import argparse
import os
from models import dbl_FC, DnCNN
from dataset import DatasetBSD, DatasetBSD68
import utils
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from get_images import generate_denoiser_images


def calc_norm(data, gray_scale=False):
    if gray_scale:
        data_norm = data.norm(dim=(2, 3)).squeeze(1)
    else:
        data_norm = torch.norm(data.norm(dim=(2, 3)), dim=1)
    return data_norm

def get_noise(data, min_sigma, max_sigma, device):
    noise = torch.randn_like(data, device=device)
    n = noise.shape[0]
    noise_tensor_array = (max_sigma - min_sigma) * torch.rand(n, device=device) + min_sigma
    for i in range(n):
        noise.data[i] = noise.data[i] * noise_tensor_array[i]
    return noise, noise_tensor_array


def save_analysis_plot_denoiser_residual(model, sigma_min, sigma_max, save_path, criterion, testloader, new_norm, input_size, input_channels, gray_scale, device):
    print('==> Saving analysis figure..')
    model.eval()
    d = input_size * input_channels
    sigma_arr = np.linspace(sigma_min, sigma_max, num=20)
    sigma_lst = list(sigma_arr)
    res_norm_lst = []
    noise_norm_lst = []
    mse_lst = []
    psnr_lst = []
    with torch.no_grad():
        for sigma in sigma_lst:
            res_norm = []
            noise_norm = []
            test_loss = 0
            psnr = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                noise = torch.randn_like(inputs, device=device) * sigma
                noisy_inputs = inputs + noise
                norm_frac = (torch.sqrt(calc_norm(noisy_inputs, gray_scale)**2 - d * (sigma**2))/new_norm)
                #norm_frac_arg = (norm_frac).unsqueeze(1).unsqueeze(2).repeat(1, 1, inputs.size(2), inputs.size(3))
                #norm_frac = (calc_norm(inputs, gray_scale) / new_norm)
                # norm_frac_arg = (norm_frac).unsqueeze(1).unsqueeze(2).repeat(1, 1, inputs.size(2), inputs.size(3))
                # noisy_inputs_arg = noisy_inputs
                # n = noisy_inputs_arg.shape[0]
                # for i in range(n):
                #     noisy_inputs_arg.data[i] = noisy_inputs_arg.data[i] * norm_frac[i]
                # outputs = norm_frac_arg*model(noisy_inputs_arg / (2 * (sigma ** 2)))
                outputs = norm_frac * model((norm_frac*noisy_inputs)/ (2 * (sigma ** 2)))
                res_norm.append(calc_norm(noisy_inputs-outputs, gray_scale).cpu().detach().numpy())
                noise_norm.append(calc_norm(noise, gray_scale).cpu().detach().numpy())
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
                psnr += utils.compute_psnr(outputs, inputs)
            mse_lst.append(test_loss / (batch_idx + 1))
            res_norm = np.asarray(res_norm)
            noise_norm = np.asarray(noise_norm)
            res_norm_lst.append(np.mean(res_norm))
            noise_norm_lst.append(np.mean(noise_norm))
            psnr_lst.append(psnr/ (batch_idx + 1))
    res_norm_arr = np.asarray(res_norm_lst)
    noise_norm_arr = np.asarray(noise_norm_lst)
    mse_arr = np.asarray(mse_lst)
    psnr_arr = np.asarray(psnr_lst)
    np.save(save_path + '/res_norm_arr.npy',  res_norm_arr)
    np.save(save_path + '/noise_norm_arr.npy', noise_norm_arr)
    np.save(save_path + '/mse_arr.npy', mse_arr)
    np.save(save_path + '/psnr_arr.npy', psnr_arr)
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
    plt.figure()
    plt.plot(sigma_arr, psnr_arr, label='Test PSNR')
    plt.legend()
    plt.ylabel('PSNR')
    plt.xlabel('Noise std')
    plt.savefig(save_path + "/psnr.png", bbox_inches='tight')


def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of spherical image denoiser')
    parser.add_argument('--dataset', default="ds_mnist", type=str,
                        help='The name of the dataset to train. [Default: ds_mnist]')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--use-output-activation', action='store_true', default=False,
                        help='use output activation')
    parser.add_argument('--layers', '-L', default=8, type=int,
                        help='number of layers')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--results-dir', type=str, default="TMP",
                        help='Results dir name')
    parser.add_argument('--sigma', default=0.5, type=float, help='noise std')
    parser.add_argument('--sigma_min_analysis', default=0.01, type=float, help='analysis noise std min')
    parser.add_argument('--sigma_max_analysis', default=1, type=float, help='analysis noise std max')
    parser.add_argument('--sigma_min_train', default=0.1, type=float, help='training noise std min')
    parser.add_argument('--sigma_max_train', default=1, type=float, help='training noise std max')
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
    if args.dataset == 'ds_mnist':
        dataset = 'mnist'
        gray_scale = True
        new_norm = 27.66902  # mean of the norm of mnist images
        input_size = 28
        input_channels = 1
    else:
        if args.dataset == 'ds_fashionmnist':
            dataset = 'fashionmnist'
            gray_scale = True
            new_norm = 27.392767  # mean of the norm of fashionmnist images
            input_size = 28
            input_channels = 1
        else:
            if args.dataset == 'ds_bsd':
                dataset = 'bsd'
                gray_scale = False
                new_norm = 45.167343  # mean of the norm of bsd images with patches of 80*80
                input_channels = 3
                input_size = 481*321

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'ds_mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist/', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        val_loader = None
    else:
        if args.dataset == 'ds_fashionmnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('data/fmnist/', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.2859,), (0.3530,))
                                      ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('data/fmnist/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2859,), (0.3530,))
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
            val_loader = None
        else:
            if args.dataset == 'ds_bsd':
                dataset_train = DatasetBSD(
                    root=os.path.join('/home/chen/Simulations/spherical_image_denoiser/BSD500', 'train'),
                    training=True,
                    normilized_image=False, new_norm=new_norm, crop_size=80, gray_scale=False, transpose=True)
                dataset_test = DatasetBSD(
                    root=os.path.join('/home/chen/Simulations/spherical_image_denoiser/BSD500', 'val'),
                    training=True,
                    normilized_image=False, new_norm=new_norm, crop_size=80, gray_scale=False, transpose=True)
                dataset_val = DatasetBSD68(
                    root='/home/chen/Simulations/spherical_image_denoiser/BSD68',
                    normilized_image=False, new_norm=new_norm, gray_scale=False, transpose=True)
                # loaders
                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                           **kwargs)
                test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True,
                                                          **kwargs)
                val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True,
                                                         **kwargs)

    print('==> Loading model..')
    if args.dataset == 'ds_bsd':
        model = DnCNN(in_nc=input_channels, out_nc=input_channels, new_norm=new_norm, use_output_activation=args.use_output_activation).to(device)
    else:
        model = dbl_FC(input_size, args.layers, new_norm, args.use_output_activation).to(device)

    model.load_state_dict(torch.load('/home/chen/Simulations/spherical_image_denoiser/logs/saved_models/mnist_denoiser.pt'))

    criterion = torch.nn.MSELoss(reduction='mean')

    save_analysis_plot_denoiser_residual(model, args.sigma_min_analysis, args.sigma_max_analysis, save_path, criterion, test_loader, new_norm,  80**2, input_channels, gray_scale, device)

    # for sigma_for_generation in np.logspace(1, -2, 10):
    #     generate_denoiser_images(val_loader, [model], sigma=sigma_for_generation, device=device,
    #                              path=save_path, labels=[dataset + "_denoiser"], img_idxes=None, gray_scale=gray_scale,
    #                              baseline=False)
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))