import torch.optim
import torch.utils.data
import os
from ed import *
import matplotlib.pyplot as plt


def generate_denoiser_images(dl, models, sigma, device, path=None, labels=[], img_idxes=None):
    if img_idxes is None:
        img_idxes = torch.randint(0, 100, [10])

    if not isinstance(models, list):
        models = [models]

    fig, axs = plt.subplots(2 + len(models), len(img_idxes))
    fig.set_size_inches(24, 12)

    for i, img_idx in enumerate(img_idxes):
        img = dl.dataset[img_idx][0].unsqueeze(0).to(device=device)

        noise = torch.randn(img.shape, device=device) * sigma

        outs = []
        for model in models:
            with torch.no_grad():
                outs.append(model(img + noise).squeeze().cpu())

        noise = noise.squeeze().cpu()
        img = img.squeeze().cpu()

        #         if len(out.shape) == 1:
        #             out = out.view(img.shape)

        vmin = -1
        vmax = 2
        axs[0, i].imshow(img, vmin=vmin, vmax=vmax, cmap='Greys')

        axs[1, i].imshow(img + noise, vmin=vmin, vmax=vmax, cmap='Greys')

        for k in range(len(models)):
            axs[k + 2, i].imshow(outs[k], vmin=vmin, vmax=vmax, cmap='Greys')

        #         axs[3,i].imshow(noise, vmin=vmin, vmax=vmax, cmap='Greys')
        #         axs[4,i].imshow(img-out, vmin=vmin, vmax=vmax, cmap='Greys')

        for j in range(2 + len(models)):
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    axs[0, 0].set_ylabel("source", fontsize=20)
    axs[1, 0].set_ylabel("noisy", fontsize=20)

    if path is None:
        path = "images/Denoiser"
        for k, label in enumerate(labels):
            axs[k + 2, 0].set_ylabel(labels[k], fontsize=20)
            name = name + "_" + labels[k]
        os.makedirs(path)

    name = "Noise%.03f" % sigma
    plt.savefig(f"{path}/{name}.png", format='png')
