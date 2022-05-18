import torch
import numpy as np
import random
import math
from PIL import Image


def tensor_to_image(x):
    image = x.squeeze(dim=0).mul_(255).add_(0.5).clamp_(0, 255).squeeze().to('cpu', torch.uint8).numpy()
    return image


def compute_psnr(x, y):
    x = np.array(tensor_to_image(x)).astype(np.float64)
    y = np.array(tensor_to_image(y)).astype(np.float64)

    mse = np.mean((x - y) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse))


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True