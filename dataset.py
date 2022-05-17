import torch
import glob
import os
import random
from torchvision import transforms
from PIL import Image
import PIL


def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac


class DatasetBSD(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', training=True, crop_size=60, normilized_image=True, new_norm=1, gray_scale=False, max_size=None, transpose=True):
        self.root = root
        self.training = training
        self.crop_size = crop_size
        self.gray_scale = gray_scale
        self.max_size = max_size
        self.normilized_image = normilized_image
        self.new_norm = new_norm
        self.transpose = transpose

        self._init()

    def _init(self):
        # data paths
        targets = glob.glob(os.path.join(self.root, 'img', '*.*'))[:self.max_size]
        self.paths = {'target' : targets}

        # transforms
        lambd = lambda x: norm_change(x, self.new_norm)

        if self.normilized_image:
            t_list = [transforms.ToTensor(), transforms.Lambda(lambd)]
        else:
            t_list = [transforms.ToTensor()]
        self.image_transform = transforms.Compose(t_list)

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))

        # position
        w_size, h_size = size
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))

        # flip
        flip = random.random() > 0.5
        return {'crop_pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params):
        x, y = aug_params['crop_pos']
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        # target image
        if self.gray_scale:
            target = Image.open(self.paths['target'][index]).convert('L')
        else:
            target = Image.open(self.paths['target'][index]).convert('RGB')

        #tarnspose to the image be 481×321
        if (target.size[0] == 321) and self.transpose:
            target = target.transpose(PIL.Image.ROTATE_90)

        # transform
        if self.training:
            aug_params = self._get_augment_params(target.size)
            target = self._augment(target, aug_params)
        target = self.image_transform(target)

        return target, target

    def __len__(self):
        return len(self.paths['target'])


class DatasetBSD68(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', normilized_image=True, new_norm=1, gray_scale=False, max_size=None, transpose=True):
        self.root = root
        self.gray_scale = gray_scale
        self.max_size = max_size
        self.normilized_image = normilized_image
        self.new_norm = new_norm
        self.transpose = transpose

        self._init()

    def _init(self):
        # data paths
        targets = glob.glob(os.path.join(self.root, 'img', '*.*'))[:self.max_size]
        self.paths = {'target' : targets}

        # transforms
        lambd = lambda x: norm_change(x, self.new_norm)

        if self.normilized_image:
            t_list = [transforms.ToTensor(), transforms.Lambda(lambd)]
        else:
            t_list = [transforms.ToTensor()]
        self.image_transform = transforms.Compose(t_list)

    def __getitem__(self, index):
        # target image
        if self.gray_scale:
            target = Image.open(self.paths['target'][index]).convert('L')
        else:
            target = Image.open(self.paths['target'][index]).convert('RGB')

        #tarnspose to the image be 481×321
        if (target.size[0] == 321) and self.transpose:
            target = target.transpose(PIL.Image.ROTATE_90)

        target = self.image_transform(target)

        return target, target

    def __len__(self):
        return len(self.paths['target'])