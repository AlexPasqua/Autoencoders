"""
Taken from GitHub Gist
Author: Joost van Amersfoort (y0ast)
link: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
"""
import copy
import random
import torch
from torchvision.datasets import MNIST

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


class NoisyMNIST(FastMNIST):
    """ subclass of FastMNIST with data=noisy_data and targets=clean_data (instead of labels) """
    def __init__(self, noise_const=0.1, patch_width=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.targets = torch.flatten(copy.deepcopy(self.data), 1)
        self.data = self.data + noise_const * torch.randn(self.data.shape).to(device)
        if patch_width > 0:
            for img in self.data:
                start = random.randint(0, img.shape[-1] - patch_width - 1)
                img[:, start: start + patch_width, start: start + patch_width] = 0
