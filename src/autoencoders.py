import copy
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms
from typing import Sequence, Union, Tuple
from training_utilities import get_clean_sets, get_noisy_sets, fit_ae
from custom_mnist import FastMNIST, NoisyMNIST
from custom_losses import ContrastiveLoss

# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, latent_dim), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)


class DeepAutoencoder(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        enc_layers = []
        dec_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU(inplace=True))
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1]))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()
        self.encoder = nn.Sequential(nn.Flatten(), *enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def pretrain_layers(self, num_epochs, bs, lr, momentum, mode='basic', **kwargs):
        tr_data = None
        val_data = None
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                print(f"Pretrain layer: {layer}")
                shallow_ae = ShallowAutoencoder(layer.in_features, layer.out_features)
                shallow_ae.fit(mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                               momentum=momentum, **kwargs)
                self.encoder[i].weight.data = copy.deepcopy(shallow_ae.encoder[1].weight.data)
                self.encoder[i].bias.data = copy.deepcopy(shallow_ae.encoder[1].bias.data)
                self.decoder[len(self.decoder) - i - 1].weight.data = copy.deepcopy(shallow_ae.decoder[0].weight.data)
                self.decoder[len(self.decoder) - i - 1].bias.data = copy.deepcopy(shallow_ae.decoder[0].bias.data)
                if i == 1 and mode == 'denoising':  # i = 1 --> fist Linear layer
                    tr_set, val_set = get_noisy_sets(**kwargs)
                    tr_data, tr_targets = tr_set.data, tr_set
                    val_data, val_targets = val_set.data, val_set.targets
                    mode = 'basic'  # for the pretraining of the deeper layers
                tr_data, val_data = self.create_next_layer_sets(shallow_ae=shallow_ae, prev_tr_data=tr_data,
                                                                prev_val_data=val_data)
                del shallow_ae

    def finetune(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)

    @staticmethod
    def create_next_layer_sets(shallow_ae, prev_tr_data=None, prev_val_data=None, unsqueeze=True):
        train_set, val_set = get_clean_sets()
        prev_tr_data = train_set.data if prev_tr_data is None else prev_tr_data
        prev_val_data = val_set.data if prev_val_data is None else prev_val_data
        with torch.no_grad():
            next_tr_data = torch.sigmoid(shallow_ae.encoder(prev_tr_data))
            next_val_data = torch.sigmoid(shallow_ae.encoder(prev_val_data))
            if unsqueeze:
                next_tr_data, next_val_data = torch.unsqueeze(next_tr_data, 1), torch.unsqueeze(next_val_data, 1)
        return next_tr_data, next_val_data


# noinspection PyTypeChecker
class ShallowConvAutoencoder(nn.Module):
    def __init__(self, channels: int = 1, n_filters: int = 10, kernel_size: int = 3, inp_area: Union[int, Tuple[int, int]] = 28):
        super().__init__()
        pad = (kernel_size - 1) // 2  # pad to keep the original area after convolution
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # set kernel size, padding and stride to get the correct output shape
        area = math.floor(inp_area / 2)
        kersize = 2 if area * 2 == inp_area else 3

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=channels, kernel_size=kersize, stride=2, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)


# noinspection PyTypeChecker
class DeepConvAutoencoder(nn.Module):
    def __init__(self, inp_area: Union[int, Tuple[int, int]] = 28, dims: Sequence[int] = (5, 10),
                 kernel_sizes: Union[int, Sequence[int]] = 3):
        super().__init__()

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        area = inp_area
        areas = [area]
        dims = (1, *dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if (i % step_pool == 0 or i == len(dims) - 1) and area > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                area = math.floor(area / 2)
                areas.append(area)
        self.encoder = nn.Sequential(*enc_layers)

        # build decoder
        areas = areas[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(areas) > 0 and area * 2 == areas.pop(-1) else 3
            pad, stride = (1, 1) if area == inp_area else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            area = area if pad == 1 else (area * 2 if kersize == 2 else area * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)
