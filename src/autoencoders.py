import copy
import json
import math
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms
from typing import Sequence, Union, Tuple
from training_utilities import get_clean_sets, get_noisy_sets, fit_ae
from custom_mnist import FastMNIST, NoisyMNIST
from custom_losses import ContractiveLoss

# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AbstractAutoencoder(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, mode='basic', tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        return fit_ae(model=self, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr,
                      momentum=momentum, **kwargs)

    def manifold(self, load=None, path=None, max_iters=1000, thresh=0.02, side_len=28):
        if load:
            images_progression = np.load(path)
        else:
            self.cpu()
            noise_img = torch.randn((1, 1, side_len, side_len))
            noise_img -= torch.min(noise_img)
            noise_img /= torch.max(noise_img)
            images_progression = [torch.squeeze(noise_img)]
            serializable_progression = [torch.squeeze(noise_img).cpu().numpy()]

            # iterate
            i = 0
            loss = 1000
            input = noise_img
            prev_output = None
            with torch.no_grad():
                while loss > thresh and i < max_iters:
                    output = self(input)
                    img = torch.reshape(torch.squeeze(output), shape=(side_len, side_len))
                    rescaled_img = (img - torch.min(img)) / torch.max(img)
                    images_progression.append(rescaled_img)
                    serializable_progression.append(rescaled_img.cpu().numpy())
                    if prev_output is not None:
                        # noinspection PyTypeChecker
                        loss = F.mse_loss(output, prev_output)
                    prev_output = output
                    input = output
                    i += 1

            # save sequence of images
            serializable_progression = np.array(serializable_progression)
            np.save(file="manifold_img_seq_2", arr=serializable_progression)

        images_progression = images_progression[:60]
        import matplotlib.cm as cm
        import matplotlib.animation as animation
        frames = []  # for storing the generated images
        fig = plt.figure()
        for i in range(len(images_progression)):
            frames.append([plt.imshow(images_progression[i], animated=True)])
        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        ani.save('movie2.gif')
        plt.show()
        exit()

        # show images progression
        img = None
        for i in range(len(images_progression)):
            if img is None:
                img = plt.imshow(images_progression[0])
            else:
                img.set_data(images_progression[i])
            plt.pause(.1)
            plt.draw()


class ShallowAutoencoder(AbstractAutoencoder):
    def __init__(self, input_dim: int = 784, latent_dim: int = 200):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.type = "shallowAE"
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, latent_dim), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())


class DeepAutoencoder(AbstractAutoencoder):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "deepAE"
        enc_layers = []
        dec_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            # enc_layers.append(nn.ReLU(inplace=True))
            enc_layers.append(nn.ReLU(inplace=True))
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1]))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()
        self.encoder = nn.Sequential(nn.Flatten(), *enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

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


class DeepRandomizedAutoencoder(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "deepRandAE"
        self.shallow_encs_params = nn.ParameterList([
            nn.Parameter(nn.init.xavier_uniform_(torch.empty(dims[i], dims[i + 1])), requires_grad=False)
            for i in range(len(dims) - 1)
        ]).to(device)
        self.shallow_decs_params = nn.ParameterList([
            nn.Parameter(nn.init.xavier_uniform_(torch.empty(dims[i], dims[i - 1])), requires_grad=True)
            for i in reversed(range(1, len(dims)))
        ]).to(device)
        self.enc_params = nn.ParameterList([])

    def fit(self, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1

        tr_set, val_set = get_clean_sets()
        tr_data, tr_targets = tr_set.data, tr_set.targets
        val_data, val_targets = val_set.data, val_set.targets
        del tr_set, val_set

        # train the shallow AEs to have trained weight matrices
        for i in range(len(self.shallow_encs_params)):
            enc_w = nn.Parameter(self.shallow_encs_params[i], requires_grad=False)
            dec_w = nn.Parameter(self.shallow_decs_params[len(self.shallow_decs_params) - 1 - i], requires_grad=True)
            optimizer = torch.optim.SGD([dec_w], lr=lr, momentum=momentum)

            # training cycle
            loss = None  # just to avoid reference before assigment
            n_batches = math.ceil(len(tr_data) / bs)
            history = {'tr_loss': [], 'val_loss': []}
            for epoch in range(num_epochs):
                tr_loss = 0
                # shuffle
                indexes = torch.randperm(tr_data.shape[0])
                tr_data = tr_data[indexes]
                tr_targets = tr_targets[indexes]
                progbar = tqdm(range(n_batches), total=n_batches, disable=False)
                progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                # iterate through batches
                for batch_idx in range(n_batches):
                    optimizer.zero_grad()
                    train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to(device)
                    train_targets_batch = tr_targets[batch_idx * bs: batch_idx * bs + bs].to(device)
                    # forward pass
                    encoded = F.relu(torch.flatten(train_data_batch, start_dim=1) @ enc_w)
                    outputs = torch.sigmoid(encoded @ dec_w)
                    # loss and backward pass
                    loss = F.mse_loss(torch.flatten(outputs, 1), train_targets_batch)
                    tr_loss += loss.item()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(dec_w, max_norm=1.0)
                    optimizer.step()
                    progbar.update()
                    progbar.set_postfix(train_loss=f"{loss.item():.4f}")
                last_batch_loss = loss.item()
                tr_loss /= n_batches
                history['tr_loss'].append(tr_loss)
                # validation
                with torch.no_grad():
                    outputs = torch.sigmoid(F.relu(torch.flatten(val_data, start_dim=1) @ enc_w) @ dec_w)
                    loss = F.mse_loss(torch.flatten(outputs, 1), val_targets)
                    val_loss = loss.item()
                history['val_loss'].append(val_loss)
                progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{val_loss:.4f}")
                progbar.close()

                # simple early stopping mechanism
                last = history['val_loss'][-10:]
                if epoch >= 20 and last[-3] < last[-2] < last[-1]:
                    break

            # save the trained weights
            self.shallow_decs_params[len(self.shallow_decs_params) - 1 - i] = dec_w  # should be unnecessary
            self.enc_params.append(nn.Parameter(dec_w.T))
            torch.cuda.empty_cache()

            # create datasets for next layer
            with torch.no_grad():
                new_tr_data = torch.empty(tr_data.shape[0], enc_w.shape[1])
                # use minibatches on training data for memory reasons, not so in validation data
                for batch_idx in range(n_batches):
                    train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to(device)
                    train_data_batch = torch.sigmoid(F.relu(torch.flatten(train_data_batch, start_dim=1) @ enc_w))
                    new_tr_data[batch_idx * bs: batch_idx * bs + bs] = train_data_batch
                tr_data = new_tr_data
                val_data = torch.sigmoid(F.relu(torch.flatten(val_data, start_dim=1) @ enc_w))
                tr_targets = tr_data
                val_targets = val_data

            # intermediate layers need less epochs -> they don't improve with more
            num_epochs = num_epochs // 2

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[-1] ** 2)
        for w in self.enc_params:
            try:
                x = F.relu(x @ w)
            except RuntimeError:
                pass
        for w in reversed(self.enc_params):
            try:
                x = F.relu(x @ w.T)
            except RuntimeError:
                pass
        return torch.sigmoid(x)

    def encoder(self, x):
        x = x.view(x.shape[0], x.shape[-1] ** 2)
        for w in self.enc_params:
            try:
                x = F.relu(x @ w)
            except RuntimeError:
                pass
        return torch.sigmoid(x)


# noinspection PyTypeChecker
class ShallowConvAutoencoder(AbstractAutoencoder):
    def __init__(self, channels=1, n_filters=10, kernel_size: int = 3, central_dim=100,
                 inp_side_len: Union[int, Tuple[int, int]] = 28):
        super().__init__()
        self.type = "shallowConvAE"
        pad = (kernel_size - 1) // 2  # pad to keep the original area after convolution
        central_side_len = math.floor(inp_side_len / 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=central_side_len ** 2 * n_filters, out_features=central_dim),
            nn.ReLU(inplace=True))

        # set kernel size, padding and stride to get the correct output shape
        kersize = 2 if central_side_len * 2 == inp_side_len else 3
        self.decoder = nn.Sequential(
            nn.Linear(in_features=central_dim, out_features=central_side_len ** 2 * n_filters),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(n_filters, central_side_len, central_side_len)),
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=channels, kernel_size=kersize, stride=2, padding=0),
            nn.Sigmoid())


# noinspection PyTypeChecker
class DeepConvAutoencoder(AbstractAutoencoder):
    def __init__(self, inp_side_len=28, dims: Sequence[int] = (5, 10),
                 kernel_sizes: int = 3, central_dim=100, pool=True):
        super().__init__()
        self.type = "deepConvAE"

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        side_len = inp_side_len
        side_lengths = [side_len]
        dims = (1, *dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if pool and (i % step_pool == 0 or i == len(dims) - 1) and side_len > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                side_len = math.floor(side_len / 2)
                side_lengths.append(side_len)

        # fully connected layers in the center of the autoencoder to reduce dimensionality
        fc_dims = (side_len ** 2 * dims[-1], side_len ** 2 * dims[-1] // 2, central_dim)
        self.encoder = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(inplace=True)
        )

        # self.enc_linear = nn.Sequential(nn.Flatten(), nn.Linear(fc_dims[0], fc_dims[1]), nn.Linear(fc_dims[1], fc_dims[2]))
        # self.dec_linear = nn.Sequential(nn.Linear(fc_dims[2], fc_dims[1]), nn.Linear(fc_dims[1], fc_dims[0]))

        # build decoder
        central_side_len = side_lengths.pop(-1)
        # side_lengths = side_lengths[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(side_lengths) > 0 and side_len * 2 == side_lengths.pop(-1) else 3
            pad, stride = (1, 1) if side_len == inp_side_len else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(fc_dims[2], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[-1], central_side_len, central_side_len)),
            *dec_layers,
        )

    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     print(decoded.shape)
    #     exit()

    # encoded = self.encoder(x)
    # enc_fc = self.enc_linear(encoded)
    # dec_fc = self.dec_linear(enc_fc)
    # depth, side_len = encoded.shape[1], encoded.shape[-1]
    # dec_conv_input = dec_fc.view(dec_fc.shape[0], depth, side_len, side_len)
    # decoded = self.decoder(dec_conv_input)
    # return decoded
