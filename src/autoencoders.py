import copy
import math
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from custom_mnist import FastMNIST, NoisyMNIST
from custom_losses import ContrastiveLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_train = FastMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = FastMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())


class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, latent_dim), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder(nn.Module):
    def __init__(self, dims):
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
        tr_data = mnist_train.data
        val_data = mnist_test.data
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                print(f"Pretrain layer: {layer}")
                shallow_ae = ShallowAutoencoder(layer.in_features, layer.out_features)
                fit(model=shallow_ae, mode=mode, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs,
                    lr=lr, momentum=momentum, **kwargs)
                self.encoder[i].weight.data = copy.deepcopy(shallow_ae.encoder[1].weight.data)
                self.encoder[i].bias.data = copy.deepcopy(shallow_ae.encoder[1].bias.data)
                self.decoder[len(self.decoder) - i - 1].weight.data = copy.deepcopy(shallow_ae.decoder[0].weight.data)
                self.decoder[len(self.decoder) - i - 1].bias.data = copy.deepcopy(shallow_ae.decoder[0].bias.data)
                if i == 1 and mode == 'denoising':  # i = 1 --> fist Linear layer
                    tr_data, val_data = get_noisy_data(**kwargs)
                    mode = 'basic'  # for the pretraining of the deeper layers
                tr_data, val_data = self.create_next_layer_sets(shallow_ae=shallow_ae, prev_tr_data=tr_data, prev_val_data=val_data)

    @staticmethod
    def create_next_layer_sets(shallow_ae, prev_tr_data, prev_val_data):
        with torch.no_grad():
            next_tr_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_tr_data)), 1)
            next_val_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_val_data)), 1)
        return next_tr_data, next_val_data


def get_noisy_data(**kwargs):
    noisy_tr_data = NoisyMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor(), **kwargs)
    noisy_val_data = NoisyMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor(), **kwargs)
    return noisy_tr_data.data, noisy_val_data.data


def fit(model, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1

    # load the dataset
    tr_data = tr_data.to(device) if tr_data is not None else mnist_train.data.to(device)
    val_data = val_data.to(device) if val_data is not None else mnist_test.data.to(device)
    noisy_tr_data = None    # just to avoid reference before assignment

    # set the device: GPU if cuda is available, else CPU
    model.to(device)

    # set optimizer and loss type (depending on the type of AE)
    mod_values = (None, 'basic', 'contractive', 'denoising')
    if mode not in mod_values:
        raise ValueError(f"value for mod must be in {mod_values}, got {mode}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    if mode == 'contractive':
        criterion = ContrastiveLoss(ae=model, lambd=1e-4)
    elif mode == 'denoising':
        noisy_tr_data, val_data = get_noisy_data(**kwargs)

    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0
        n_batches = math.ceil(len(tr_data) / bs)
        # shuffle
        indexes = torch.randperm(tr_data.shape[0])
        tr_data = tr_data[indexes]
        if mode == 'denoising':
            noisy_tr_data = noisy_tr_data[indexes]
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # zero the gradient
            optimizer.zero_grad()
            # select a (mini)batch from the training set and compute net's outputs
            train_batch = tr_data[batch_idx * bs: batch_idx * bs + bs]
            if mode == 'denoising':
                noisy_batch = noisy_tr_data[batch_idx * bs: batch_idx * bs + bs]
                outputs = model(noisy_batch)
            else:
                outputs = model(train_batch)
            # compute loss
            loss = criterion(outputs, torch.flatten(train_batch, start_dim=1))
            tr_loss += loss.item()
            # propagate back the loss
            loss.backward()
            optimizer.step()
            # update progress bar
            progbar.update()
            progbar.set_postfix(train_loss=f"{loss.item():.4f}")
        last_batch_loss = loss.item()
        tr_loss /= n_batches
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate(model=model, data=val_data, criterion=criterion)
        history['val_loss'].append(round(val_loss, 5))
        progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{val_loss:.4f}")
        progbar.close()

        # simple early stopping mechanism
        if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (
                    last_values[-3] < last_values[-2] < last_values[-1]):
                return history

    return history


def evaluate(model, data, criterion):
    model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, torch.flatten(data, start_dim=1))
    return loss.item()
