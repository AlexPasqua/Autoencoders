import copy
import math
import random
import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from custom_mnist import FastMNIST, NoisyMNIST
from custom_losses import ContrastiveLoss


# set device and datasets globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# these variables will be set only if needed
mnist_train = None
mnist_test = None
noisy_mnist_train = None
noisy_mnist_test = None


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
        tr_data = None
        val_data = None
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
                    tr_set, val_set = get_noisy_sets(**kwargs)
                    tr_data, tr_targets = tr_set.data, tr_set
                    val_data, val_targets = val_set.data, val_set.targets
                    mode = 'basic'  # for the pretraining of the deeper layers
                tr_data, val_data = self.create_next_layer_sets(shallow_ae=shallow_ae, prev_tr_data=tr_data, prev_val_data=val_data)
                del shallow_ae
                torch.cuda.empty_cache()

    @staticmethod
    def create_next_layer_sets(shallow_ae, prev_tr_data=None, prev_val_data=None):
        train_set, val_set = get_clean_sets()
        prev_tr_data = train_set.data if prev_tr_data is None else prev_tr_data
        prev_val_data = val_set.data if prev_val_data is None else prev_val_data
        with torch.no_grad():
            next_tr_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_tr_data)), 1)
            next_val_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_val_data)), 1)
        return next_tr_data, next_val_data


def get_clean_sets():
    global mnist_train
    global mnist_test
    if mnist_train is None:
        mnist_train = FastMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor())
        mnist_test = FastMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())
    return mnist_train, mnist_test


def get_noisy_sets(**kwargs):
    global noisy_mnist_train
    global noisy_mnist_test
    if noisy_mnist_train is None:
        noisy_mnist_train = NoisyMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor(), **kwargs)
        noisy_mnist_test = NoisyMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor(), **kwargs)
    return noisy_mnist_train, noisy_mnist_test


def fit(model, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
    model_values = (None, 'basic', 'contractive', 'denoising')
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1 and mode in model_values

    # set the device: GPU if cuda is available, else CPU
    model.to(device)

    # set optimizer, loss type and datasets (depending on the type of AE)
    tr_targets = None   # to avoid reference before assignment
    val_targets = None  # to avoid reference before assignment
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    if mode == 'contractive':
        criterion = ContrastiveLoss(ae=model, lambd=1e-4)
    elif mode == 'denoising':
        if tr_data is not None or val_data is not None:
            warnings.warn("'denoising' flag was set, so NoisyMNIST will be used for training and validation")
        noisy_train, noisy_val = get_noisy_sets(**kwargs)
        tr_data, tr_targets = noisy_train.data, noisy_train.targets
        val_data, val_targets = noisy_val.data, noisy_val.targets
        del noisy_train, noisy_val
    else:
        tr_set, val_set = get_clean_sets()
        if tr_data is None:
            tr_data, tr_targets = tr_set.data, tr_set.targets
        else:
            tr_data = tr_data.to(device)
            tr_targets = torch.flatten(copy.deepcopy(tr_data), start_dim=1)
        if val_data is None:
            val_data, val_targets = val_set.data, val_set.targets
        else:
            val_data = val_data.to(device)
            val_targets = torch.flatten(copy.deepcopy(val_data), start_dim=1)
        del tr_set, val_set
    torch.cuda.empty_cache()

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
        tr_targets = tr_targets[indexes]
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # zero the gradient
            optimizer.zero_grad()
            # select a (mini)batch from the training set and compute net's outputs
            train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs]
            train_targets_batch = tr_targets[batch_idx * bs: batch_idx * bs + bs]
            outputs = model(train_data_batch)
            # compute loss
            # loss = criterion(outputs, torch.flatten(train_targets_batch, start_dim=1))
            loss = criterion(outputs, train_targets_batch)
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
        val_loss = evaluate(model=model, data=val_data, targets=val_targets, criterion=criterion)
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


def evaluate(model, criterion, mode='basic', data=None, targets=None, **kwargs):
    # set the data
    if data is None:
        if mode == 'basic':
            # use standard MNIST test set
            _, val_set = get_clean_sets()
            data, targets = val_set.data, val_set.targets
        elif mode == 'denoising':
            _, noisy_val = get_noisy_sets(**kwargs)
            data, targets = noisy_val.data, noisy_val.targets
        else:
            raise RuntimeWarning(f"mode not valid, got {mode}")
    elif targets is None:
        targets = copy.deepcopy(data)

    # evaluate
    model.to(device)
    data = data.to(device)
    targets = targets.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, targets)
    return loss.item()
