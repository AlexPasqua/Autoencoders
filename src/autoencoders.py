import copy
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from fast_mnist import FastMNIST
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

    def pretrain_layers(self, num_epochs, bs, lr, momentum):
        tr_data = mnist_train.data
        val_data = mnist_test.data
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                print(f"Pretrain layer: {layer}")
                shallow_ae = ShallowAutoencoder(layer.in_features, layer.out_features)
                fit(model=shallow_ae, tr_data=tr_data, val_data=val_data, num_epochs=num_epochs, bs=bs, lr=lr, momentum=momentum)
                self.encoder[i].weight.data = copy.deepcopy(shallow_ae.encoder[1].weight.data)
                self.encoder[i].bias.data = copy.deepcopy(shallow_ae.encoder[1].bias.data)
                self.decoder[len(self.decoder) - i - 1].weight.data = copy.deepcopy(shallow_ae.decoder[0].weight.data)
                self.decoder[len(self.decoder) - i - 1].bias.data = copy.deepcopy(shallow_ae.decoder[0].bias.data)
                tr_data, val_data = self.create_next_layer_sets(shallow_ae=shallow_ae, prev_tr_data=tr_data, prev_val_data=val_data)

    @staticmethod
    def create_next_layer_sets(shallow_ae, prev_tr_data, prev_val_data):
        if prev_tr_data is None:
            prev_tr_data = mnist_train.data
            prev_val_data = mnist_test.data
        with torch.no_grad():
            next_tr_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_tr_data)), 1)
            next_val_data = torch.unsqueeze(torch.sigmoid(shallow_ae.encoder(prev_val_data)), 1)
        return next_tr_data, next_val_data


def fit(model, mode=None, tr_data=None, val_data=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1

    # load the dataset
    tr_data = tr_data.to(device) if tr_data is not None else mnist_train.data.to(device)
    val_data = val_data.to(device) if val_data is not None else mnist_test.data.to(device)

    # set the device: GPU if cuda is available, else CPU
    model.to(device)

    # set optimizer and loss type (depending on the type of AE)
    mod_values = (None, 'basic', 'contractive')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if mode is None or mode == 'basic':
        criterion = nn.MSELoss()
    elif mode == 'contractive':
        criterion = ContrastiveLoss(ae=model, lambd=1e-4)
    else:
        raise ValueError(f"value for mod must be in {mod_values}, got {mode}")

    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0
        n_batches = math.ceil(len(tr_data) / bs)
        tr_data = tr_data[torch.randperm(tr_data.shape[0])]  # shuffle
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # select a (mini)batch from the training set
            train_batch = tr_data[batch_idx * bs: batch_idx * bs + bs]
            # move data to GPU if possible -> commented because whole dataset already in GPU
            # train_batch = train_batch.to(device)
            # zero the gradient
            optimizer.zero_grad()
            # compute net's input
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
