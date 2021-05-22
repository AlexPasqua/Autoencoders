import torch
import argparse
import time
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from fast_mnist import FastMNIST
from custom_losses import *


class Autoencoder(nn.Module):
    def __init__(self, dims):
        """ :param dims: (iterable) dimensions of the layers of the encoder (the decoder's ones will be specular) """
        super().__init__()
        assert all(d > 0 for d in dims) and len(dims) > 0
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
        return torch.reshape(decoded, (x.shape[0], x.shape[1], 28, 28))


def fit(model, mode=None, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
    """
    Perform training of the autoencoder
    :param model: the autoencoder to train
    :param mode: (str) to choose the type of AE (e.g. basic, contractive, denoising..)
    :param num_epochs: (int) number of epochs for training
    :param bs: (int) batch size
    :param lr: (float) learning rate
    :param momentum: (float) momentum coefficient
    :raises: ValueError
    :returns: history -> tr and val loss for each epoch
    """
    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1

    # load the dataset
    mnist_train = FastMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = FastMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(mnist_train, [50000, 10000])

    # set the device: GPU if cuda is available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set optimizer and loss type (depending on the type of AE)
    mod_values = (None, 'basic', 'contractive')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if mode is None or mode == 'basic':
        criterion = nn.MSELoss()
    elif mode == 'contractive':
        criterion = ContrastiveLoss(ae=model, lambd=0.1)
    else:
        raise ValueError(f"value for mod must be in {mod_values}, got {mode}")

    # training cycle
    loss = None     # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0
        n_batches = math.ceil(len(mnist_train.data) / bs)
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            # select a (mini)batch from the training set
            train_batch = mnist_train.data[batch_idx * bs: batch_idx * bs + bs]
            # move data to GPU if possible -> commented because whole dataset already in GPU
            # train_batch = train_batch.to(device)
            # zero the gradient
            optimizer.zero_grad()
            # compute net's input
            outputs = model(train_batch)
            # compute loss
            loss = criterion(outputs, train_batch)
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
        val_loss = evaluate(model=model, data=mnist_test.data, criterion=criterion)
        history['val_loss'].append(round(val_loss, 5))
        progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{val_loss:.4f}")
        progbar.close()

        # simple early stopping mechanism
        if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (last_values[-3] < last_values[-2] < last_values[-1]):
                return history

    return history


def evaluate(model, data, criterion):
    """
    Evaluate the ae
    :param model: the PyTorch ae to evaluate
    :param criterion: the criterion to use (loss)
    :returns: loss
    """
    model.eval()
    with torch.no_grad():
        # move data to GPU if possible -> commented because whole dataset already in GPU
        # val_batch = val_batch.to(device)
        # compute net's input
        outputs = model(data)
        # compute loss
        loss = criterion(outputs, data)
    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of a basic ae")
    parser.add_argument('--mode', action='store', type=str, help="Modality {basic | contractive}")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', action='store', type=int, default=1, help='Number of epochs of training')
    parser.add_argument('--bs', action='store', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', action='store', type=float, default=0., help='Momentum coefficient')
    parser.add_argument('--save', '-s', action='store_true', help="Save or not the ae")
    parser.add_argument('--save_path', action='store', type=str, help="Path to save the ae")
    parser.add_argument('--load', '-ldw', action='store_true', help="Load or not the ae from file")
    parser.add_argument('--model_path', action='store', type=str, help="Path to the ae's file")
    args = parser.parse_args()

    # load the ae if requested, otherwise create one
    ae = torch.load(args.model_path) if args.load else Autoencoder((28 * 28, 100))

    # train the ae
    if args.train:
        start = time.time()
        hist = fit(model=ae, mode=args.mode, num_epochs=args.epochs, bs=args.bs, lr=args.lr, momentum=args.momentum)
        print(f"Training time: {time.time() - start}")
        plt.plot(hist['tr_loss'], linestyle='dashed', label='Training loss')
        plt.plot(hist['val_loss'], linestyle='solid', label='Validation loss')
        plt.title("Training curve")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    # save ae
    if args.save:
        torch.save(ae, args.save_path)

    # print the first reconstructions
    # mnist_test = datasets.MNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())
    # test_loader = torch.utils.data.DataLoader(mnist_test)
    # ae = ae.to('cpu')
    # for i, (img, _) in enumerate(test_loader):
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(torch.reshape(img, (28, 28)))
    #     ax[1].imshow(torch.reshape(ae(img).data, (28, 28)))
    #     plt.show()
    #     if i >= 4:
    #         break
