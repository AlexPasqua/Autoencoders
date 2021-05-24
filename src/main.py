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
from autoencoders import ShallowAutoencoder, mnist_train, mnist_test, device, DeepAutoencoder, evaluate, fit
from custom_losses import ContrastiveLoss


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
    # ae = torch.load(args.model_path) if args.load else DeepAutoencoder((28 * 28, 625, 200, 100))

    ae = DeepAutoencoder((784, 500, 200, 100, 50))
    ae.pretrain_layers(num_epochs=10, bs=32, lr=0.3, momentum=0.7)
    loss = evaluate(model=ae, data=mnist_test.data, criterion=nn.MSELoss())
    print(f"Loss before fine tuning: {loss}\n\nFine tuning:")
    fit(model=ae, num_epochs=10, bs=32, lr=0.2, momentum=0.7)
    loss = evaluate(model=ae, data=mnist_test.data, criterion=nn.MSELoss())
    print(f"Loss after fine tuning: {loss}")

    # aes = []
    # dims = (784, 500, 200, 100, 50)
    # for i in range(len(dims) - 1):
    #     aes.append(ShallowAutoencoder(dims[i], dims[i+1]))
    #
    # tr_data = mnist_train.data
    # val_data = mnist_test.data
    # for i in range(len(aes)):
    #     hist = fit(model=aes[i], mode='basic', tr_data=tr_data, val_data=val_data, num_epochs=1, bs=32, lr=0.5, momentum=0.7)
    #     with torch.no_grad():
    #         tr_data = torch.unsqueeze(torch.sigmoid(aes[i].encoder(tr_data)), 1).to(device)
    #         val_data = torch.unsqueeze(torch.sigmoid(aes[i].encoder(val_data)), 1).to(device)

    # ae = ShallowAutoencoder(28*28, 100)
    # ae2 = ShallowAutoencoder(28*28, 50)
    # hist2 = fit(model=ae2, mode='basic', num_epochs=2, bs=32, lr=0.5, momentum=0.7)

    # print the first reconstructions
    ae = ae.to('cpu')
    ts_data = mnist_test.data.to('cpu')
    test_loader = torch.utils.data.DataLoader(ts_data)
    for i, img in enumerate(test_loader):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.reshape(img, (28, 28)))
        ax[1].imshow(torch.reshape(ae(img).data, (28, 28)))
        plt.show()
        if i >= 4:
            break
