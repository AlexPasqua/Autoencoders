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


class BaseAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.reshape(decoded, (x.shape[0], x.shape[1], 28, 28))


def fit_ae(model, num_epochs, bs, lr, momentum):
    # load the dataset
    mnist_train = FastMNIST(root='../MNIST/', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = FastMNIST(root='../MNIST/', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(mnist_train, [50000, 10000])
    # train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=len(mnist_train), shuffle=True, num_workers=0)
    # val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=bs)
    # test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=len(mnist_test), num_workers=0)

    # set the device: GPU if cuda is available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # training cycle
    for epoch in range(num_epochs):
        # training
        model.train()
        n_batches = math.ceil(len(mnist_train.data) / bs)
        progbar = tqdm(range(n_batches), total=n_batches)
        progbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx in range(n_batches):
            train_batch = mnist_train.data[batch_idx * bs: batch_idx * bs + bs]
            # move data to GPU if possible -> commented because whole dataset already in GPU
            # train_batch = train_batch.to(device)
            # zero the gradient
            optimizer.zero_grad()
            # compute net's output
            outputs = model(train_batch)
            # compute loss
            loss = criterion(outputs, train_batch)
            # propagate back the loss
            loss.backward()
            optimizer.step()
            # update progress bar
            progbar.update()
            progbar.set_postfix(train_loss=f"{loss.item():.4f}")
        last_batch_loss = loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            # move data to GPU if possible -> commented because whole dataset already in GPU
            # val_batch = val_batch.to(device)
            # compute net's output
            outputs = model(mnist_test.data)
            # compute loss
            loss = criterion(outputs, mnist_test.data)
        # update progress bar
        progbar.set_postfix(train_loss=f"{last_batch_loss:.4f}", val_loss=f"{loss.item():.4f}")
        progbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of a basic ae")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', action='store', type=int, default=1, help='Number of epochs of training')
    parser.add_argument('--bs', action='store', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', action='store', type=float, default=0., help='Momentum coefficient')
    parser.add_argument('--save', '-s', action='store_true', help="Save or not the model")
    parser.add_argument('--save_path', action='store', type=str, help="Path to save the model")
    parser.add_argument('--load', '-ldw', action='store_true', help="Load or not the model from file")
    parser.add_argument('--model_path', action='store', type=str, help="Path to the model's file")
    args = parser.parse_args()

    # load the model if requested, otherwise create one
    ae = torch.load(args.model_path) if args.load else BaseAE(28 * 28, 200)

    # train the model
    if args.train:
        start = time.time()
        fit_ae(model=ae, num_epochs=args.epochs, bs=args.bs, lr=args.lr, momentum=args.momentum)
        print(f"Execution time: {time.time() - start}")

    # save model
    if args.save:
        torch.save(ae, args.save_path)

    # # print the first reconstructions
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
