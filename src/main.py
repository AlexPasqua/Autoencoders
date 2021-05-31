import torch
import argparse
import time
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_mnist import FastMNIST
from autoencoders import ShallowAutoencoder, DeepAutoencoder, ShallowConvAutoencoder, DeepConvAutoencoder
from custom_losses import ContrastiveLoss
from training_utilities import get_clean_sets, get_noisy_sets


def tsne(model, n_components=2, noisy=False, save=False, path="../plots/tsne.png", **kwargs):
    """
    Compute t-SNE
    :param model: the model used to encode the data
    :param n_components: dimensionality of the points in the plot (2D / 3D)
    :param noisy: if True, encode noisy data
    :param save: if True, save the plot
    :param path: if 'save', path where to save the plot
    """
    _, mnist_test = get_noisy_sets(**kwargs) if noisy else get_clean_sets()
    with torch.no_grad():
        model.cpu()
        encoded = torch.flatten(model.encoder(mnist_test.data.cpu()), 1)
        embedded = TSNE(n_components=n_components, verbose=1).fit_transform(encoded)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedded[:, 0], y=embedded[:, 1],
            hue=mnist_test.labels.cpu(),
            palette=sns.color_palette("hls", 10),
            legend="full",
            # alpha=0.3
        )
        if save:
            plt.savefig(path)
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of a basic AE")
    parser.add_argument('--mode', action='store', type=str, help="Modality {basic | contractive}")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', action='store', type=int, default=1, help='Number of epochs of training')
    parser.add_argument('--bs', action='store', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', action='store', type=float, default=0., help='Momentum coefficient')
    parser.add_argument('--save', '-s', action='store_true', help="Save or not the AE")
    parser.add_argument('--save_path', action='store', type=str, help="Path to save the AE")
    parser.add_argument('--load', '-ld', action='store_true', help="Load or not the ae from file")
    parser.add_argument('--model_path', action='store', type=str, help="Path to the AE's file")
    args = parser.parse_args()

    # load the ae if requested, otherwise create one
    noise_const = 0.1
    if args.load:
        ae = torch.load(args.model_path)
    else:
        ae = DeepConvAutoencoder(dims=(5, 10, 20, 50), kernel_sizes=3)
        # ae = ShallowConvAutoencoder(channels=1, n_filters=10, kernel_size=3)
        # ae = DeepAutoencoder(dims=(784, 500, 250, 100, 50, 20))

        mode = 'denoising'
        start = time.time()
        # ae.pretrain_layers(mode=mode, num_epochs=1, bs=512, lr=0.5, momentum=0.7, noise_const=noise_const, patch_width=0)
        # summary(ae.cpu(), input_size=(1, 28, 28), device='cpu')

        # loss = evaluate(model=ae, mode=mode, criterion=nn.MSELoss())
        # print(f"Loss before fine tuning: {loss}\n\nFine tuning:")
        ae.fit(mode=mode, num_epochs=10, bs=10000, lr=0.5, momentum=0.7, noise_const=noise_const, patch_width=0)
        # loss = evaluate(model=ae, mode=mode, criterion=nn.MSELoss())
        # print(f"Loss after fine tuning: {loss}")
        print(f"Total training and evaluation time: {round(time.time() - start, 3)}s")
        torch.save(ae, "../models/deep_ae_500-200-100-50")

    # ae = DeepConvAutoencoder(inp_area=28, dims=(5, 10, 20), kernel_sizes=3)
    # ae = ShallowConvAutoencoder(channels=1, n_filters=20, kernel_size=3)
    # summary(ae.to(device), (1, 28, 28), batch_size=5000)
    # ae.pretrain_layers(mode='denoising', patch_width=0, num_epochs=1, bs=5000, lr=0.5, momentum=0.7)
    # fit(model=ae, mode='denoising', patch_width=0, num_epochs=10, bs=64, lr=0.5, momentum=0.7)
    # ae.tr(mode='denoising', patch_width=0, num_epochs=10, bs=64, lr=0.5, momentum=0.7, )

    # t-SNE
    tsne(model=ae)

    # print the first reconstructions
    ae = ae.to('cpu')
    _, ts_data = get_noisy_sets(patch_width=0)
    # _, ts_data = get_clean_sets()
    ts_data = ts_data.data.cpu()
    test_loader = torch.utils.data.DataLoader(ts_data)
    for i, img in enumerate(test_loader):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.reshape(img, (28, 28)))
        ax[1].imshow(torch.reshape(ae(img).data, (28, 28)))
        plt.show()
        if i >= 4:
            break
