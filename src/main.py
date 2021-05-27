import torch
import argparse
import time
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_mnist import FastMNIST
from autoencoders import ShallowAutoencoder, device, DeepAutoencoder, evaluate, fit
from custom_losses import ContrastiveLoss


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
    noise_const = 0.2
    if args.load:
        ae = torch.load(args.model_path)
    else:
        ae = DeepAutoencoder((784, 500, 200, 100, 50, 20))
        mode = 'denoising'
        start = time.time()
        ae.pretrain_layers(mode=mode, num_epochs=2, bs=32, lr=0.2, momentum=0.7, noise_const=noise_const, patch_width=10)
        loss = evaluate(model=ae, mode=mode, criterion=nn.MSELoss())
        print(f"Loss before fine tuning: {loss}\n\nFine tuning:")
        fit(model=ae, mode='denoising', num_epochs=1, bs=128, lr=0.5, momentum=0.7, noise_const=noise_const, patch_width=10)
        loss = evaluate(model=ae, mode=mode, criterion=nn.MSELoss())
        print(f"Loss after fine tuning: {loss}")
        print(f"Total training and evaluation time: {round(time.time() - start, 3)}s")
        torch.save(ae, "../models/deep_ae_500-200-100-50")

    # # t-SNE
    # with torch.no_grad():
    #     ae.cpu()
    #     encoded = ae.encoder(mnist_test.data.cpu() + noise_const * torch.randn(mnist_test.data.shape))
    #     embedded = TSNE(n_components=2, verbose=1).fit_transform(encoded)
    #     print(embedded.shape)
    #     plt.figure(figsize=(10, 8))
    #     sns.scatterplot(
    #         x=embedded[:, 0], y=embedded[:, 1],
    #         hue=mnist_test.targets.cpu(),
    #         palette=sns.color_palette("hls", 10),
    #         legend="full",
    #         # alpha=0.3
    #     )
    #     plt.show()

    # print the first reconstructions
    from autoencoders import get_noisy_sets
    ae = ae.to('cpu')
    _, ts_data = get_noisy_sets(patch_width=10)
    ts_data = ts_data.data.cpu()
    test_loader = torch.utils.data.DataLoader(ts_data)
    for i, img in enumerate(test_loader):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.reshape(img, (28, 28)))
        ax[1].imshow(torch.reshape(ae(img).data, (28, 28)))
        plt.show()
        if i >= 4:
            break
