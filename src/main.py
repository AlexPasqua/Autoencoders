import copy
import json
import os
import pickle

import torch
import argparse
import time
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
import tensorflow as tf
from tensorflow.keras.layers import *
from torchsummary import summary
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_mnist import FastMNIST
import torch.nn.functional as F
from autoencoders import ShallowAutoencoder, DeepAutoencoder, ShallowConvAutoencoder, DeepConvAutoencoder, AbstractAutoencoder, DeepRandomizedAutoencoder
from custom_losses import ContractiveLoss
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
        plt.tight_layout()
        # plt.tick_params(labelbottom=False, labelleft=False)
        plt.axis('off')
        if save:
            plt.savefig(path)
        else:
            plt.show()


def classification_test(ae: AbstractAutoencoder, noisy=False, tr_data=None, tr_labels=None, ts_data=None, ts_labels=None, **kwargs):
    # initial checks and data set
    assert (tr_data is None and tr_labels is None) or (tr_data is not None and tr_labels is not None)
    assert (ts_data is None and ts_labels is None) or (ts_data is not None and ts_labels is not None)
    tr_set, ts_set = get_noisy_sets(**kwargs) if noisy else get_clean_sets()
    if tr_data is None:
        tr_data, tr_labels = tr_set.data, tr_set.labels
    if ts_data is None:
        ts_data, ts_labels = ts_set.data, ts_set.labels

    # pass the data through the encoder
    with torch.no_grad():
        if 'Conv' in ae.type:
            bs = 1000
            n_batches = math.ceil(len(tr_data) / bs)
            tr_data, ts_data = tr_data.cpu(), ts_data.cpu()
            new_tr_data = torch.empty(tr_data.shape[0], ae.encoder[-2].weight.shape[0])
            new_ts_data = torch.empty(ts_data.shape[0], ae.encoder[-2].weight.shape[0])
            for batch_idx in range(n_batches - 1):
                train_data_batch = tr_data[batch_idx * bs: batch_idx * bs + bs].to('cuda:0')
                encoded_batch = ae.encoder(train_data_batch)
                new_tr_data[batch_idx * bs: batch_idx * bs + bs] = encoded_batch
                if batch_idx < math.ceil(len(ts_data) / bs):
                    test_data_batch = ts_data[batch_idx * bs: batch_idx * bs + bs].to('cuda:0')
                    encoded_batch = ae.encoder(test_data_batch)
                    new_ts_data[batch_idx * bs: batch_idx * bs + bs] = encoded_batch
            tr_data = new_tr_data.cpu().numpy()
            ts_data = new_ts_data.cpu().numpy()
        else:
            tr_data = ae.encoder(tr_data).cpu().detach().numpy()
            ts_data = ae.encoder(ts_data).cpu().detach().numpy()
    tr_labels = tf.keras.utils.to_categorical(np.array(tr_labels.cpu()), num_classes=10)
    ts_labels = tf.keras.utils.to_categorical(np.array(ts_labels.cpu()), num_classes=10)

    # create and train a classifier made of one softmax layer
    classifier = tf.keras.Sequential([Dense(input_dim=tr_data.shape[-1], units=10, activation='softmax')])
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')
    classifier.fit(x=tr_data, y=tr_labels, batch_size=32, epochs=10, verbose=0)

    # test the classifier
    return classifier.evaluate(x=ts_data, y=ts_labels, batch_size=len(ts_data))


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

    # ae = torch.load("../models/deepConvAE/deepConvAE_filts(10, 20, 50)_central100_basic__lr0.1_bs32_ep100")
    # ae = torch.load("../models/deepAE/deepAE_(784, 500, 200, 100, 10)_basic_lr0.3_bs32_ep100")
    # ae = torch.load("../models/deepAE/deepAE_(784, 500, 200, 100, 10)_contractive_lr0.1_bs32_ep100")
    # ae.manifold(load=True, path="manifold_img_seq.npy", max_iters=100, thresh=0.0)

    ae = DeepRandomizedAutoencoder(dims=(784, 500))
    # ae = ShallowAutoencoder(784, 500)
    ae.fit(num_epochs=5, bs=128, lr=0.9, momentum=0.9)

    # modes = ('basic', 'denoising')
    # dims_combos = ((784, 500),)
    # combos = ((0.6, 64, 100),)
    # momentum = 0.7
    # for dims in dims_combos:
    #     ae = DeepRandomizedAutoencoder(dims=dims)
    #     for lr, bs, epochs in combos:
    #         path = f"{ae.type}_{dims}_lr{lr}_bs{bs}_ep{epochs}"
    #         print(f"Training model: {path}")
    #         hist = ae.fit(num_epochs=epochs, bs=bs, lr=lr, momentum=momentum)
    #         torch.save(ae, "../models/deepRandAE/" + path)
    #         with open("../results/deepRandAE/hist_" + path, 'w') as f:
    #             json.dump(hist, f)
    # exit()

    # create summary + select best ones for each category
    # loss_types = {'basic': {}, 'denoising': {}, 'contractive': {}}
    # results = {'shallowAE': copy.deepcopy(loss_types),
    #            'deepAE': copy.deepcopy(loss_types),
    #            'shallowConvAE': copy.deepcopy(loss_types),
    #            'deepConvAE': copy.deepcopy(loss_types)}
    # best_for_categ = copy.deepcopy(results)
    # for model_type in results.keys():
    # dir_path = "../models/deepRandAE/"
    # results = {}
    # best_for_categ = copy.deepcopy(results)
    # for filename in os.listdir(dir_path):
    #     ae = torch.load(dir_path + filename)
    #     _, ts_set = get_clean_sets()
    #     ts_data, ts_targets = ts_set.data, ts_set.targets
    #     with torch.no_grad():
    #         outputs = ae(ts_data)
    #         loss = F.mse_loss(torch.flatten(outputs, 1), ts_targets)
    #         results[filename] = loss.item()
    #
    # with open("../results/summary_randAE.json", 'w') as f:
    #     json.dump(results, f, indent='\t')
    #
    # # for model in results.keys():
    # best_keys = sorted(results, key=results.get)
    #
    # print(best_keys)
    # exit()
    #
    # with open("../results/best_for_categ.json", 'w') as f:
    #     json.dump(best_for_categ, f, indent='\t')

    # print the best one for each mode (not conv)
    # with open("../results/best_for_categ.json", 'r') as f:
    #     best_for_categ = json.load(f)
    #     for model_type in ('shallowAE', 'deepAE'):
    #         for mode in ('basic', 'contractive', 'denoising'):
    #             print(f"{mode + ' ' + model_type + ':':25s}{next(iter(best_for_categ[model_type][mode]))}")

    # t-SNE and classification test
    # classif_results = {}
    # paths = (
    #     "deepRandAE/deepRandAE_(784, 500)_lr0.6_bs64_ep100",
    # )
    # for filename in paths:
    #     ae = torch.load("../models/" + filename)
    #     tsne(model=ae, save=True, path="../plots/" + filename.split('/')[1] + '.png')
        # noisy = True if 'denoising' in filename else False
        # res = classification_test(ae=ae, noisy=noisy, noise_const=0.1)
        # classif_results[filename.split('/')[1]] = {'loss': res[0], 'accuracy': res[1]}

        # print the first reconstructions
        # ae.cpu()

    with torch.no_grad():
        ae.cpu()
        _, ts_data = get_clean_sets()
        ts_data = ts_data.data.cpu()
        img = ts_data[10]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.squeeze(img))
        ax[0].set_title("Original image")
        ax[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1].imshow(torch.reshape(ae(torch.unsqueeze(img, 0)).data, (28, 28)))
        ax[1].set_title("Reconstructed image")
        ax[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        plt.show()

        # test_loader = torch.utils.data.DataLoader(ts_data)
        # for i, img in enumerate(test_loader):
        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(torch.reshape(img, (28, 28)))
        #     ax[1].imshow(torch.reshape(ae(img).data, (28, 28)))
        #     plt.show()
        #     if i >= 4:
        #         break

    # with open("../results/classification_randAE_5.json", 'w') as f:
    #     json.dump(classif_results, f, indent='\t')
