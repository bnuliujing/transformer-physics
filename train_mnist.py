import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformer import TraDE


def train(args, model, device, train_dataloader, optimizer, epoch):
    model.train()
    nll_train = 0.
    pbar = tqdm(total=len(train_dataloader.dataset))
    for batch_idx, data in enumerate(train_dataloader):
        pbar.update(len(data))
        data = data.to(device)
        optimizer.zero_grad()
        log_prob = model.log_prob(data)
        nll_loss = -1.0 * torch.mean(log_prob)
        nll_train += -1.0 * torch.sum(log_prob)
        nll_loss.backward()
        optimizer.step()
        pbar.set_description('Train Epoch: {}, NLL: {:.2f}'.format(epoch, nll_loss.item()))
    pbar.close()
    nll_train = nll_train / len(train_dataloader.dataset)
    return nll_train


def test(model, device, test_dataloader):
    model.eval()
    with torch.no_grad():
        nll_test = 0.
        for batch_idx, data in enumerate(test_dataloader):
            data = data.to(device)
            log_prob = model.log_prob(data)
            nll_test += -1.0 * torch.sum(log_prob)
    nll_test = nll_test / len(test_dataloader.dataset)
    return nll_test


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('transformer params')
    group.add_argument('--n', type=int, default=784, help='system size, default: 784')
    group.add_argument('--d-model', type=int, default=64, help='d_model, default: 64')
    group.add_argument('--d-ff', type=int, default=128, help='d_feed_forward, default: 128')
    group.add_argument('--n-layers', type=int, default=2, help='number of layers, default: 2')
    group.add_argument('--n-heads', type=int, default=4, help='number of heads, default: 4')

    group = parser.add_argument_group('training params')
    group.add_argument('--batch-size', type=int, default=32, help='batch size, default: 32')
    group.add_argument('--epochs', type=int, default=50, help='number of epochs to train, default: 50')
    group.add_argument('--lr', type=float, default=1e-3, help='leanring rate, default: 1e-3')
    group.add_argument('--seed', type=float, default=2050, help='random seed, default: 2050')
    group.add_argument('--gpu', type=str, default='0', help='default gpu id, default: 0')
    group.add_argument('--logdir', type=str, default='./out/test', help='log directory to save results')
    group.add_argument('--no-cuda', action='store_true', default=False, help='disable cuda')
    group.add_argument('--sample-image', action='store_true', default=False, help='sample iamges from model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:' + str(args.gpu) if use_cuda else 'cpu')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    mnist_data = np.load('/home/liujing/data/binarized_mnist.npz')
    train_data = torch.from_numpy(mnist_data['train_data'])
    test_data = torch.from_numpy(mnist_data['test_data'])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    model = TraDE(**vars(args)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    for epoch in range(1, args.epochs + 1):
        nll_train = train(args, model, args.device, train_dataloader, optimizer, epoch)
        nll_test = test(model, args.device, test_dataloader)
        print('Epoch: {}, Train NLL: {:.2f}, Test NLL: {:.2f}\n'.format(epoch, nll_train, nll_test))
        with open(os.path.join(args.logdir, 'log.txt'), 'a', newline='\n') as f:
            f.write('{} {} {}\n'.format(epoch, nll_train, nll_test))

        if args.sample_image:
            with torch.no_grad():
                images = model.sample(64).cpu().numpy()
            images = images.reshape(-1, 28, 28)
            plt.figure(figsize=(8, 8))
            for i in range(64):
                plt.subplot(8, 8, i + 1)
                plt.imshow(images[i, :, :], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.logdir, 'samples-{:0>3d}.jpeg'.format(epoch)), dpi=300)
            plt.close()


if __name__ == '__main__':
    main()
