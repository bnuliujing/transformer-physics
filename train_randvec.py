import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformer import TraDE


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('transformer params')
    group.add_argument('--n', type=int, default=20, help='system size, default: 20')
    group.add_argument('--d-model', type=int, default=64, help='d_model, default: 64')
    group.add_argument('--d-ff', type=int, default=128, help='d_feed_forward, default: 128')
    group.add_argument('--n-layers', type=int, default=2, help='number of layers, default: 2')
    group.add_argument('--n-heads', type=int, default=4, help='number of heads, default: 4')

    group = parser.add_argument_group('training params')
    group.add_argument('--sample-size', type=int, default=100, help='sample size, default: 100')
    group.add_argument('--epochs', type=int, default=10000, help='number of epochs to train, default: 10000')
    group.add_argument('--lr', type=float, default=1e-4, help='leanring rate, default: 1e-4')
    group.add_argument('--seed', type=float, default=2050, help='random seed, default: 2050')
    group.add_argument('--gpu', type=str, default='0', help='default gpu id, default: 0')
    group.add_argument('--logdir', type=str, default='./out/test', help='log directory to save results')
    group.add_argument('--no-cuda', action='store_true', default=False, help='disable cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:' + str(args.gpu) if use_cuda else 'cpu')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    randvec = torch.randint(2, size=(args.sample_size, args.n), dtype=torch.float, device=args.device)
    model = TraDE(**vars(args)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    pbar = tqdm(total=args.epochs)
    for epoch in range(1, args.epochs + 1):
        pbar.update()
        optimizer.zero_grad()
        log_prob = model.log_prob(randvec)
        nll_loss = -1.0 * torch.mean(log_prob)
        nll_loss.backward()
        optimizer.step()
        pbar.set_description('Epoch: {}, NLL: {:.6f}'.format(epoch, nll_loss))
        with open(os.path.join(args.logdir, 'log.txt'), 'a', newline='\n') as f:
            f.write('{} {}\n'.format(epoch, nll_loss))
    pbar.close()
    print('Theoretical NLL: {:.6f}'.format(np.log(args.sample_size)))


if __name__ == '__main__':
    main()
