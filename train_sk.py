import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sk import SKModel
from transformer import TraDE


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('sk params')
    parser.add_argument('--beta-init', type=float, default=0.1, help='initial beta, default: 0.1')
    parser.add_argument('--beta-final', type=float, default=5.0, help='final beta, default: 5.0')
    parser.add_argument('--beta-interval', type=float, default=0.1, help='interval of annealing, default: 0.1')

    group = parser.add_argument_group('transformer params')
    group.add_argument('--n', type=int, default=20, help='system size, default: 20')
    group.add_argument('--d-model', type=int, default=64, help='d_model, default: 64')
    group.add_argument('--d-ff', type=int, default=128, help='d_feed_forward, default: 128')
    group.add_argument('--n-layers', type=int, default=2, help='number of layers, default: 2')
    group.add_argument('--n-heads', type=int, default=4, help='number of heads, default: 4')

    group = parser.add_argument_group('training params')
    group.add_argument('--batch-size', type=int, default=1000, help='batch size, default: 1000')
    group.add_argument('--epochs', type=int, default=5000, help='number of epochs to train, default: 5000')
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

    sk = SKModel(n=args.n, device=args.device)
    model = TraDE(**vars(args)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    steps = (args.beta_final - args.beta_init) / args.beta_interval + 1
    for beta in np.linspace(args.beta_init, args.beta_final, int(steps)):
        f_list = []
        f_std_list = []
        e_list = []
        s_list = []

        pbar = tqdm(total=args.epochs)
        for step in range(1, args.epochs + 1):
            pbar.update()
            optimizer.zero_grad()
            with torch.no_grad():
                samples = model.sample(args.batch_size)
            log_prob = model.log_prob(samples)
            assert not samples.requires_grad
            assert log_prob.requires_grad
            with torch.no_grad():
                samples = samples * 2.0 - 1.0
                energy = -0.5 * torch.sum((samples @ sk.J) * samples, dim=1)
                loss = log_prob + beta * energy
            loss_reinforce = torch.mean(log_prob * (loss - loss.mean()))
            loss_reinforce.backward()
            optimizer.step()
            with torch.no_grad():
                free_energy = loss.mean().item() / beta / args.n
                free_energy_std = loss.std().item() / beta / args.n
                entropy = -1.0 * log_prob.mean().item() / args.n
                energy = energy.mean().item() / args.n
                f_list.append(free_energy)
                f_std_list.append(free_energy_std)
                e_list.append(energy)
                s_list.append(entropy)
                pbar.set_description('beta: {:.2f}, f: {:.6f}, std: {:.6f}, e: {:.6f}, s: {:.6f}'.format(
                    beta, free_energy, free_energy_std, energy, entropy))
        pbar.close()

        free_energy = np.mean(f_list[-50:])
        f_std = np.mean(f_std_list[-50:])
        e = np.mean(e_list[-50:])
        s = np.mean(s_list[-50:])
        with open(os.path.join(args.logdir, 'log.txt'), 'a', newline='\n') as f:
            f.write('{:.2f} {} {} {} {}\n'.format(beta, free_energy, f_std, e, s))


if __name__ == '__main__':
    main()
