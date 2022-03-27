import torch


def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()
