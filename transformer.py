import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from utils import gen_all_binary_vectors


class PositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, n).reshape(n, 1)
        pos_embedding = torch.zeros((n, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding


class TraDE(BaseModel):
    """
    Transformers for density estimation or stat-mech problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']

        self.fc_in = nn.Linear(1, self.d_model)
        self.positional_encoding = PositionalEncoding(self.n, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=0,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))

    def forward(self, x):
        x = torch.cat((torch.ones(x.shape[0], 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.unsqueeze(2)))  # (batch_size, n, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)
        return torch.sigmoid(self.fc_out(x)).squeeze(2)

    def log_prob(self, x):
        x_hat = self(x)
        log_prob = torch.log(x_hat + 1e-8) * x + torch.log(1 - x_hat + 1e-8) * (1 - x)
        return log_prob.sum(dim=1)

    def sample(self, batch_size):
        samples = torch.randint(0, 2, size=(batch_size, self.n), dtype=torch.float, device=self.device)
        for i in range(self.n):
            x_hat = self(samples)
            samples[:, i] = torch.bernoulli(x_hat[:, i])
        return samples


if __name__ == '__main__':
    kwargs_dict = {
        'n': 12,
        'd_model': 512,
        'd_ff': 2048,
        'n_layers': 6,
        'n_heads': 2,
        'device': 'cpu'
    }

    model = TraDE(**kwargs_dict).to(kwargs_dict['device'])
    print(model)

    # test normalization condition
    x = gen_all_binary_vectors(kwargs_dict['n']).to(kwargs_dict['device'])
    log_prob = model.log_prob(x)
    print(log_prob.exp().sum())

    # test sampling
    y = model.sample(10)
    print(y)