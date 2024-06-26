import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        positionalEncode = torch.zeros(max_len, d_model).float()
        positionalEncode.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        positionalEncode[:, 0::2] = torch.sin(position * div_term)
        positionalEncode[:, 1::2] = torch.cos(position * div_term)
        positionalEncode = positionalEncode.unsqueeze(0)
        self.register_buffer('pe', positionalEncode)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
