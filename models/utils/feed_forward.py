import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """
        Implements FFN equation.
        :param embedding_dim: embedding_dim 也等于 multi_output_dim
        :param hidden_num: 中间隐藏单元的个数
        :param dropout: dropout
    """
    def __init__(self, embedding_dim, hidden_num, dropout=0.1):  # embedding_dim(d_model), hidden_num(d_ff)
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_num),
            nn.GELU(),  # F.relu
            nn.Dropout(dropout),
            nn.Linear(hidden_num, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
