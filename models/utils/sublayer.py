import torch.nn as nn
from .LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # x.shape=(batch_size, seq_len, embedding_dim)

        # x + 对应那个残差操作
        # 这个dropout和原文不太一样，加在这里是为了防止过拟合吧
        # layer normalization的位置也和原文不太一样，原文是放在最后，这里是放在最前面并且在最后一层再加一层layer normalization
        # 为了方便SubEncodeLayer层的复用，self-attention和feed_forward作为参数<function_layer>

        return x + self.dropout(sublayer(self.norm(x)))
