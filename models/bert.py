import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, h_dim):
        """

        :param vocab_size:
        :param h_dim:
        :return:
        """
        super(WordEmbedding, self).__init__()
        self.h_dim = h_dim
        self.word_embeddings = nn.Embedding(vocab_size, h_dim)

class Encoder(nn.Module):
    def __init__(self, vocab_size, h_dim, pf_dim, n_heads, n_layers, dropout, device, max_length=200):
        """

        :return:
        """
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.device = device
        self.word_embeddings = WordEmbedding(vocab_size, h_dim)
        self.positional_encoding = PositionalEncodings(h_dim, max_length)
        self.layers = nn.ModuleList([EncodeLayer(h_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([h_dim])).to(device)

    def forward(self, src, src_mask):
        """

        :param src:
        :param src_mask:
        :return:
        """
        output = self.word_embeddings(src) * self.scale
        src_len = src.shape[1]

        pos = self.positional_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return output

class EncodeLayer(nn.Module):
    """
    transformer中Encoder部分的encode layer，一共6个encode layer组成一个Encoder
    一个encode layer包含两个子层, 每个子层包括self-attention、feed_forward等操作
    """
    def __init__(self, size, self_attention, feed_forward, dropout, device):
        """
        构造函数
        :param size:
        :param self_attention:
        :param feed_forward:
        :param dropout:
        :return:
        """
        super(EncodeLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        # 两个子层
        self.sublayer = clones(SubLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向函数
        :param x: 输入
        :param mask:
        :return:
        """
        # x.shape=(batch_size, seq_len, embedding_dim)
        # mask.shape=(batch_size, 1, seq_len)
        ##
        # 进行self attention
        # self-attention需要四个输入， 分别是Query，Key，Value和最后的Mask
        # lambda 表达式中的x不是EncodeLayer的输入x，而是一个形式参数，可以是y或者其他任何名称，最终输入到lambda中的应该是SubEncodeLayer层中的self.norm(x)
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        # 进行feed_forward
        return self.sublayer[1](x, self.feed_forward)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x