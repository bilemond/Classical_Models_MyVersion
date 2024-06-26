import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, head_num, d_model, dropout=0.1):
        '''

        :param head_num: head的数目
        :param d_model: Multi-Head输出的维度，就等于embedding_dim
        :param dropout:
        '''
        super().__init__()
        assert d_model % head_num == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // head_num
        self.head_num = head_num

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query.shape=key.shape=value.shape=(batch_size, seq_len, embedding_dim)
        # mask.shape=(batch_size, 1, seq_len) in Encoder(self-attention)
        # mask.shape=(batch_size, seq_len, seq_len) in Decoder(self-attention)
        # 注意这里的embedding_dim = multi_output_dim，为了保证后面计算的一致性
        if mask is not None:
            # 扩维，原本mask.shap = (batch_size, 1, seq_len)
            # 扩充之后变为(batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1)
            # mask.shape=(batch_size, 1, 1, seq_len)
            # 由于广播机制，所有head的mask都一样
        batch_size = query.size(0)

        # 首先使用线性变换，然后将d_model分配给head_num个头，每个头为 d_k = d_model / head_num
        # 经过线性变换后，query，key，value等的维度不变，还是(batch_size, seq_len, embedding_dim)
        # 在经过view操作后，query.shape=(batch_size, seq_len, head_num, d_k)
        # 再经过transpose操作后，query.shape=(batch_size, head_num, seq_len, d_k)和attention函数要求的维度一致

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # key.shape=value.shape=query.shape=(batch_size, head_num, seq_len, d_k)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # attention 函数操作完之后x.shape=(batch_size, head_num, seq_len, d_k)
        # attn.shape=(batch_size, head_num, seq_len, seq_len)

        # 3) "Concat" using a view and apply a final linear.
        # 下面将multi head的最后一个维度d_k拼接在一起，然后再使用一个线性变换，最终embendding_dim维度不变
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)

        return self.output_linear(x)
