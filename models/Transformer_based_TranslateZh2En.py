import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from models.utils.feed_forward import PositionwiseFeedForward
from models.attention.multi_head import MultiHeadedAttention
from models.embedding.position import PositionalEmbedding
from utils import clones


class TranslateZh2En(nn.Module):
    """
    英文翻译为中文的模型, 通用的Encoder和Decoder框架
    """

    def __init__(self, head_num, embedding_dim, hidden_num, size, dropout, decoder_dim, enVocabLen, zhVocabLen, N=6):
        """
        构造函数, 使用Encoder和Decoder通用框架实现一个Transformer模型
        :param encoder: 编码器, 本例中使用Transformer的Encoder
        :param decoder: 解码器, 本例中使用Transformer的Decoder
        :param src_embedding: 源语言的embedding
        :param dst_embedding: 目标语言的embedding
        :param generator: 将Decoder输入的隐状态输入一个全连和softmax用于输出概率
        """
        super(TranslateZh2En, self).__init__()
        self.encoder = TransformerEncoder(size=size, head_num=head_num, embedding_dim=embedding_dim, hidden_num=hidden_num, dropout=dropout, N=N)
        self.decoder = TransformerDecoder(size=size, head_num=head_num, embedding_dim=embedding_dim, hidden_num=hidden_num, dropout=dropout, N=N)
        self.positon = PositionalEncoding(embedding_dim, dropout)
        self.src_embedding = nn.Sequential(Embeddings(embedding_dim, zhVocabLen), self.positon)
        self.dst_embedding = nn.Sequential(Embeddings(embedding_dim, enVocabLen), self.positon)
        self.generator = Generator(decoder_dim=decoder_dim, vocab_len=enVocabLen)


    def forward(self, english_seq, chinese_seq, english_mask, chinese_mask):
        """
        前向传播函数
        :param english_seq: 英文序列
        :param chinese_seq: 中文序列
        :param english_mask: 原始序列的mask, 主要作用是mask掉padding
        :param chinese_mask: 目标序列的mask，防止标签泄露，所以是一个下三角矩阵
        :return:
        """
        # english_seq.shape=(batch_size, seq_len)
        # chinese_seq.shape=(batch_size, seq_len)
        # english_mask.shape=(batch_size, 1, seq_len)
        # chinese_mask.shape=(batch_size, seq_len, seq_len)
        memory = self.encode(chinese_seq=chinese_seq, chinese_mask=chinese_mask)
        # memory.shape=(batch_size, seq_len, embedding_dim)
        output = self.decode(memory=memory, english_seq=english_seq, english_mask=english_mask,
                             chinese_mask=chinese_mask)
        return output

    def encode(self, chinese_seq, chinese_mask):
        return self.encoder(self.src_embedding(chinese_seq), chinese_mask)

    def decode(self, memory, chinese_mask, english_seq, english_mask):
        return self.decoder(self.dst_embedding(english_seq), memory, chinese_mask, english_mask)


class Generator(nn.Module):
    """
    根据Decoder输出的隐藏状态输出一个词
    """

    def __init__(self, decoder_dim, vocab_len):
        """
        generator的构造函数
        :param decoder_dim: decoder的输出的维度
        :param vocab_len: 词典大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(decoder_dim, vocab_len)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输出x，这里是decoder的输出
        :return:
        """
        proj = self.proj(x)
        # 做一次softmax, 返回的是softmax的log值
        # log_softmax + NLLLoss 效果类似与 softmax + CrossEntropyLoss
        return F.log_softmax(proj, dim=-1)


class TransformerEncoder(nn.Module):
    """
    Transformer的Encoder部分
    由6个EncoderLayer堆叠而成，而每个EncoderLayer又包含一个self-attention层和全连层
    """

    def __init__(self, size, head_num, embedding_dim, hidden_num, dropout, N=6):
        """
        构造函数
        :param encode_layer: encode_layer, 包含一个self-attention层和一个全连层
        :param N: encoder_layer 重复的次数，transformer中为6
        """
        super(TransformerEncoder, self).__init__()
        # 6层encode_layer
        self.EncodeLayer = EncodeLayer(head_num, embedding_dim, hidden_num, size, dropout)
        self.layers = clones(self.EncodeLayer, N)
        # 再加一层Norm层
        self.norm = nn.LayerNorm(self.EncodeLayer.size)

    def forward(self, x, mask):
        """
        前向函数
        :param x: 待编码的数据
        :param mask:
        :return: Transformer的Encoder编码后的数据
        """
        # x.shape=(batch_size, seq_len, embedding_dim)
        # mask.shape=(batch_size, 1, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
            # x.shape=(batch_size, seq_len, embedding_dim)
        # 最后加一层Normalization层
        return self.norm(x)


class EncodeLayer(nn.Module):
    """
    transformer中Encoder部分的encode layer，一共6个encode layer组成一个Encoder
    一个encode layer包含两个子层, 每个子层包括self-attention、feed_forward等操作
    """

    def __init__(self, head_num, embedding_dim, hidden_num, size, dropout):
        """
        构造函数
        :param size:
        :param self_attention:
        :param feed_forward:
        :param dropout:
        :return:
        """
        super(EncodeLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(head_num=head_num, d_model=embedding_dim, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_dim=embedding_dim, hidden_num=hidden_num, dropout=dropout)
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


class SubLayer(nn.Module):
    """
    Encode Layer或者Decode Layer的一个子层(通用结构)
    这里会构造LayerNorm 和 Dropout，但是Self-Attention 和 Dense 不在这里构造，作为参数传入
    """

    def __init__(self, size, dropout):
        """
        构造函数
        :param size:
        :param dropout:
        """
        super(SubLayer, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, function_layer):
        """
        前向函数
        :param x: 传入数据
        :param sublayer: 功能层，Encoder中可以为self-attention 或者 feed_forward中的一个
        :return:
        """
        # x.shape=(batch_size, seq_len, embedding_dim)

        # x + 对应那个残差操作
        # 这个dropout和原文不太一样，加在这里是为了防止过拟合吧
        # layer normalization的位置也和原文不太一样，原文是放在最后，这里是放在最前面并且在最后一层再加一层layer normalization
        # 为了方便SubEncodeLayer层的复用，self-attention和feed_forward作为参数<function_layer>
        return x + self.dropout(function_layer(self.norm(x)))


class TransformerDecoder(nn.Module):
    """
    Transformer的Decoder端，包含6个DecodeLayer, 每个DecoderLayer又包含三个子层，分别是self-attention层，attention层和feed_foraward层
    """

    def __init__(self, size, head_num, embedding_dim, hidden_num, dropout, N=6):
        """
        构造函数
        :param layer: DecodeLayer
        :param N: Transformer中的Deocder包含6个Deocde层，所以N为6
        """
        super(TransformerDecoder, self).__init__()
        self.decode_layer = DecodeLayer(size, head_num, embedding_dim, hidden_num, dropout)
        self.layers = clones(self.decode_layer, N)
        self.norm = nn.LayerNorm(size)


    def forward(self, x, memory, src_mask, dst_mask):
        """
        前向传播函数
        :param x: 自回归输入，一开始只有一个起始符
        :param memory: 应该是Encoder编码的信息
        :param src_mask: padding mask
        :param dst_mask: 防止标签泄露的mask
        :return:
        """
        # x.shape=(batch_size, seq_len, embedding_dim)
        # memory.shape=(batch_size, seq_len, embedding_dim)
        # src_mask.shpae=(batch_size, 1, seq_len)
        # dst_mask.shape=(batch_size, 1, seq_len)
        for layer in self.layers:
            x = layer(x, memory, src_mask, dst_mask)
        return self.norm(x)


class DecodeLayer(nn.Module):
    """
    Transformer的Decoder的一个Decode层
    包含三个子层，分别对应self-attention, attention(与Encoder编码的memory做attention), feed_forward
    """

    def __init__(self, size, head_num, embedding_dim, hidden_num, dropout):
        """
        构造函数
        :param size:
        :param self_attenton: self attention层
        :param attention: 与Encoder编码的memory做attention
        :param feed_forward: 全连层
        :param dropout: 防止过拟合加的dropout层
        """
        super(DecodeLayer, self).__init__()
        self.size = size
        self.attention = MultiHeadedAttention(head_num=head_num, d_model=embedding_dim, dropout=dropout)
        self.sublayer = clones(SubLayer(size, dropout), 3)
        self.self_attention = MultiHeadedAttention(head_num=head_num, d_model=embedding_dim, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_dim=embedding_dim, hidden_num=hidden_num, dropout=dropout)

    def forward(self, x, memory, src_mask, dst_mask):
        """
        前向传播函数
        :param x: 输出
        :param memory: Encoder编码的memory
        :param src_mask: 源端的padding mask
        :param dst_mask: 防止标签泄露的mask
        :return:
        """
        # x.shape=(batch_size, seq_len, embedding_dim)
        # memory.shape=(batch_size, seq_len, embedding_dim)
        # src_mask.shpae=(batch_size, 1, seq_len)
        # dst_mask.shape=(batch_size, 1, seq_len)
        m = memory
        # 这里是self-attention子层，对于self-attention来说，Query, Key和Value都是等于x
        # lambda表达式中的x只是一个形式参数，不是输入x
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, dst_mask))
        # 第二个子层是attention层，与memory做attention, 此时的Query为x，Key为m，Value也为m
        x = self.sublayer[1](x, lambda x: self.attention(x, m, m, src_mask))
        # 第三个子层是一个全连层
        output = self.sublayer[2](x, self.feed_forward)
        return output


class Embeddings(nn.Module):
    """
    原始输出的seq是每个word在vocab中的index的序列，需要embedding序列
    """

    def __init__(self, embedding_dim, vocab_len):
        """
        构造函数
        :param embedding_dim: embedding的维度
        :param vocab_len: 词表vocab的长度
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_len, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # 这里看懂为啥要乘以embedding_dim的平方根
        return self.lut(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.PositionalEmbedding = PositionalEmbedding(embedding_dim, max_len)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入
        :return:
        """
        x = x + self.PositionalEmbedding(x)
        return self.dropout(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help="train or test")
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=128, help='number of word embedding')
    parser.add_argument('--gpu', type=int, default=0, help='GPU No, only support 1 or 2')
    parser.add_argument('--head_num', type=int, default=8, help="Multi head number")
    parser.add_argument('--hidden_num', type=int, default=2048, help="hidden neural number")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--padding', type=int, default=50, help="padding length")

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    translateZh2En = TranslateZh2En().to(device)
