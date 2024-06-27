import torch
from torch.utils.data import Dataset
import os
import numpy as np

def idx_to_sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res

class zh2EnDataset(Dataset):
    def __init__(self, PAD_ID, padding=128, mode='train'):
        super().__init__()
        print("load {} data".format(mode))
        assert mode in ['train', 'dev', 'test'], "mode must in ['train', 'dev', 'test']"
        self.mode = mode
        self.padding = padding
        self.PAD_ID = PAD_ID

        self.src_filename = "../data/NiuTrans/{}.zh".format(mode)
        self.trg_filename = "../data/NiuTrans/{}.en".format(mode)
        self.src_vocab_filename = "../data/NiuTrans/vocab.zh"
        self.trg_vocab_filename = "../data/NiuTrans/vocab.en"
        self.src_line, self.trg_line = self.__read_data()
        self.src_vocab, self.trg_vocab = self.__deal_vocab()

    def __len__(self):
        return len(self.src_line)

    def getSrcVocabLen(self):
        return self.src_vocab.size

    def getTrgVocabLen(self):
        return self.trg_vocab.size

    def getSrcVocab(self):
        return self.src_vocab.word2id, self.src_vocab.id2word

    def getTrgVocab(self):
        return self.trg_vocab.word2id, self.trg_vocab.id2word

    def __getitem__(self, index):

        src_line = self.src_line[index]
        trg_line = self.trg_line[index]
        len_trg_line = len(trg_line) - 1

        src_id = []
        trg_id = []
        for src_tokens in src_line:
            src_id.append(self.src_vocab.word2id[src_tokens]
                                 if src_tokens in self.src_vocab.word2id else self.src_vocab.word2id['<unk>'])
        for trg_tokens in trg_line:
            trg_id.append(self.trg_vocab.word2id[trg_tokens]
                                 if trg_tokens in self.trg_vocab.word2id else self.trg_vocab.word2id['<unk>'])

        if len(src_id) < self.padding:
            src_id.extend([self.PAD_ID] * (self.padding - len(src_id)))
        else:
            src_id = src_id[:self.padding]
        if len(trg_id) < self.padding:
            trg_id.extend([self.PAD_ID] * (self.padding - len(trg_id)))
        else:
            trg_id = trg_id[:self.padding]
        src_id = torch.tensor(src_id)
        trg_id = torch.tensor(trg_id)
        if self.mode == 'train':
            return src_id, trg_id[:-1], trg_id[1:], len_trg_line
        else:
            return src_id, trg_id, trg_id, len_trg_line

    def __read_data(self):
        print("Reading data from {} and {}".format(self.src_filename, self.trg_filename))
        with open(self.src_filename, 'r', encoding='utf-8') as f1, open(self.trg_filename, 'r', encoding='utf-8') as f2:
            src_lines = f1.readlines()
            trg_lines = f2.readlines()

        assert len(src_lines) == len(trg_lines), "The number of lines in source and target files are not equal"

        print("Read {} lines".format(len(src_lines)))
        print("preprocessing data...")
        src_lines = [['<SOS>'] + line.strip().split() + ['<EOS>'] for line in src_lines]
        trg_lines = [['<SOS>'] + line.strip().split() + ['<EOS>'] for line in trg_lines]

        return src_lines, trg_lines

    def __deal_vocab(self):
        with open(self.src_vocab_filename, 'r', encoding='utf-8') as f1, open(self.trg_vocab_filename, 'r', encoding='utf-8') as f2:
            src_vocab_list = f1.readlines()
            trg_vocab_list = f2.readlines()
        src_vocab_list.insert(self.PAD_ID, '<PAD>')
        trg_vocab_list.insert(self.PAD_ID, '<PAD>')

        src_vocab = type('Vocabulary', (), {})()
        src_vocab.word2id = {word.strip(): idx for idx, word in enumerate(src_vocab_list)}
        src_vocab.id2word = {idx: word.strip() for idx, word in enumerate(src_vocab_list)}
        src_vocab.size = len(src_vocab_list)

        trg_vocab = type('Vocabulary', (), {})()
        trg_vocab.word2id = {word.strip(): idx for idx, word in enumerate(trg_vocab_list)}
        trg_vocab.id2word = {idx: word.strip() for idx, word in enumerate(trg_vocab_list)}
        trg_vocab.size = len(trg_vocab_list)

        return src_vocab, trg_vocab