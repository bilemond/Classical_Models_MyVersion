import os
import argparse
import torch
import torch.nn as nn
import utils as train_utils
from torch.utils.data import DataLoader
from datasets.zh2EnDataset import zh2EnDataset
from datasets.zh2EnDataset import idx_to_sentence
from models.base_models._transformer import Transformer
from models.utils.metric import compute_bleu
from tqdm import tqdm
import time
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="Transformer_NMT")
    parser.add_argument("--model_name", type=str, default="Bert_NMT")

    # Training settings
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--embedding_dim", type=int, default=128)
    # parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="train")  # train or inference
    parser.add_argument("--max_length", type=int, default=50)
    args = parser.parse_args()

    save_path = f"../results/{args.pipeline_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    seed = args.seed
    train_utils.set_seed(seed)
    device = args.device
    batch_size = args.batch_size
    PAD_ID = 0

    train_data = zh2EnDataset(PAD_ID=PAD_ID, padding=args.max_length, mode='train')
    dev_data = zh2EnDataset(PAD_ID=PAD_ID, padding=args.max_length, mode='dev')
    test_data = zh2EnDataset(PAD_ID=PAD_ID, padding=args.max_length, mode='test')
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_data, batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size, shuffle=True)

    SrcVocabLen = train_data.getSrcVocabLen()  # cn
    TrgVocabLen = train_data.getTrgVocabLen()  # en
    src_word2id, src_id2word = train_data.getSrcVocab()
    trg_word2id, trg_id2word = train_data.getTrgVocab()

    if args.mode == "train":
        bleu = []
        print_interval = 100
        model = Transformer(src_vocab_size=SrcVocabLen, dst_vocab_size=TrgVocabLen, pad_idx=PAD_ID, d_model=512, d_ff=2048,
                        n_layers=6, heads=8, dropout=0.2, max_seq_len=50)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        tic = time.time()
        cnter = 0

        for epoch in range(args.n_epochs):
            for batchIndex, (src, trg_input, trg_label, _) in enumerate(tqdm(train_data_loader)):
                src = src.to(device)
                trg_input = trg_input.to(device)
                trg_label = trg_label.to(device)
                trg_hat = model(src, trg_input)
                trg_label_mask = trg_label != PAD_ID
                preds = torch.argmax(trg_hat, -1)
                correct = preds == trg_label
                acc = torch.sum(trg_label_mask * correct) / torch.sum(trg_label_mask)

                n, seq_len = trg_label.shape
                trg_hat = torch.reshape(trg_hat, (n * seq_len, -1))
                trg_label = torch.reshape(trg_label, (n * seq_len,))
                loss = citerion(trg_hat, trg_label)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                if cnter % print_interval == 0:
                    toc = time.time()
                    interval = toc - tic
                    minutes = int(interval // 60)
                    seconds = int(interval % 60)
                    print(f'epoch {epoch}'
                          f' {cnter:08d} {minutes:02d}:{seconds:02d}'
                          f' loss: {loss.item()} acc: {acc.item()}')
                cnter += 1

            if epoch % 10 == 0 and epoch != 0:
                filename = save_path + f"models_epoch{epoch}.pth"
                torch.save(model.state_dict(), filename)

            # # valid
            # with torch.no_grad():
            #     blue_socres = []
            #
            #     for valid_index, (src, trg_input, trg_label, len_trg_line) in enumerate(tqdm(dev_data_loader)):
            #         # englishSeq.shape=(batch_size, paded_seq_len)
            #         # chineseSeq.shape=(batch_size, paded_seq_len)
            #         # chineseSeqY.shape=(batch_size, padded_seq_len)
            #         src = src.to(device)
            #         trg_input = trg_input.to(device)
            #         trg_label = trg_label.to(device)
            #
            #         src_mask = (src != PAD_ID).unsqueeze(-2)
            #         # src_mask.shape=(batch_size, 1, seq_len)
            #         memory = translateEn2Zh.encode(src, src_mask)
            #         # memory.shape=(batch_size, seq_len, embedding_dim)
            #         translate = torch.ones(args.batch_size, 1).fill_(0).type_as(src.data)
            #         # translate_ = trg_label[:, 0]
            #         # ys.shape=(1, 1)
            #         for i in range(args.padding):
            #             translate_mask = make_std_mask(translate, 3)
            #             out = translateEn2Zh.decode(memory, src_mask, translate, translate_mask)
            #             prob = translateEn2Zh.generator(out[:, -1])
            #             _, next_word = torch.max(prob, dim=1)
            #             next_word = next_word.unsqueeze(1)
            #             translate = torch.cat([translate, next_word], dim=1)
            #             # translate_ = chineseSeqY[:, :]
            #         blue_socres += compute_bleu(translate, trg_label, len_trg_line)
            #         if (valid_index + 1) % 1 == 0:
            #             reference_sentence = trg_label[0].tolist()
            #             translate_sentence = translate[0].tolist()
            #             src_sentence = src[0].tolist()
            #             reference_sentence_len = len_trg_line.tolist()[0]
            #             if 1 in translate_sentence:
            #                 index = translate_sentence.index(1)
            #             else:
            #                 index = len(translate_sentence)
            #             print("原文: {}".format(" ".join([src_id2word[x] for x in src_sentence])))
            #             print("机翻译文: {}".format("".join([trg_id2word[x] for x in translate_sentence[:index]])))
            #             print("参考译文: {}".format(
            #                 "".join([trg_id2word[x] for x in reference_sentence[:reference_sentence_len]])))
            #     epoch_bleu = np.sum(blue_socres) / len(blue_socres)
            #     bleu.append(epoch_bleu)




    elif args.mode == "inference":
        model = Transformer(src_vocab_size=SrcVocabLen, dst_vocab_size=TrgVocabLen, pad_idx=PAD_ID, d_model=512, d_ff=2048,
                        n_layers=6, heads=8, dropout=0.2, max_seq_len=args.max_length)
        model.to(device)
        model.eval()

        filename = save_path + f"models_epoch{args.n_epochs}.pth"
        model.load_state_dict(torch.load(filename))

        # my_input = ['we', 'should', 'protect', 'environment']
        my_input = ['我', '爱', '你']
        x_batch = torch.LongTensor([[src_word2id[x] for x in my_input]]).to(device)

        cn_sentence = idx_to_sentence(x_batch[0], src_id2word, True)
        print(cn_sentence)

        y_input = torch.ones(batch_size, args.max_length,
                             dtype=torch.long).to(device) * PAD_ID
        y_input[0] = src_word2id['<S>']
        # y_input = y_batch
        with torch.no_grad():
            for i in range(1, y_input.shape[1]):
                y_hat = model(x_batch, y_input)
                for j in range(batch_size):
                    y_input[j, i] = torch.argmax(y_hat[j, i - 1])
        output_sentence = idx_to_sentence(y_input[0], trg_id2word, True)
        print(output_sentence)



