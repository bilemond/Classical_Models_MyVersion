import os
import argparse
import torch
import torch.nn as nn
import utils as train_utils
from torch.utils.data import DataLoader
from datasets.zh2EnDataset import zh2EnDataset
from datasets.zh2EnDataset import idx_to_sentence
from models.base_models._transformer import Transformer
from models.utils.bleu import idx_to_word, get_bleu
from tqdm import tqdm
import time
import math


def train(args, train_data_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for batchIndex, (src, trg_input, trg_label, _) in enumerate(train_data_loader):
        src = src.to(device)
        trg_input = trg_input.to(device)
        trg_label = trg_label.to(device)
        trg_hat = model(src, trg_input)

        trg_label_mask = trg_label != args.PAD_ID
        preds = torch.argmax(trg_hat, -1)
        correct = preds == trg_label
        acc = torch.sum(trg_label_mask * correct) / torch.sum(trg_label_mask)

        n, seq_len = trg_label.shape
        trg_hat = torch.reshape(trg_hat, (n * seq_len, -1))
        trg_label = torch.reshape(trg_label, (n * seq_len,))
        loss = criterion(trg_hat, trg_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        print('step :', round((batchIndex / len(train_data_loader)) * 100, 2),
              '% , loss :', loss.item(),
              '% , acc:', acc.item(),
              '% , bleu:', bleu)

    return epoch_loss / len(train_data_loader)


def evaluate(args, dev_data_loader, model, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for batchIndex, (src, trg, _, _) in enumerate(tqdm(dev_data_loader)):
            src = src.to(device)
            trg = trg.to(device)
            trg_label = trg[:, 1:]
            trg_hat = model(src, trg[:, :-1])
            preds = torch.argmax(trg_hat, -1)

            n, seq_len = trg_label.shape
            trg_hat = torch.reshape(trg_hat, (n * seq_len, -1))
            trg_label = torch.reshape(trg_label, (n * seq_len,))
            loss = criterion(trg_hat, trg_label)

            epoch_loss += loss.item()

            total_bleu = []
            for j in range(args.batch_size):
                trg_words = idx_to_word(trg[j], trg_id2word)
                output_words = idx_to_word(preds[j], trg_id2word)
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(dev_data_loader), batch_bleu



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="NMT")
    parser.add_argument("--model_name", type=str, default="Transformer")

    # Training utils settings
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")  # train or inference

    # Training settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--PAD_ID", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)

    args = parser.parse_args()

    save_path = f"../results/{args.pipeline_name}/{args.model_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    train_utils.set_seed(args.seed)
    device = args.device
    batch_size = args.batch_size
    # PAD_ID = args.PAD_ID

    train_data = zh2EnDataset(PAD_ID=args.PAD_ID, padding=args.max_length, mode='train')
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True)

    dev_data = zh2EnDataset(PAD_ID=args.PAD_ID, padding=args.max_length, mode='dev')
    dev_data_loader = DataLoader(dev_data, batch_size, shuffle=True, drop_last=True)

    test_data = zh2EnDataset(PAD_ID=args.PAD_ID, padding=args.max_length, mode='test')
    test_data_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)

    global SrcVocabLen, TrgVocabLen, src_word2id, src_id2word, trg_word2id, trg_id2word
    SrcVocabLen = train_data.getSrcVocabLen()  # cn
    TrgVocabLen = train_data.getTrgVocabLen()  # en
    src_word2id, src_id2word = train_data.getSrcVocab()
    trg_word2id, trg_id2word = train_data.getTrgVocab()

    model = Transformer(src_vocab_size=SrcVocabLen, dst_vocab_size=TrgVocabLen, pad_idx=args.PAD_ID, d_model=args.d_model,
                        d_ff=args.d_ff,
                        n_layers=args.n_layers, heads=args.heads, dropout=args.dropout, max_seq_len=args.max_length)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=args.PAD_ID)

    if args.mode == "train":
        train_losses, test_losses, bleus = [], [], []

        # print_interval = 100
        cnter = 0
        best_loss = float('inf')

        for epoch in range(args.n_epochs):
            start_time = time.time()
            train_loss = train(args, train_data_loader, model, criterion, optimizer)
            valid_loss, bleu = evaluate(args, dev_data_loader, model, criterion)
            end_time = time.time()

            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            bleus.append(bleu)
            epoch_mins, epoch_secs = train_utils.epoch_time(start_time, end_time)

            if valid_loss < best_loss:
                best_loss = valid_loss
                filename = save_path + f"models_epoch{epoch}_{valid_loss}.pth"
                torch.save(model.state_dict(), filename)

            f = open(save_path + f'train_loss.txt', 'w')
            f.write(str(train_losses))
            f.close()

            f = open(save_path + f'bleu.txt', 'w')
            f.write(str(bleus))
            f.close()

            f = open(save_path + f'test_loss.txt', 'w')
            f.write(str(test_losses))
            f.close()

            print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
            print(f'\tBLEU Score: {bleu:.3f}')


    elif args.mode == "inference":
        model = Transformer(src_vocab_size=SrcVocabLen, dst_vocab_size=TrgVocabLen, pad_idx=args.PAD_ID, d_model=512, d_ff=2048,
                        n_layers=6, heads=8, dropout=0.2, max_seq_len=args.max_length)
        model.to(device)
        model.eval()

        filename = save_path + f"models_epoch30.pth"
        model.load_state_dict(torch.load(filename))

        # my_input = ['we', 'should', 'protect', 'environment']
        my_input = ['我', '爱', '你']
        x_batch = torch.LongTensor([[src_word2id[x] for x in my_input]]).to(device)

        cn_sentence = idx_to_sentence(x_batch[0], src_id2word, True)
        print(cn_sentence)

        y_input = torch.ones(batch_size, args.max_length,
                             dtype=torch.long).to(device) * args.PAD_ID
        y_input[0] = src_word2id["<sos>"]
        # y_input = y_batch
        with torch.no_grad():
            for i in range(1, y_input.shape[1]):
                y_hat = model(x_batch, y_input)
                for j in range(batch_size):
                    y_input[j, i] = torch.argmax(y_hat[j, i - 1])
        output_sentence = idx_to_sentence(y_input[0], trg_id2word, True)
        print(output_sentence)



