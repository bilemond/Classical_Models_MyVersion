import os
import argparse
import torch
import utils as train_utils
from torch.utils.data import DataLoader
from dataset.zh2EnDataset import zh2EnDataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="Bert_NMT")
    parser.add_argument("--model_name", type=str, default="Bert_NMT")

    # Training settings
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--embedding_dim", type=int, default=128)
    # parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--mode", type=str, default="train")  # train or inference
    args = parser.parse_args()

    save_path = f"../results/{args.pipeline_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    seed = args.seed
    train_utils.set_seed(seed)
    batch_size = args.batch_size

    train_data = zh2EnDataset(padding=128, mode='train')
    dev_data = zh2EnDataset(padding=128, mode='dev')
    test_data = zh2EnDataset(padding=128, mode='test')
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_data, batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size, shuffle=True)

    SrcVocabLen = train_data.getSrcVocabLen()
    TrgVocabLen = train_data.getTrgVocabLen()

    if args.mode == "train":

        for i, (src, trg) in enumerate(tqdm(train_data_loader)):
            print(i, src, trg)
    elif args.mode == "inference":
        pass



