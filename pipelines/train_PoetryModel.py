import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from models.PoetryModel import PoetryModel
import argparse
import time
import utils as train_utils
import os

def prepareData():
    datas = np.load(r"D:\MyPhd\2-Class-phd1-2\深度学习\作业\实验3_202318013229058_刘梦迪\tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=16,
                            shuffle=True,
                            num_workers=2)
    return dataloader, ix2word, word2ix


def generate(args, start_words, ix2word, word2ix, model_path):
    # 读取模型
    model = PoetryModel(len(word2ix), args.embedding_dim, args.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)

    results = list(start_words)
    start_words_len = len(start_words)

    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(args.device)
    hidden = None

    with torch.no_grad():
        for i in range(args.max_gen_len):
            output, hidden = model(input, hidden)
            # 如果在给定的句首中，input 为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
            # 否则将 output 作为下一个 input 进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break

    return results


def gen_acrostic(args, start_words, ix2word, word2ix, model_path):
    # 读取模型
    model = PoetryModel(len(word2ix), args.embedding_dim, args.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)

    # 读取唐诗的“头”
    results = []
    start_word_len = len(start_words)

    # 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(args.device)
    hidden = None

    index = 0  # 指示已生成了多少句
    pre_word = '<START>'  # 上一个词

    # 生成藏头诗
    for i in range(args.max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        # 如果遇到标志一句的结尾，喂入下一个“头”
        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果生成的诗已经包含全部“头”，则结束
            if index == start_word_len:
                break
            # 把“头”作为输入喂入模型
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)

        # 否则，把上一次预测作为下一个词输入
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, default="PoetryModel")
    parser.add_argument("--model_name", type=str, default="PoetryModel")
    # Training settings
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max_gen_len", type=int, default=125)
    parser.add_argument("--max_gen_len_acrostic", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--mode", type=str, default="inference")  # train or inference
    args = parser.parse_args()

    save_path = f"../results/{args.pipeline_name}/"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    seed = args.seed
    train_utils.set_seed(seed)

    dataloader, ix2word, word2ix = prepareData()
    # model_path = r'D:\MyPhd\2-MyResearch\0github\Classical_Models_MyVersion\pipelines\model1718349508.3736255.pth'  # 预训练模型路径

    if args.mode == 'train':
        model = PoetryModel(len(word2ix),
                            embedding_dim=args.embedding_dim,
                            hidden_dim=args.hidden_dim)
        # if model_path:
        #     model.load_state_dict(torch.load(model_path))

        model.to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            for batch_idx, data in enumerate(dataloader):
                data = data.long().transpose(1, 0).contiguous()
                data = data.to(args.device)
                input, target = data[:-1, :], data[1:, :]
                output, _ = model(input)
                loss = criterion(output, target.view(-1))

                if batch_idx % 900 == 0:
                    print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data[1]), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader), loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        filename = save_path + f"models_{str(time.time())}.pth"
        torch.save(model.state_dict(), filename)

    elif args.mode == 'inference':
        start_words = '湖光秋月两相和'
        start_words_acrostic = '湖光秋月两相和'
        filename = save_path + f"models_1718402923.7745762.pth"
        # filename = f"../pretrain/" + f"model.pth"
        results = generate(args, start_words, ix2word, word2ix, filename)
        poetry = ''
        for word in results:
            poetry += word
            if word == '。' or word == '!':
                poetry += '\n'
        print(poetry)

        results_acrostic = gen_acrostic(args, start_words_acrostic, ix2word, word2ix, filename)
        poetry1 = ''
        for word in results_acrostic:
            poetry1 += word
            if word == '。' or word == '!':
                poetry1 += '\n'
        print(poetry1)
        # print(results_acrostic)

