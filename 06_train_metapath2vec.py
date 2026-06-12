import numpy as np
import random
import time
import tqdm
import dgl
import sys
import os

embedding_dimensions = 32
iteration_time = 1

####此处需要修改，注意后续embedding results存储位置
path_in = './data/test5'# 获取目标文件夹的路径
path_out = './data/test6'
####此处需要修改，注意后续embedding results存储位置


start =time.time()
print('Running time: %s Seconds'%(start))

import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
parser = argparse.ArgumentParser(description = 'Metapath2vec')
parser.add_argument('--aminer',action = 'store_true',help = 'Use AMiner dataset')
# 输入文件os.path.join(path, "result.txt")
parser.add_argument('--path',type = str,help = 'input_path',default = os.path.join(path_in, "output_path.txt"))
# 输出文件
parser.add_argument('--output_file',type = str,help = 'output_file',default = os.path.join(path_out, "result.txt"))
# embedding维度
parser.add_argument('--dim',default = embedding_dimensions,type = int,help = 'embedding dimensions')
# 窗口大小
parser.add_argument('--window_size',default = 7,type = int,help = 'context window size')
# 迭代次数
parser.add_argument('--iterations',default = iteration_time,type = int,help = 'iterations')
# batch size
parser.add_argument('--batch_size',default = 50,type = int,help = 'batch_size')
# 0:metapath2vec; 1:metapath2vec++
parser.add_argument('--care_type',default = 0,type = int,help="if 1, heterogeneous negative sampling, else normal negative sampling")
# 学习率
parser.add_argument('--initial_lr',default = 0.025,type = float,help = 'learning rate')
# skip2gram模型的词频
parser.add_argument('--min_count',default = 0,type = int,help = 'min count')
# 资源数
parser.add_argument('--num_workers',default = 16,type = int,help = 'number of workders')
args = parser.parse_args(args = [])
import numpy as np
import torch
from torch.utils.data import Dataset


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, dataset, min_count, care_type):
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = dataset.fn
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding='ISO-8859-1'):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1
                        #print(word)
                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")
        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
        print('Total embeddings:' + str(len(self.word2id)))

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def getNegatives(self, target, size):
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[:self.negpos]))
        return response
class Metapath2vecDataset(Dataset):
    def __init__(self,data,window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName,encoding='ISO-8859-1')
    def __len__(self):
        return self.data.sentences_count
    def __getitem__(self,idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0,0)
                line = self.input_file.readline()
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                       w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                pair_catch = []
                for i,u in enumerate(word_ids):
                    for j,v in enumerate(
                        word_ids[max(i - self.window_size,0):i + self.window_size]
                    ):
                        assert u < self.data.word_count
                        assert v < self.data.word_count
                        if i == j:
                            continue
                        pair_catch.append((u,v,self.data.getNegatives(v,5)))
                return pair_catch
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u,_,_ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _,v,_ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _,_,neg_v in batch if len(batch) > 0]
        assert len(all_u) == len(all_v) == len(all_neg_v)
        # print(1)
        # return torch.LongTensor(all_u),torch.LongTensor(all_v),torch.LongTensor(all_neg_v)  #直接用这个会报警告，说是很慢
        return torch.LongTensor(np.array(all_u)), torch.LongTensor(np.array(all_v)), torch.LongTensor(
            np.array(all_neg_v))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# u_embedding:center word
# v_embedding:Embedding for neighbor words
class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)  # [batch_size,embedding_dim]
        emb_v = self.v_embeddings(pos_v)  # [batch_size,embedding_dim]
        emb_neg_v = self.v_embeddings(neg_v)  # [batch_size,num_negative,embedding_dim]

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)  # [batch_size,1]
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)  # [batch_size,1]
        # [batch_size,num_negative,embedding_dim] * [batch_size,embedding_dim,1]
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()  # [batch_size,num_negative]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)  # [batch_size,1]
        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w', encoding="ISO-8859-1") as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))


import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomDataset(object):
    def __init__(self, path):
        self.fn = path
        pass


pass


class Metapath2VecTrainer:
    def __init__(self, args):
        dataset = CustomDataset(args.path)
        self.data = DataReader(dataset, args.min_count, args.care_type)
        dataset = Metapath2vecDataset(self.data, args.window_size)
        self.dataloader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     collate_fn=dataset.collate
                                     )
        self.output_file_name = args.output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        # print(self.skip_gram_model.parameters())
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        for iteration in range(self.iterations):
            print('\n\n\nIteration:' + str(iteration + 1))
            print(self.skip_gram_model.parameters())
            optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print('Loss:' + str(running_loss))
            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


trainer = Metapath2VecTrainer(args)
trainer.train()

end = time.time()
print('Running time: %s Seconds'%(end))
print('Running time: %s Seconds'%(end-start))

print('test6 finished')