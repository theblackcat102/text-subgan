import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from preprocess import clean_text, segment_text, pad_sequence
from constant import (
    CACHE_DIR, DCARD_DATA, PTT_DATA, DCARD_WHITE_LIST, PTT_WHITE_LIST,
    MAX_LENGTH, Constants)
from tokenizer import CharTokenizer, WordTokenizer

class TextDataset(Dataset):
    def __init__(self, chunk_size, filename, prefix, embedding, max_length, force_fix_len=False, token_level='word'):
        self.chunk_size = chunk_size
        self.embedding = embedding
        self.force_fix_len = force_fix_len
        assert token_level in ['word', 'char']
        if token_level == 'word':
            tokenizer = WordTokenizer()
        else:
            tokenizer = CharTokenizer()
        prefix = token_level+'_'+prefix

        self.idx2word = tokenizer.idx2word
        self.word2idx = tokenizer.word2idx
        self.vocab_size = len(self.idx2word)
        self.tokenizer = tokenizer

        cache_data_name = 'cache_{}_data.pkl'.format(prefix)
        if os.path.isfile(os.path.join(CACHE_DIR, cache_data_name)):
            self.data = pickle.load(
                open(os.path.join(CACHE_DIR, cache_data_name), 'rb'))
        else:
            self.data = []
            with open(filename, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    title = line.strip()
                    title = tokenizer.split(title)
                    if len(title) > 0 and len(title) < max_length:
                        encoded = []
                        for c in title:
                            if c in self.word2idx:
                                encoded.append(self.word2idx[c])
                            else:
                                encoded.append(self.word2idx[Constants.UNK_WORD])
                        encoded = np.asarray(encoded)
                        # print(title)
                        # print(encoded)
                        self.data.append(encoded)
            self.data = np.array(self.data)
            pickle.dump(
                self.data,
                open(os.path.join(CACHE_DIR, cache_data_name), 'wb'))

        self.size = len(self.data)
        self.max_length = max_length

    def __getitem__(self, item):
        seq = self.data[item]
        if self.force_fix_len:
            length = self.max_length
        else:
            length = len(seq)

        if self.chunk_size > 0:
            seq = pad_sequence(seq, self.chunk_size)
            length = min(length + 2, self.chunk_size)

        if self.embedding:
            embedding_seq = np.zeros((len(seq), self.vocab_size))
            embedding_seq[np.arange(len(seq)), seq] = 1.0
            seq = embedding_seq

        return {'seq': seq, 'length': length}

    def __len__(self):
        return self.size


class TextSubspaceDataset(Dataset):
    def __init__(self, chunk_size, filename, prefix, embedding, max_length, k_bins=5,force_fix_len=False, token_level='word'):
        self.chunk_size = chunk_size
        self.embedding = embedding
        self.force_fix_len = force_fix_len
        self.k_bins = k_bins
        self.max_length = max_length
        if token_level == 'word':
            tokenizer = WordTokenizer()
        else:
            tokenizer = CharTokenizer()
        prefix = token_level+'_'+prefix

        self.idx2word = tokenizer.idx2word
        self.word2idx = tokenizer.word2idx
        self.vocab_size = len(self.idx2word)
        self.tokenizer = tokenizer

        initial_cluster = 'initial_cluster_{}.pkl'.format(k_bins)
        if os.path.exists(initial_cluster):
            cluster = pickle.load(open(initial_cluster, 'rb'))
            # self.k_bins = cluster['p']
            # self.latent = cluster['latent']
        else:
            raise ValueError("Intialize cluster first! ")
        cache_data_name = '{}_bins_cache_{}_data.pkl'.format(k_bins, prefix)
        if os.path.isfile(os.path.join(CACHE_DIR, cache_data_name)):
            cache = pickle.load(
                open(os.path.join(CACHE_DIR, cache_data_name), 'rb'))
            self.data = cache['data']
            self.p = cache['p']
            self.latent = cache['latent']
        else:
            self.data = []
            self.p = []
            self.latent = []
            with open(filename, 'r', encoding='UTF-8') as f:
                for idx, line in enumerate(f.readlines()):
                    title_ = line.strip()
                    title = tokenizer.split(title_)
                    if len(title) > 0 and len(title) < max_length:
                        encoded = []
                        self.p.append(cluster['p'][title_])
                        self.latent.append(cluster['latent'][title_])
                        for c in title:
                            if c in self.word2idx:
                                encoded.append(self.word2idx[c])
                            else:
                                encoded.append(self.word2idx[Constants.UNK_WORD])
                        encoded = np.asarray(encoded)
                        # print(title)
                        # print(encoded)
                        self.data.append(encoded)
            self.data = np.array(self.data)
            self.p = np.array(self.p)
            self.latent = np.array(self.latent)
            assert len(self.p) == len(self.data)

            pickle.dump(
                {'data': self.data,'p': self.p, 'latent': self.latent },
                open(os.path.join(CACHE_DIR, cache_data_name), 'wb'))

        self.size = len(self.data)

    def calculate_stats(self):
        stats = defaultdict(int)
        for p in self.p:
            stats[p] += 1
        weights = [0] * self.k_bins
        for key, value in stats.items():
            weights[key] = value
        return torch.from_numpy(np.array(weights))

    def __getitem__(self, item):
        seq = self.data[item]

        if self.force_fix_len:
            length = self.max_length
        else:
            length = len(seq)

        if self.chunk_size > 0:
            seq = pad_sequence(seq, self.chunk_size)
            length = min(length + 2, self.chunk_size)

        if self.embedding:
            embedding_seq = np.zeros((len(seq), self.vocab_size))
            embedding_seq[np.arange(len(seq)), seq] = 1.0
            seq = embedding_seq

        return {'seq': seq, 'length': length, 'latent': self.latent[item], 'bins': self.p[item] }

    def __len__(self):
        return self.size


def seq_collate(batch): # only use for word index
    sequences = []
    max_length = max([d['length'] for d in batch ])
    latents, bins = [], []

    for data in batch:
        if 'bins' in data:
            bins.append(data['bins'])
            latents.append(data['latent'])

        sequences.append(pad_sequence(data['seq'], max_length))
    sequences = np.stack(sequences)
    sequences = torch.from_numpy(sequences).long()
    data = {'seq': sequences, 'length': max_length }
    if len(latents) > 0:
        latents = torch.from_numpy(np.array(latents)).float()
        bins = torch.from_numpy(np.array(bins)).long()
        data['bins'] = bins
        data['latents'] = latents
    return data

if __name__ == "__main__":
    import torch

    dataset = TextSubspaceDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, max_length=50, token_level='char')
    # dataset = TextDataset(-1, 'data/kkday_dataset/train_article.txt', prefix='train_article', embedding=None, max_length=256)
    # dataset = TextDataset(-1, 'data/kkday_dataset/valid_title.txt', prefix='valid_title', embedding=None, max_length=128)
    # dataset = TextDataset(-1, 'data/kkday_dataset/valid_article.txt', prefix='valid_article', embedding=None, max_length=256)
    # dataset = TextDataset(-1, 'data/kkday_dataset/test_title.txt', prefix='test_title', embedding=None, max_length=128)
    # dataset = TextDataset(-1, 'data/kkday_dataset/test_article.txt', prefix='test_article', embedding=None, max_length=256)
    print(dataset.vocab_size)
    dataloader = torch.utils.data.DataLoader(dataset, 
        collate_fn=seq_collate, batch_size=64)
    dataset.calculate_stats()
    from tqdm import tqdm
    for batch in tqdm(dataloader):
        batch['seq'].shape
        batch['latents'].shape