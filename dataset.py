import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from preprocess import clean_text, segment_text, pad_sequence
from constant import (
    CACHE_DIR, DCARD_DATA, PTT_DATA, DCARD_WHITE_LIST, PTT_WHITE_LIST,
    MAX_LENGTH, Constants)


class TextDataset(Dataset):
    def __init__(self, chunk_size, filename, prefix, embedding, max_length):
        self.chunk_size = chunk_size
        self.embedding = embedding
        self.idx2word = pickle.load(
            open(os.path.join(CACHE_DIR, "idx2word.pkl"), 'rb'))
        self.word2idx = pickle.load(
            open(os.path.join(CACHE_DIR, "word2idx.pkl"), 'rb'))
        self.vocab_size = len(self.idx2word)

        cache_data_name = 'cache_{}_data.pkl'.format(prefix)
        if os.path.isfile(os.path.join(CACHE_DIR, cache_data_name)):
            self.data = pickle.load(
                open(os.path.join(CACHE_DIR, cache_data_name), 'rb'))
        else:
            self.data = []
            with open(filename, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    title = line.strip()
                    title = title.split(' ')
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

    def __getitem__(self, item):
        seq = self.data[item]
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


def seq_collate(batch): # only use for word index
    sequences = []
    max_length = max([d['length'] for d in batch ])

    for data in batch:
        sequences.append(pad_sequence(data['seq'], max_length))
    sequences = np.stack(sequences)
    sequences = torch.from_numpy(sequences).long()
    return {'seq': sequences, 'length': max_length }

if __name__ == "__main__":
    import torch

    dataset = TextDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, max_length=128)
    dataset = TextDataset(-1, 'data/kkday_dataset/train_article.txt', prefix='train_article', embedding=None, max_length=256)
    dataset = TextDataset(-1, 'data/kkday_dataset/valid_title.txt', prefix='valid_title', embedding=None, max_length=128)
    dataset = TextDataset(-1, 'data/kkday_dataset/valid_article.txt', prefix='valid_article', embedding=None, max_length=256)
    dataset = TextDataset(-1, 'data/kkday_dataset/test_title.txt', prefix='test_title', embedding=None, max_length=128)
    dataset = TextDataset(-1, 'data/kkday_dataset/test_article.txt', prefix='test_article', embedding=None, max_length=256)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
        collate_fn=seq_collate, batch_size=64)
    for batch in dataloader:
        print(batch['seq'].shape)