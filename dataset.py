import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

from preprocess import clean_text, segment_text, pad_sequence
from constant import (
    CACHE_DIR, DCARD_DATA, PTT_DATA, DCARD_WHITE_LIST, PTT_WHITE_LIST,
    MAX_LENGTH)


class TextDataset(Dataset):
    def __init__(self, chunk_size, filepath, prefix, embedding, style,
                 white_list, max_length):
        self.chunk_size = chunk_size
        self.embedding = embedding
        self.style = style
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
            filenames = glob.glob(filepath)
            if white_list is not None:
                filenames = list(filter(
                    lambda f: os.path.basename(f) in white_list, filenames))
            for filename in tqdm(filenames, desc='load %s' % prefix):
                with open(filename, 'r', encoding='UTF-8') as f:
                    for line in f.readlines():
                        title = clean_text(line.strip())
                        title = segment_text(title)
                        if len(title) > 0 and len(title) < max_length:
                            encoded = [self.word2idx[c] for c in title]
                            encoded = np.asarray(encoded)
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

        return {'seq': seq, 'length': length, 'style': self.style}

    def __len__(self):
        return self.size


class PttDataset(TextDataset):
    def __init__(self, chunk_size=MAX_LENGTH, max_length=MAX_LENGTH,
                 filepath=PTT_DATA, embedding=False):
        """
            chunk_size: sequence maximum size, if -1 then the sequence size is
                not changed
            embedding: whether to return one hot embedding value
        """
        super().__init__(chunk_size, filepath, 'ptt', embedding, style=0,
                         white_list=PTT_WHITE_LIST, max_length=max_length)


class DcardDataset(TextDataset):
    def __init__(self, chunk_size=MAX_LENGTH, max_length=MAX_LENGTH,
                 filepath=DCARD_DATA, embedding=False):
        """
            chunk_size: sequence maximum size, if -1 then the sequence size is
                not changed
            embedding: whether to return one hot embedding value
        """
        super().__init__(chunk_size, filepath, 'dcard', embedding, style=1,
                         white_list=DCARD_WHITE_LIST, max_length=max_length)


class AllDataset(Dataset):
    def __init__(self, chunk_size=-1, embedding=False):
        self.ptt_dataset = PttDataset(
            chunk_size=chunk_size, embedding=embedding)
        self.dcard_dataset = DcardDataset(
            chunk_size=chunk_size, embedding=embedding)

        self.chunk_size = chunk_size
        self.embedding = embedding
        self.idx2word = self.ptt_dataset.idx2word
        self.word2idx = self.ptt_dataset.word2idx
        self.vocab_size = self.ptt_dataset.vocab_size

    def __getitem__(self, item):
        if item < len(self.ptt_dataset):
            data = self.ptt_dataset[item]
        else:
            item -= len(self.ptt_dataset)
            data = self.dcard_dataset[item]
        return data

    def __len__(self):
        return len(self.ptt_dataset) + len(self.dcard_dataset)


if __name__ == "__main__":

    corpus = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            corpus.append(line.strip())

    dataset = AllDataset()
    for i in range(5):
        idx = np.random.randint(len(dataset))
        print(dataset[idx]['seq'], dataset[idx]['length'])
        for idx in dataset[idx]['seq']:
            print(dataset.idx2word[idx], end=' ')
        print()