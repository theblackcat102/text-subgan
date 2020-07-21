import glob, os, pickle
from collections import defaultdict
import re
import numpy as np

from constant import Constants, EMOJI_PATTERN, CACHE_DIR


class WordTokenizer():


    def __init__(self, corpus=None, cache_path='cache', postfix='word'):
        word2idx_f = os.path.join(cache_path, postfix+'_word2idx.pkl')
        idx2word_f = os.path.join(cache_path, postfix+'_idx2word.pkl')
        self.corpus_cache = os.path.join(cache_path, postfix+'_corpus.pkl')
        if os.path.exists(word2idx_f) or corpus == None:
            
            with open(word2idx_f, 'rb') as f:
                word2idx = pickle.load(f)
            with open(idx2word_f, 'rb') as f:
                idx2word = pickle.load(f)

        else:
            word2freq = defaultdict(int)
            for line in corpus:
                tokens = line.strip().split(' ')
                for t in tokens:
                    word2freq[t] += 1

            word2idx = Constants.word2idx()
            # print(len(word2freq))
            for key, freq in word2freq.items():
                if key not in word2idx:
                    word2idx[key] = len(word2idx)

            idx2word = {}
            for token, idx in word2idx.items():
                idx2word[idx] = token

            with open(word2idx_f, 'wb') as f:
                pickle.dump(word2idx, f)
            with open(idx2word_f, 'wb') as f:
                pickle.dump(idx2word, f)

        self.word2idx = word2idx
        self.idx2word = idx2word

    def tokenize(self, sentence):
        output = []
        for t in sentence.split(' '):
            if t in self.word2idx:
                output.append(self.word2idx[t])
            else:
                output.append(self.word2idx['</UNK>'])
        return output

    def split(self, sentence):
        return sentence.split(' ')

    def encode(self, text):
        def encode_( sent):
            text = self.split(sent)
            encoded = []
            for c in text:
                if c in self.word2idx:
                    # print(c, self.word2idx[c], self.idx2word[self.word2idx[c]] )
                    encoded.append(self.word2idx[c])
                else:
                    encoded.append(self.word2idx[Constants.UNK_WORD])
            encoded = np.asarray(encoded)
            return encoded

        if isinstance(text, str):
            return encode_(text)
        else:
            sentences = []
            for sent in text:
                sentences.append(encode_(sent))
            return np.array(sentences)


class CharTokenizer():


    def __init__(self, corpus=None,cache_path='cache', postfix='char'):
        word2idx_f = os.path.join(cache_path, postfix+'_word2idx.pkl')
        idx2word_f = os.path.join(cache_path, postfix+'_idx2word.pkl')
        self.corpus_cache = os.path.join(cache_path, postfix+'_corpus.pkl')
        if os.path.exists(word2idx_f) or corpus == None:
            
            with open(word2idx_f, 'rb') as f:
                word2idx = pickle.load(f)
            with open(idx2word_f, 'rb') as f:
                idx2word = pickle.load(f)

        else:
            word2freq = defaultdict(int)
            for line in corpus:
                tokens = list(line.strip())
                for t in tokens:
                    word2freq[t] += 1

            word2idx = Constants.word2idx()
            # print(len(word2freq))
            for key, freq in word2freq.items():
                word2idx[key] = len(word2idx)
            idx2word = Constants.idx2word()
            for token, idx in word2idx.items():
                idx2word[idx] = token

            with open(word2idx_f, 'wb') as f:
                pickle.dump(word2idx, f)
            with open(idx2word_f, 'wb') as f:
                pickle.dump(idx2word, f)

        self.word2idx = word2idx
        self.idx2word = idx2word

    def split(self, sentence):
        return list(sentence.replace(' ', ''))


    def encode(self, text):
        def encode_(self, sent):
            text = self.split(sent)
            encoded = []
            for c in text:
                if c in self.word2idx:
                    encoded.append(self.word2idx[c])
                else:
                    encoded.append(self.word2idx[Constants.UNK_WORD])
            encoded = np.asarray(encoded)
            return encoded

        if isinstance(text, str):
            return encode_(text)
        else:
            sentences = []
            for sent in text:
                sentences.append(encode_(sent))
            return np.array(sentences)


if __name__ == "__main__":
    tokenizer = WordTokenizer(postfix='tempest_word')
    tokenizer.word2idx['##']
    print(tokenizer.encode('【 首 ## 人氣 美食 ## 】 ## 排骨 烤肉 ##'))