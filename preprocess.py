import re
import numpy as np

from constant import Constants, EMOJI_PATTERN, CACHE_DIR


def handle_emojis(tweet):
    '''
        手動清楚一些emoji??
    '''
    post_emo = '_{}_'.format(Constants.SIMLE_WORD)
    sad_emo = '_{}_'.format(Constants.SAD_WORD)
    xd_emo = '_{}_'.format(Constants.XD_WORD)
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|\(:\s?D|:-D)',
                   post_emo, tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(x-?D|X-?D|XD)',  xd_emo, tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)',  post_emo, tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', sad_emo, tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', sad_emo, tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()',  sad_emo, tweet)
    return tweet


def clean_emoji(texts):
    return EMOJI_PATTERN.sub(r'', texts)


def clean_text(texts):
    texts = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '_{}_'.format(
        Constants.URL_WORD), texts)
    texts = handle_emojis(texts)
    return texts


def segment_text(texts):
    first_pass = texts.split('_')
    second_pass = []
    for section in first_pass:
        if len(section) <= 0:
            continue
        if len(section) > 8:
            second_pass += list(section)
        elif section[-1] == '>' and section[:2] == '</':
            second_pass.append(section)
        else:
            second_pass += list(section)
    return second_pass


def pad_sequence(encoded, chunk_size):
    encoded = encoded[:chunk_size - 2]
    encoded = np.concatenate(([Constants.BOS], encoded, [Constants.EOS]))
    encoded = np.pad(
        encoded,
        pad_width=[0, chunk_size - len(encoded)],
        mode='constant',
        constant_values=Constants.PAD)

    return encoded


def sort_by_length(seq, length, label):
    length, indices = length.sort(descending=True)
    seq = seq[indices]
    label = label[indices]
    return seq, length, label


if __name__ == "__main__":
    import glob, os, pickle
    from collections import defaultdict
    from tokenizer import CharTokenizer, WordTokenizer
    word2freq = defaultdict(int)
    training_filename = 'data/kkday_dataset/train_title.txt'
    corpus = []
    with open(training_filename, 'r') as f:
        for line in f.readlines():
            corpus.append(line.strip())
    tokenizer = CharTokenizer(corpus)
    tokenizer = WordTokenizer(corpus)


    # training_filename = 'data/kkday_dataset/train_article.txt'
    # with open(training_filename, 'r') as f:
    #     for line in f.readlines():
    #         tokens = line.strip().split(' ')
    #         for t in tokens:
    #             word2freq[t] += 1

    # word2idx = Constants.word2idx()
    # # print(len(word2freq))
    # for key, freq in word2freq.items():
    #     word2idx[key] = len(word2idx)
    # idx2word = Constants.idx2word()
    # for token, idx in word2idx.items():
    #     idx2word[idx] = token

    # with open(os.path.join(CACHE_DIR, "word2idx.pkl"), 'wb') as f:
    #     pickle.dump(word2idx, f)
    # with open(os.path.join(CACHE_DIR, "idx2word.pkl"), 'wb') as f:
    #     pickle.dump(idx2word, f)