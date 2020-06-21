import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.vmt import VMT, TemplateD, QHead, ProductHead
from module.vae import GaussianKLLoss
from dataset import TemPest, tempest_collate
from constant import Constants
import fasttext
from utils import get_fixed_temperature, get_losses
import sklearn
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from shutil import copyfile
import pickle
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

def build_user2word_graph(cache_path='dataset'):
    train_dataset = TemPest(cache_path, 'train')
    id_mapping = torch.load(os.path.join(cache_path, 'id_mapping.pt'))
    user_size = len(id_mapping['user2id'])
    vocab_size = train_dataset.vocab_size

    mapping = {}
    max_val = 0
    for data in tqdm(train_dataset, dynamic_ncols=True):
        tgt_tokens = data['tgt']
        user2prod_relations = data['user_prod']
        for (prod, user) in user2prod_relations:
            if user not in mapping:
                mapping[user] = {}
            for token in tgt_tokens:
                if token not in mapping[user]:
                    mapping[user][token] = 0
                mapping[user][token] += 1
                if mapping[user][token] > max_val:
                    max_val = mapping[user][token]
    del train_dataset
    print('most frequent words, ', max_val)
    rows = []
    cols = []
    data = []
    for user, values in mapping.items():
        for token_id, count in mapping[user].items():
            rows.append(user)
            cols.append(token_id)
            data.append(count/max_val) # normalize

    print('start running NMF..')
    sparse_X = csr_matrix((data, (rows, cols)), shape=(user_size, vocab_size))
    model = NMF(n_components=32, init='nndsvd', random_state=0)
    user_embeddings = model.fit_transform(sparse_X)
    word_embeddings = model.components_
    with open('cache/nmf_user_embedding.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)
    with open('cache/nmf_word_embedding.pkl', 'wb') as f:
        pickle.dump(word_embeddings, f)

def build_user2prod_graph(cache_path='dataset'):
    id_mapping = torch.load(os.path.join(cache_path, 'id_mapping.pt'))
    user2id = id_mapping['user2id']
    prod2id = id_mapping['prod2id']
    mapping = {}
    max_val = 0

    with open(os.path.join(cache_path+'/user_data/', 'user_records.txt'), 'r') as f, open('found_title.txt', 'w') as g:
        for line in f.readlines():
            user_id, products = line.strip().split(',', maxsplit=1)
            user_id = user2id[user_id]
            for prod_id in products.split(','):
                if len(prod_id) == 0:
                    continue
                prod_id = prod2id[prod_id]
                if user_id not in mapping:
                    mapping[user_id] = {}

                if prod_id not in mapping[user_id]:
                    mapping[user_id][prod_id] = 0

                mapping[user_id][prod_id] += 1

                if mapping[user_id][prod_id] > max_val:
                    max_val = mapping[user_id][prod_id]

    rows = []
    cols = []
    data = []
    for user, values in mapping.items():
        for token_id, count in mapping[user].items():
            rows.append(user)
            cols.append(token_id)
            data.append(count/max_val) # normalize
    print('start running NMF..')
    print('user size ', len(user2id), ' prod size ', len(prod2id))
    sparse_X = csr_matrix((data, (rows, cols)), shape=(len(user2id), len(prod2id)))
    model = NMF(n_components=32, init='nndsvd', random_state=0)
    user_embeddings = model.fit_transform(sparse_X)
    prod_embeddings = model.components_
    with open('cache/nmf_user_embedding2.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)
    print(user_embeddings.shape)

    with open('cache/nmf_prod_embedding2.pkl', 'wb') as f:
        pickle.dump(prod_embeddings, f)

    print(prod_embeddings.shape)

if __name__ == "__main__":
    build_user2prod_graph()
    # build_user2word_graph()