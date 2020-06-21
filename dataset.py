import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import torch
import random
from torch.utils.data import Dataset
from collections import defaultdict
from preprocess import clean_text, segment_text, pad_sequence
from constant import (
    CACHE_DIR, 
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

class KKDayUser(Dataset):
    def __init__(self, chunk_size, datapath, graph_embedding, 
        prefix, embedding, max_length, is_train=True,force_fix_len=False, token_level='word'):

        self.chunk_size = chunk_size
        self.embedding = embedding
        self.force_fix_len = force_fix_len

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
        with open(graph_embedding, 'rb') as f:
            self.graph_embedding = pickle.load(f)
        cache_data_name = 'kkday_gan_{}_corpus_v3.pkl'.format(prefix)
        if os.path.isfile(os.path.join(CACHE_DIR, cache_data_name)):
            cache = pickle.load(
                open(os.path.join(CACHE_DIR, cache_data_name), 'rb'))
            self.data = cache['data']
            self.prod2id = cache['prod2id']
            self.user2id = cache['user2id']
            self.users_pid = cache['users_pid']

        else:
            self.data = []
            self.user2id = {}
            self.prod2id = {}
            data = {}
            title_files = [ f for f in glob.glob('data/kkday_dataset/user_data/prod_title_after/*.txt') ]
            desc_files = [ f for f in glob.glob('data/kkday_dataset/user_data/prod_desc_after/*.txt') ]
            template_files = [ f for f in glob.glob('data/kkday_dataset/user_data/prod_template_after/*.txt') ]

            name2id = self.graph_embedding['name2id']

            print('total: ',len(title_files))
            print('total: ',len(desc_files))

            data = self.encode_text(title_files, data, 'title')
            data = self.encode_text(desc_files, data, 'description')
            data = self.encode_text(template_files, data, 'template')

            products, users = [], []
            users_pid = {}
            with open('data/kkday_dataset/user_data/useritem_relations.txt', 'r') as f:
                for line in tqdm(f.readlines()):
                    user_id, prod_id = line.strip().split('\t')
                    if user_id not in users_pid:
                        users_pid[user_id] = []

                    if prod_id not in users_pid[user_id]:
                        users_pid[user_id].append(prod_id)

                    if user_id in name2id and prod_id in name2id and prod_id in data and len(data[prod_id]) > 1 \
                        and len(data[prod_id]['title']) > 0:
                        products.append(prod_id)
                        users.append(user_id)
                        self.data.append({
                            'user': user_id,
                            'prod': prod_id,
                            'title': data[prod_id]['title'],
                            'template': data[prod_id]['template'],
                            'description': data[prod_id]['description']
                        })

            unique_prod = list(set(products))
            for id_ in unique_prod:
                self.prod2id[id_] = len(self.prod2id)
            unique_user = list(set(users))
            for id_ in unique_user:
                self.user2id[id_] = len(self.user2id)

            print(f'Unique {len(self.user2id)}, {len(self.prod2id)}')

            random.shuffle(self.data)

            with open(os.path.join(CACHE_DIR, cache_data_name), 'wb') as f:
                pickle.dump({'data': self.data, 'user2id': self.user2id, 
                    'prod2id': self.prod2id, 
                    'users_pid': users_pid}, f)
            self.users_pid = users_pid

        # print(len(self.data))
        flag = int(len(self.data)*0.8)
        if is_train:
            self.data = self.data[:flag]
        else:
            self.data = self.data[flag:]

        self.size = len(self.data)

        self.product_lists = list(self.prod2id)


    def __getitem__(self, item):
        row = self.data[item]
        def encode_seq(seq):
            if self.force_fix_len:
                length = self.max_length
            else:
                length = len(seq)


            if self.embedding:
                embedding_seq = np.zeros((len(seq), self.vocab_size))
                embedding_seq[np.arange(len(seq)), seq] = 1.0
                seq = embedding_seq
            return seq, length

        item_id = row['prod']

        neg_id = random.sample(self.product_lists, 1)[0]
        user_id = row['user']
        while neg_id in self.users_pid[user_id]:
            neg_id = random.sample(self.product_lists, 1)[0]

        item_embedding = self.graph_embedding['embedding'][self.graph_embedding['name2id'][item_id]]
        user_embedding = self.graph_embedding['embedding'][self.graph_embedding['name2id'][user_id]]
        neg_embedding = self.graph_embedding['embedding'][self.graph_embedding['name2id'][neg_id]]

        title_seq, title_length = encode_seq(row['title'])
        description_seq, description_length = encode_seq(row['description'])
        template_seq, template_length = encode_seq(row['template'])

        return {
            'item': item_embedding, 'user': user_embedding, 'neg': neg_embedding,
            'title_seq': title_seq, 'title_length': title_length, 
            'description_seq': description_seq, 'description_length': description_length, 
            'template_seq': template_seq, 'template_length': template_length,
            'pid': self.prod2id[item_id],
            'uid': self.user2id[user_id], 
            'nid': self.prod2id[neg_id],
        }


    def encode_text(self, filenames, data, field):
        for f_ in filenames:
            filename = os.path.basename(f_)
            item_id = filename.split('.')[0].split('_')[-1]
            if item_id not in data:
                data[item_id] = {}
            with open(f_, 'r', encoding='UTF-8') as f:
                title = f.readline().strip().replace('\u3000', ' ')
                title = self.tokenizer.split(title)

                if len(title) > 0 and len(title) < self.max_length:
                    encoded = []
                    for c in title:
                        if c in self.word2idx:
                            encoded.append(self.word2idx[c])
                        else:
                            encoded.append(self.word2idx[Constants.UNK_WORD])

                    encoded = np.asarray(encoded)
                    data[item_id][field] = encoded
        return data

    def prep_embedding(self):
        user_embeddings = np.zeros( (len(self.user2id), len(self.graph_embedding['embedding'][0]) ))
        for userid, id_ in self.user2id.items():
            user_embeddings[id_] = self.graph_embedding['embedding'][self.graph_embedding['name2id'][userid]]

        prod_embeddings = np.zeros((len(self.prod2id), len(self.graph_embedding['embedding'][0])))
        for prodid, id_ in self.prod2id.items():
            prod_embeddings[id_] = self.graph_embedding['embedding'][self.graph_embedding['name2id'][prodid]]

        return user_embeddings, prod_embeddings

    def __len__(self):
        return self.size

class TemPest(Dataset):
    def __init__(self, cache_path, data_type,
        max_length=-1, is_train=True,force_fix_len=False, token_level='word'):
        self.max_length = max_length
        assert data_type in ['train', 'valid', 'test']
        self.data_pt = torch.load(os.path.join(cache_path, data_type+'.pt'))
        self.max_length = max_length
        if token_level == 'word':
            tokenizer = WordTokenizer(postfix='tempest_word')
        else:
            tokenizer = CharTokenizer(postfix='tempest_word')
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word2idx)
        

    def encode_seq(seq):
        if self.force_fix_len:
            length = self.max_length
        else:
            length = len(seq)

        if self.embedding:
            embedding_seq = np.zeros((len(seq), self.vocab_size))
            embedding_seq[np.arange(len(seq)), seq] = 1.0
            seq = embedding_seq
        return seq, length

    def __getitem__(self, item):
        '''
            This returns 
            target, article text( source ), template, user, product pair
            ext vocab source
        '''
        data = self.data_pt[item]
        src = self.tokenizer.encode(data[0])
        tgt = self.tokenizer.encode(data[1])
        tmt = self.tokenizer.encode(data[2])
        user_prod = data[3]

        return {
            'src': src,
            'tgt': tgt,
            'tmt': tmt,
            'user_prod': user_prod,
            'src_length': len(src), 
            'tgt_length': len(tgt),             
            'tmt_length': len(tmt), 
        }


    def __len__(self):
        return len(self.data_pt)

def tempest_collate(batch, tgt_upper_max_len=40):
    src_sequences = []
    tmp_sequences = []
    tgt_sequences = []
    tgt_max_length = max(max([d['tgt_length'] for d in batch ]), tgt_upper_max_len)
    src_max_length = max([d['src_length'] for d in batch ])
    tmp_max_length = max([d['tmt_length'] for d in batch ])

    user_prods = []
    users = []
    prods = []
    for data in batch:
        if len(data['user_prod']) > 0:
            idx = random.randint(0, len(data['user_prod'])-1)
            while min(data['user_prod'][idx]) < 0:
                idx = random.randint(0, len(data['user_prod'])-1)

            prods.append(data['user_prod'][idx][0])
            users.append(data['user_prod'][idx][1])
        else:
            users.append(-1)
            prods.append(-1)
        # add eos, bos here
        tmp_sequences.append(pad_sequence(data['tmt'], tmp_max_length))
        tgt_sequences.append(pad_sequence(data['tgt'], tgt_max_length))
        src_sequences.append(pad_sequence(data['src'], src_max_length))

    # user_prods = np.concatenate(user_prods, axis=0)
    # user_prods = torch.from_numpy(user_prods).long()

    users = torch.from_numpy(np.array(users)).long()
    prods = torch.from_numpy(np.array(prods)).long()

    src_sequences = np.stack(src_sequences)
    src_sequences = torch.from_numpy(src_sequences).long()

    tgt_sequences = np.stack(tgt_sequences)
    tgt_sequences = torch.from_numpy(tgt_sequences).long()

    tmp_sequences = np.stack(tmp_sequences)
    tmp_sequences = torch.from_numpy(tmp_sequences).long()

    return {
        'users': users,
        'prods': prods,
        'src': src_sequences,
        'tgt': tgt_sequences,
        'tmt': tmp_sequences,
        # 'user_prods': user_prods,
    }

def seq_collate(batch, tgt_upper_max_len=64): # only use for word index
    src_sequences = []
    tmp_sequences = []
    tgt_sequences = []

    if 'length' in batch[0]:
        tgt_max_length = max(max([d['length'] for d in batch ], tgt_upper_max_len))
        src_max_length = max([d['length'] for d in batch ])
    else:
        tgt_max_length = max(max([d['description_length'] for d in batch ]), tgt_upper_max_len)
        src_max_length = max([d['title_length'] for d in batch ])
        tmp_max_length = max([d['template_length'] for d in batch ])

    latents, bins, users, items, item_ids, user_ids, neg_ids = [], [], [], [], [], [], []
    negs = []
    for data in batch:
        if 'bins' in data:
            bins.append(data['bins'])
            latents.append(data['latent'])
        if 'item' in data:
            items.append(data['item'])
            users.append(data['user'])
            negs.append(data['neg'])
            item_ids.append(data['pid'])
            user_ids.append(data['uid'])
            neg_ids.append(data['nid'])
        if 'seq' in data:
            src_sequences.append(pad_sequence(data['seq'], src_max_length))
        else:
            tmp_sequences.append(pad_sequence(data['template_seq'], tgt_max_length))
            tgt_sequences.append(pad_sequence(data['title_seq'], tgt_max_length))
            src_sequences.append(pad_sequence(data['description_seq'], src_max_length))

    src_sequences = np.stack(src_sequences)
    src_sequences = torch.from_numpy(src_sequences).long()


    data = {'seq': src_sequences, 'length': tgt_max_length }

    if len(tgt_sequences) > 0:
        tgt_sequences = np.stack(tgt_sequences)
        tgt_sequences = torch.from_numpy(tgt_sequences).long()
        tmp_sequences = np.stack(tmp_sequences)
        tmp_sequences = torch.from_numpy(tmp_sequences).long()

        data = {
            'tmp': tmp_sequences, 'tmp_len': tmp_max_length,
            'src': src_sequences, 'src_len': src_max_length,
            'tgt': tgt_sequences, 'tgt_len': tgt_max_length,
        }

    if len(latents) > 0:
        latents = torch.from_numpy(np.array(latents)).float()
        bins = torch.from_numpy(np.array(bins)).long()
        data['bins'] = bins
        data['latents'] = latents

    if len(items) > 0:
        # users = torch.from_numpy(np.array(users)).float()
        items = torch.from_numpy(np.array(items)).float()
        data['items'] = items
        negs = torch.from_numpy(np.array(negs)).float()
        data['negs'] = negs

        users = torch.from_numpy(np.array(users)).float()
        data['users'] = users
        item_ids = torch.from_numpy(np.array(item_ids)).long()
        data['item_ids'] = item_ids

        user_ids = torch.from_numpy(np.array(user_ids)).long()
        data['user_ids'] = user_ids
        neg_ids = torch.from_numpy(np.array(neg_ids)).long()
        data['neg_ids'] = neg_ids
        # data['users'] = users
    return data



def pad_items(items, max_items=6):
    item_dim = len(items[0])
    pad_size = max(max_items - len(items), 0)
    for _ in range(pad_size):
        items.append(np.zeros(item_dim))
    return items[:max_items]

def convert_bipartile():
    with open('data/kkday_dataset/user_data/user_records.txt', 'r') as f, open('data/kkday_dataset/user_data/useritem_relations.txt', 'w') as g:
        for line in f.readlines():
            relations = line.strip().split(',')
            user = relations[0]
            items = relations[1:]
            for item in items:
                if len(item) > 0:
                    g.write(f'{user}\t{item}\n')
            

if __name__ == "__main__":
    import torch

    dataset = KKDayUser(-1, 'data/kkday_dataset/user_data', 
        'data/kkday_dataset/matrix_factorized_64.pkl',
        prefix='item_graph', embedding=None, max_length=500, token_level='word')
    # dataset = TextDataset(-1, 'data/kkday_dataset/train_article.txt', prefix='train_article', embedding=None, max_length=256)
    # dataset = TextDataset(-1, 'data/kkday_dataset/valid_title.txt', prefix='valid_title', embedding=None, max_length=128)
    # dataset = TextDataset(-1, 'data/kkday_dataset/valid_article.txt', prefix='valid_article', embedding=None, max_length=256)
    # dataset = TextDataset(-1, 'data/kkday_dataset/test_title.txt', prefix='test_title', embedding=None, max_length=128)
    # dataset = TextDataset(-1, 'data/kkday_dataset/test_article.txt', prefix='test_article', embedding=None, max_length=256)
    # print(dataset.vocab_size)
    # print(len(dataset))
    # dataset.prep_embedding()
    # dataloader = torch.utils.data.DataLoader(dataset, 
    #     collate_fn=seq_collate, batch_size=89, num_workers=0)
    # # dataset.calculate_stats()
    # from tqdm import tqdm

    # for batch in tqdm(dataloader):
    #     batch['items'].shape
    #     # print(len(batch['src']))
    # # convert_bipartile()

    # dataset = TemPest('dataset/', 'train')
    # print(len(dataset))

    dataset = TemPest('dataset/', 'valid')
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, 
        collate_fn=tempest_collate, batch_size=32, num_workers=0)

    for batch in tqdm(dataloader):
        print(batch['users'].shape)
        users = batch['users']
        empty_users = users == -1
        batch_size = len(empty_users)
        users_fill = users.detach()

        users_fill[empty_users] = torch.randint(0, 1000, (int(empty_users.sum()), ))
