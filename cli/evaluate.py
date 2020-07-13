import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.relgan_d import RelSpaceGAN_D
from module.cluster import Cluster
from module.relgan_g import RelSpaceG
from dataset import TextSubspaceDataset, seq_collate
from constant import Constants
from utils import get_fixed_temperature, get_losses
from sklearn.cluster import KMeans
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks
from sklearn.manifold import SpectralEmbedding
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from shutil import copyfile
from collections import namedtuple
from dataset import TemPest, tempest_collate

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def obtain_results(log_name, checkpoint):
    log_name = 'save/tempest_rec-mf-final-2020-07-08-12-42-37/'
    from module.vmt import VMT
    with open(os.path.join(log_name, 'params.json'), 'r') as f:
        params = json.load(f)
    args = Struct(**params)
    id_mapping = torch.load(os.path.join(args.cache_path, 'id_mapping.pt'))
    user_size = len(id_mapping['user2id'])
    user_size = user_size
    prod_size = len(id_mapping['prod2id'])

    model = VMT(args.gen_embed_dim, args.vocab_size,
                enc_hidden_size=500, dec_hidden_size=500, tmp_category=args.tmp_cat_dim,
                tmp_latent_dim=args.tmp_latent_dim, desc_latent_dim=args.desc_latent_dim, user_latent_dim=args.user_latent_dim,
                biset=args.biset, user_embedding=True, user_size=user_size,
                max_seq_len=args.max_seq_len-1, gpu=True).cuda()
    model.load_state_dict(torch.load(log_name+'checkpoint_40000.pt')['model'])
    
    id2user = { value: key for key, value in id_mapping['user2id'].items() }
    valid_dataset = TemPest(args.cache_path, 'train')
    eval_dataloader = torch.utils.data.DataLoader(valid_dataset, num_workers=8,
                        collate_fn=tempest_collate, batch_size=20, shuffle=False, drop_last=True)

    sentences, references = [], []
    temp_latent = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            src_inputs = batch['src']
            tmp = batch['tmt']
            inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]
            users = batch['users']
            empty_users = users == -1
            batch_size = len(empty_users)
            users_filled = users.detach()
            users_filled[empty_users] = torch.randint(0, user_size, ( int(empty_users.sum()),  ))

            if cfg.CUDA:
                src_inputs = src_inputs.cuda()
                target = target.cuda()
                tmp = tmp.cuda()
                inputs = inputs.cuda()
                users_filled = users_filled.cuda()

            desc_outputs, desc_latent, desc_mean, desc_std = model.encode_desc(src_inputs)
            tmp_outputs, tmp_latent = model.encode_tmp(tmp)

            for latent_ in tmp_latent.cpu():
                temp_latent.append(latent_.numpy().flatten())

            user_embeddings = model.user_embedding( users_filled )
            _, output_title = model.decode(tmp_latent, desc_latent, user_embeddings, 
                    desc_outputs, tmp_outputs,
                    max_length=target.shape[1])

            for idx, sent_token in enumerate(batch['tgt'][:, 1:]):
                reference = []
                for token in sent_token:
                    if token.item() == Constants.EOS:
                        break
                    reference.append(valid_dataset.tokenizer.idx2word[token.item()] )
                references.append(''.join(reference))

                sent = output_title[idx]
                sentence = []
                for token in sent:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(valid_dataset.tokenizer.idx2word[token.item()])
                sentences.append(''.join(sentence))
    return obtain_results

if __name__ == "__main__":
    import argparse
    # args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, args.gen_hidden_dim
    # args.dis_embed_dim, args.max_seq_len, args.num_rep
    # args.gen_lr args.gen_adv_lr, args.dis_lr
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log-name', type=str)
    parser.add_argument('--iter', type=int)
    args = parser.parse_args()

    checkpoint_name = 'relgan_G_{}.pt'.format(args.iter)
    generate(args.log_name, checkpoint_name)

