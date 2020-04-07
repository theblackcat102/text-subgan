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

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def generate(log_name, checkpoint_name, K_BINS=20):
    assert os.path.exists(os.path.join(log_name, 'relgan_g.py')) == True
    assert os.path.exists(os.path.join(log_name, checkpoint_name)) == True

    copyfile(os.path.join(log_name, 'relgan_g.py'), 'module/temp_g.py')
    from module.temp_g import RelSpaceG
    checkpoint = torch.load(os.path.join(log_name, checkpoint_name))
    # checkpoint should contain model state dict,
    assert len(checkpoint) == 3
    with open(os.path.join(log_name, 'params.json'), 'r') as f:
        params = json.load(f)
    args = Struct(**params)


    p = checkpoint['p']
    latent = checkpoint['latent']
    dataset = TextSubspaceDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, 
        max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, k_bins=K_BINS, token_level=args.tokenize)
    dataset.p = p
    dataset.latent = latent
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4,
                    collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True)

    model = RelSpaceG(args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, 
            args.gen_hidden_dim, dataset.vocab_size,
            k_bins=K_BINS, latent_dim=args.gen_latent_dim, noise_dim=args.gen_noise_dim,
            max_seq_len=args.max_seq_len-1, padding_idx=Constants.PAD, gpu=True)
    
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    with torch.no_grad(), open(os.path.join(log_name, 'output.jsonl'), 'w') as f, open(os.path.join(log_name, 'references.jsonl'), 'w') as rf:
        for batch in tqdm(dataloader, dynamic_ncols=True):
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            batch_size = inputs.shape[0]
            kbins_, latent = batch['bins'], batch['latents']
            kbins = F.one_hot(kbins_, K_BINS).float()
            kbins, latent = kbins.cuda(), latent.cuda()
            results = model.sample(kbins, latent, batch_size, batch_size)
            for idx, sent in enumerate(results):
                sentence = []
                for token in sent:
                    if token.item() == Constants.EOS:
                        break
                    if token.item() == Constants.PAD:
                        continue
                    sentence.append(  dataset.idx2word[token.item()])
                P = kbins_[idx].item()
                f.write(json.dumps({'p': P, 'text': ' '.join(sentence)})+'\n')

            for idx, sent_token in enumerate(batch['seq'][:, 1:]):
                reference = []
                for token in sent_token:
                    if token.item() == Constants.EOS:
                        break
                    reference.append(dataset.idx2word[token.item()] )
                P = kbins_[idx].item()
                rf.write(json.dumps({'p': P, 'text': ' '.join(reference)})+'\n')


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

