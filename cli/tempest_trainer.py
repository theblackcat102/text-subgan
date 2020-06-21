import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.vmt import VMT
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


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()


class TemplateTrainer():

    def __init__(self, args):
        self.args = args

        self.train_dataset = TemPest(args.cache_path, 'train')
        self.valid_dataset = TemPest(args.cache_path, 'valid')
        self.id_mapping = torch.load(os.path.join(args.cache_path, 'id_mapping.pt'))
        user_size = len(self.id_mapping['user2id'])
        self.user_size = user_size
        self.prod_size = len(self.id_mapping['prod2id'])

        self.args.vocab_size = self.train_dataset.vocab_size

        self.model = VMT(args.gen_embed_dim, self.train_dataset.vocab_size,
            enc_hidden_size=128, dec_hidden_size=128, tmp_category=args.tmp_cat_dim,
            tmp_latent_dim=args.tmp_latent_dim, desc_latent_dim=args.desc_latent_dim, user_latent_dim=args.user_latent_dim,
            biset=args.biset, user_embedding=True, user_size=user_size,
            max_seq_len=args.max_seq_len-1, gpu=True).cuda()

        self.train_iter = data_iter(torch.utils.data.DataLoader(self.train_dataset, num_workers=3,
                        collate_fn=tempest_collate, batch_size=args.batch_size, shuffle=True, 
                        drop_last=True))
        # self.valid_iter = data_iter(torch.utils.data.DataLoader(self.valid_dataset, num_workers=3,
        #                 collate_fn=tempest_collate, batch_size=args.batch_size, shuffle=False, 
        #                 drop_last=True))
        self.prod_embeddings = nn.Embedding(self.prod_size+1, args.user_latent_dim).cuda()

        self.gen_opt = optim.Adam(self.model.parameters(), lr=args.gen_lr)
        self.tmp_opt = optim.Adam(self.model.template_vae.parameters(), lr=args.gen_lr)
        self.rec_opt = optim.Adam(list(self.model.user_embedding.parameters()) + list(self.prod_embeddings.parameters()), 
            args.rec_lr, weight_decay=1e-5  )

        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.KL_loss = GaussianKLLoss()
        self.mse_criterion = nn.MSELoss()
        self.xent_criterion = nn.CrossEntropyLoss()

        self.gumbel_temp = 1.0
        N = args.iterations
        self.gumbel_anneal_rate = 1 / N
        self.temp_anneal = 1.0 / N
        self.temp = args.temperature_min 

    def init_sample_inputs(self):
        batch = next(self.train_iter)
        src_inputs = batch['src']
        tmp = batch['tmt']
        inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]
        users = batch['users']
        empty_users = users == -1
        batch_size = len(empty_users)
        users_filled = users.detach()
        users_filled[empty_users] = torch.randint(0, self.user_size, ( int(empty_users.sum()) ))


    def calculate_bleu(self, writer, step=0, size=1000, ngram=4):
        eval_dataloader = torch.utils.data.DataLoader(self.valid_dataset, num_workers=8,
                        collate_fn=tempest_collate, batch_size=20, shuffle=False)
        sentences, references = [], []
        scores_weights = { str(gram): [1/gram] * gram for gram in range(1, ngram+1)  }
        scores = { str(gram): 0 for gram in range(1, ngram+1)  }

        # print('Evaluate bleu scores', scores)
        with torch.no_grad():
            for batch in eval_dataloader:
                src_inputs = batch['src']
                tmp = batch['tmt']
                inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]
                users = batch['users']
                empty_users = users == -1
                batch_size = len(empty_users)
                users_filled = users.detach()
                users_filled[empty_users] = torch.randint(0, self.user_size, ( int(empty_users.sum()),  ))

                if cfg.CUDA:
                    src_inputs = src_inputs.cuda()
                    target = target.cuda()
                    tmp = tmp.cuda()
                    inputs = inputs.cuda()
                    users_filled = users_filled.cuda()

                desc_outputs, desc_latent, desc_mean, desc_std = self.model.encode_desc(src_inputs)
                tmp_outputs, tmp_latent = self.model.encode_tmp(tmp)

                user_embeddings = self.model.user_embedding( users_filled )
                _, output_title = self.model.decode(tmp_latent, desc_latent, user_embeddings, 
                        desc_outputs, tmp_outputs,
                        max_length=target.shape[1])                
                # output_title = torch.argmax(output_logits, dim=-1)
                for idx, sent_token in enumerate(batch['tgt'][:, 1:]):
                    reference = []
                    for token in sent_token:
                        if token.item() == Constants.EOS:
                            break
                        reference.append(self.valid_dataset.tokenizer.idx2word[token.item()] )
                    references.append(reference)

                    sent = output_title[idx]
                    sentence = []
                    for token in sent:
                        if token.item() == Constants.EOS:
                            break
                        sentence.append(  self.valid_dataset.tokenizer.idx2word[token.item()])
                    sentences.append(sentence)
                    for key, weights in scores_weights.items():
                        scores[key] += sentence_bleu([reference], sentence, weights, 
                            smoothing_function=SmoothingFunction().method1)

                if len(sentences) > size:
                    break

        with open(os.path.join(self.save_path, '{}_reference.txt'.format(0)), 'w') as f:
            for sent in references:
                f.write(' '.join(sent)+'\n')

        with open(os.path.join(self.save_path, '{}_generate.txt'.format(step)), 'w') as f:
            for sent in sentences:
                f.write(' '.join(sent)+'\n')

        if writer != None:
            for key, weights in scores.items():
                scores[key] /= len(sentences)
                writer.add_scalar("Bleu/score-"+key, scores[key], step)
            writer.flush()

    def test(self):
        self.step(0)

    def step(self, i):
        batch = next(self.train_iter)
        src_inputs = batch['src']
        tmp = batch['tmt']
        inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]
        users = batch['users']
        empty_users = users == -1
        batch_size = len(empty_users)
        users_filled = users.detach()
        users_filled[empty_users] = torch.randint(0, self.user_size, ( int(empty_users.sum()),  ))

        if cfg.CUDA:
            src_inputs = src_inputs.cuda()
            target = target.cuda()
            tmp = tmp.cuda()
            inputs = inputs.cuda()
            users_filled = users_filled.cuda()

        desc_outputs, desc_latent, desc_mean, desc_std = self.model.encode_desc(src_inputs)
        tmp_outputs, tmp_latent = self.model.encode_tmp(tmp)

        user_embeddings = self.model.user_embedding( users_filled )
        output_target, output_logits = self.model.decode(tmp_latent, desc_latent, user_embeddings, 
                desc_outputs, tmp_outputs,
                max_length=target.shape[1])

        self.temp = self.args.kl_weight
        self.gumbel_temp = np.minimum(self.args.gumbel_max - (self.gumbel_temp * np.exp(-self.gumbel_anneal_rate * i)), 0.00005)


        nll_loss, kl_loss = self.model.cycle_template(tmp[:, :-1], tmp[:, 1:], temperature=self.gumbel_temp)
        self.tmp_opt.zero_grad()

        nll_loss.backward(retain_graph=True)
        kl_loss.backward(retain_graph=True)

        self.tmp_opt.step()

        self.gen_opt.zero_grad()

        desc_kl_loss = self.KL_loss(desc_mean, desc_std)
        
        construct_loss = self.mle_criterion(output_target.view(-1, self.args.vocab_size), target.flatten())

        total_loss = construct_loss + desc_kl_loss

        total_loss.backward()
        self.gen_opt.step()

        non_empty_users = (batch['users'] != -1) & (batch['prods'] != -1)
        mf_loss = 0
        if non_empty_users.sum() > 0:
            prods = batch['prods'][non_empty_users]
            users = batch['users'][non_empty_users]
            users, prods = users.cuda(), prods.cuda()

            # print(users.max(), prods.max(), prods.min(), users.min())

            user_embed = self.model.user_embedding( users )

            prod_embed  = self.prod_embeddings( prods )

            prediction = (user_embed*prod_embed).sum(1)

            rating = torch.ones(prediction.shape).float().cuda()
            
            self.rec_opt.zero_grad()
            mf_loss = self.mse_criterion(prediction, rating)
            mf_loss.backward()
            self.rec_opt.step()

            mf_loss = mf_loss.item()
        return total_loss.item(), construct_loss.item(), desc_kl_loss.item(), nll_loss.item(), kl_loss.item(), mf_loss

    def train(self):
        if self.args.pretrain_embeddings != None:
            model = fasttext.load_model(self.args.pretrain_embeddings)
            embedding_weight = self.model.embedding.cpu().weight.data
            hit = 0
            for word, idx in self.dataset1.word2idx.items():
                embedding_weight[idx] = torch.from_numpy(model[word]).float()
                hit += 1
            embedding_weight = embedding_weight.cuda()
            self.model.embedding.weight.data.copy_(embedding_weight)
            self.model.embedding.cuda()
            self.model.title_decoder.decoder.outputs2vocab.weight.data.copy_(self.model.embedding.weight.data)

        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = 'save/tempest_{}-{}'.format(self.args.name, cur_time)
        os.makedirs(save_path, exist_ok=True)
        copyfile('module/vmt.py', os.path.join(save_path, 'vmt.py'))
        copyfile('module/vae.py', os.path.join(save_path, 'vae.py'))
        self.save_path = save_path
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/tempest_{}-{}'.format(self.args.name, cur_time))


        i = 0
        self.temp = self.args.kl_weight
        self.gumbel_temp = np.minimum(self.args.gumbel_max - (self.gumbel_temp * np.exp(-self.gumbel_anneal_rate * i)), 0.00005)
        prev_mf_loss = 0
        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                self.model.train()
                total_loss, construct_loss, desc_kl_loss, nll_loss, kl_loss, mf_loss = self.step(i)

                pbar.update(1)
                pbar.set_description(
                    'loss: %.4f, c_loss: %.4f, mf: %.4f' % (total_loss, construct_loss, prev_mf_loss))

                if i % cfg.adv_log_step == 0 and writer != None:
                    if mf_loss != 0:
                        prev_mf_loss = mf_loss
                    loss_val = {
                        'loss': total_loss,
                        'xent': construct_loss,
                        'd_kl': desc_kl_loss,
                        't_kl': kl_loss,
                        't_xent': nll_loss,
                        'mf': prev_mf_loss,
                    }
                    for key, value in loss_val.items():
                        writer.add_scalar('G/'+key, value, i)

                if i % self.args.bleu_iter == 0:
                    self.model.eval()
                    self.calculate_bleu(writer, i)
                    self.model.train()

                if i % args.check_iter == 0:
                    torch.save({
                        'model': self.model.state_dict(),
                    }, os.path.join(save_path,'checkpoint_{}.pt'.format(i)))

                    torch.save({
                        'gen_opt': self.gen_opt,
                        'tmp_opt': self.tmp_opt,
                        'mf_opt': self.rec_opt,
                        # 'tmp_adv_opt': self.tmp_adv_opt.state_dict(),
                    }, os.path.join(save_path,'optimizers.pt'))


if __name__ == "__main__":
    
    import argparse
    # args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, args.gen_hidden_dim
    # args.dis_embed_dim, args.max_seq_len, args.num_rep
    # args.gen_lr args.gen_adv_lr, args.dis_lr
    parser = argparse.ArgumentParser(description='KKDay users')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pre-batch-size', type=int, default=48)
    parser.add_argument('--cache-path', type=str, default='dataset')
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--pretrain-embeddings', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--check-iter', type=int, default=1000, help='checkpoint every 1k')
    parser.add_argument('--bleu-iter', type=int, default=400, help='bleu evaluation step')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)
    parser.add_argument('--tokenize', '-t', type=str, default='word', choices=['word', 'char'])

    parser.add_argument('--name', type=str, default='rec')

    parser.add_argument('--tmp-latent-dim', type=int, default=16)
    parser.add_argument('--tmp-cat-dim', type=int, default=10)

    parser.add_argument('--desc-latent-dim', type=int, default=32)
    parser.add_argument('--user-latent-dim', type=int, default=64)
    parser.add_argument('--gen-embed-dim', type=int, default=128)

    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--dis-num-layers', type=int, default=5)
    parser.add_argument('--max-seq-len', type=int, default=64)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gumbel-max', type=float, default=5)

    parser.add_argument('--anneal-rate', type=float, default=0.00002)

    parser.add_argument('--gen-lr', type=float, default=0.0001)
    parser.add_argument('--rec-lr', type=float, default=0.0001)

    # parser.add_argument('--dis-lr', type=float, default=0.001)
    parser.add_argument('--grad-penalty', type=str2bool, nargs='?',
                        default=False, help='Apply gradient penalty')
    parser.add_argument('--full-text', type=str2bool, nargs='?',
                        default=False, help='Dataset return full max length')
    parser.add_argument('--update-latent', type=str2bool, nargs='?',
                        default=True, help='Update latent assignment every epoch?')
    parser.add_argument('--biset',type=str2bool, nargs='?',
                        default=False, help='Use BiSET module to fuse article/template feature')

    # parser.add_argument('--dis-weight', type=float, default=0.1)
    # parser.add_argument('--kl-weight', type=float, default=1.0)
    # parser.add_argument('--opt-level', type=str, default='O1')
    # parser.add_argument('--cycle-weight', type=float, default=0.2)
    # parser.add_argument('--re-weight', type=float, default=0.5)
    # parser.add_argument('--gp-weight', type=float, default=10)
    # parser.add_argument('--bin-weight', type=float, default=0.5)
    # parser.add_argument('--loss-type', type=str, default='rsgan', 
    #                     choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = TemplateTrainer(args)
    # trainer.pretrain(5)
    # trainer.sample_results(None)
    # trainer.step(1)
    # trainer.calculate_bleu(None, size=1000)
    # trainer.test()
    trainer.train()