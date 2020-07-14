import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.vmt import VMT
from module.discriminator import CNNClassifierModel
from module.recommend import SimpleMF
from module.vae import GaussianKLLoss
from module.optimizer import AdamW
from dataset import TemPest, tempest_collate
from constant import Constants
import fasttext
from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from utils import get_fixed_temperature, get_losses
import sklearn
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from shutil import copyfile
import pickle
from time import time

def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()


class TemplateTrainer():

    def __init__(self, args):
        self.args = args
        #                                                  reduce memory load
        self.train_dataset = TemPest(args.cache_path, 'valid' if args.evaluate else 'train')
        self.valid_dataset = TemPest(args.cache_path, 'test' if args.evaluate else 'valid')
        self.id_mapping = torch.load(os.path.join(args.cache_path, 'id_mapping.pt'))
        user_size = len(self.id_mapping['user2id'])
        self.user_size = user_size
        self.prod_size = len(self.id_mapping['prod2id'])
        vocab_size = self.valid_dataset.vocab_size
        self.args.vocab_size = vocab_size

        self.model = VMT(args.gen_embed_dim, vocab_size,
            enc_hidden_size=256, dec_hidden_size=256, tmp_category=args.tmp_cat_dim,
            tmp_latent_dim=args.tmp_latent_dim, desc_latent_dim=args.desc_latent_dim, user_latent_dim=args.user_latent_dim,
            biset=args.biset, user_embedding=True, user_size=user_size,
            max_seq_len=args.max_seq_len-1, gpu=True).cuda()

        self.discriminator = CNNClassifierModel(vocab_size, 32, args.max_seq_len-1, 1).cuda()

        self.train_iter = data_iter(torch.utils.data.DataLoader(self.train_dataset, num_workers=3,
                        collate_fn=tempest_collate, batch_size=args.batch_size, shuffle=True, 
                        drop_last=True))
        self.prod_embeddings = nn.Embedding(self.prod_size+1, args.user_latent_dim).cuda()
        torch.nn.init.xavier_uniform_(self.prod_embeddings.weight)


        self.gen_opt = optim.Adam(self.model.title_decoder.parameters(), lr=args.gen_lr)
        self.dis_opt = optim.Adam(self.discriminator.parameters(), lr=args.dis_lr)

        self.text_opt = optim.Adam(self.model.parameters(), lr=args.text_lr)
        self.tmp_opt = optim.Adam(self.model.template_vae.parameters(), lr=args.text_lr)
        self.rec_opt = optim.Adam(list(self.model.user_embedding.parameters()) + list(self.prod_embeddings.parameters()), 
            args.rec_lr, weight_decay=1e-5 )

        self.title_rec = SimpleMF( self.prod_size+1, vocab_size, user_embedding=self.prod_embeddings, n_factors=args.user_latent_dim).cuda()
        self.trec_opt = AdamW(self.title_rec.parameters(), lr=args.dis_lr, weight_decay=1e-5)

        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.KL_loss = GaussianKLLoss()
        self.mse_criterion = nn.MSELoss()
        self.xent_criterion = nn.CrossEntropyLoss()

        self.gumbel_temp = 1.0
        self.temperature = 1.0
        self.l2 = args.l2_weight
        N = args.iterations
        self.gumbel_anneal_rate = 1 / N
        self.temp_anneal = 1.0 / N
        self.temp = args.temperature_min 
        self.init_sample_inputs()

    def evaluate(self, checkpoint_name):
        self.save_path = os.path.dirname(os.path.abspath(checkpoint_name))
        step = int(os.path.basename(checkpoint_name).split('_')[-1].replace('.pt',''))

        checkpoint = torch.load(checkpoint_name)
        self.model.eval(), self.prod_embeddings.eval()
        self.model.load_state_dict(checkpoint['model'])
        self.prod_embeddings.load_state_dict(checkpoint['prod_embeddings'])
        scores = self.calculate_bleu(None, step=step, size=100000, smoothing_function=SmoothingFunction().method3)
        with open('results.txt', 'a') as f:
            f.write('--------------------------------\n')
            f.write('checkpoint : {}\n'.format(checkpoint_name))
            f.write('filter method 3\n')
            f.write('|  BLEU  |  score   |\n')
            f.write('|--------|----------|\n'.format(checkpoint_name))
            for key, value in scores.items():
                f.write('|BLEU-{:<5}| {:>5.3f}|\n'.format(key, value))

        scores = self.calculate_bleu(None, step=step, size=100000, smoothing_function=SmoothingFunction().method5)
        with open('results.txt', 'a') as f:
            f.write('filter method 5\n')
            f.write('|  BLEU  |  score   |\n')
            f.write('|--------|----------|\n'.format(checkpoint_name))
            for key, value in scores.items():
                f.write('|BLEU-{:<5}| {:>5.3f}|\n'.format(key, value))

        # already mapped to id value
        user2product = self.id_mapping['user2product']
        eval_dataloader = torch.utils.data.DataLoader(self.valid_dataset, num_workers=8,
                        collate_fn=tempest_collate, batch_size=20, shuffle=False, drop_last=True)

        self.prod_embeddings.load_state_dict(checkpoint['prod_embeddings'])
        actual = []
        predicted = []
        with torch.no_grad():
            for batch in eval_dataloader:
                users = batch['users']
                non_empty_users = users != -1 
                if non_empty_users.sum() > 0:
                    users = users[non_empty_users].cuda()
                    user_embeddings = self.model.user_embedding( users )
                    for user_embed, user_id in zip(user_embeddings, users):
                        rank = torch.argsort((user_embed * self.prod_embeddings.weight).mean(1), descending=True)
                        actual.append( user2product[user_id.item()] )
                        predicted.append(rank.cpu().numpy())

        if len(predicted) > 10:
            print(actual[0], actual[1])
            with open('results.txt', 'a') as f:
                f.write('Recommendation Performance: \n')
                f.write('Recall@5 {:>10.3f}\n'.format(recall_at_k(actual, predicted, 5)))
                f.write('Recall@10 {:>10.3f}\n'.format(recall_at_k(actual, predicted, 10)))

                f.write('Precision@5 {:>10.3f}\n'.format(precision_at_k(actual, predicted, 5)))
                f.write('Precision@10 {:>10.3f}\n'.format(precision_at_k(actual, predicted, 10)))

                f.write('NDCG@5 {:>10.3f}\n'.format(ndcg_k(actual, predicted, 5)))
                f.write('NDCG@10 {:>10.3f}\n'.format(ndcg_k(actual, predicted, 10)))

    def init_sample_inputs(self):
        batch = next(self.train_iter)
        src_inputs = batch['src']
        tmp = batch['tmt']
        inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]
        users = batch['users']
        empty_users = users == -1
        batch_size = len(empty_users)
        users_filled = users.detach()
        users_filled[empty_users] = torch.randint(0, self.user_size, ( int(empty_users.sum()),  ))

        self.tmt_ = tmp.cuda()
        self.users_ = users_filled.cuda()
        self.inputs_ = inputs.cuda()
        self.target_ = target.cuda()
        self.src_ = src_inputs.cuda()
        
    def sample_results(self, writer, step=0):
        sample_size = 5

        desc_outputs, desc_latent, _, _ = self.model.encode_desc(self.src_)
        tmp_outputs, tmp_latent = self.model.encode_tmp(self.tmt_)

        user_embeddings = self.model.user_embedding( self.users_ )
        output_target, output_title = self.model.decode(tmp_latent, desc_latent, user_embeddings, 
                desc_outputs, tmp_outputs,
                max_length=self.target_.shape[1])

        # output_title = torch.argmax(output1_logits, dim=-1)
        # output_title2 = torch.argmax(output2_logits, dim=-1)
        samples, new_sample = '', ''
        with torch.no_grad():
            for idx, sent in enumerate(output_title):

                sentence = []
                for token in self.tmt_[idx][1:]:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.valid_dataset.tokenizer.idx2word[token.item()])
                samples += str(idx) + '. [tmp]: ' +' '.join(sentence) + '\n\n'

                sentence = []
                for token in sent:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.valid_dataset.tokenizer.idx2word[token.item()])
                samples += '       [out]: ' +' '.join(sentence[:30]) + '\n\n'


        if writer != None:
            writer.add_text("Text", samples, step)
            writer.flush() 

    def calculate_bleu(self, writer, step=0, size=2000, ngram=4, smoothing_function=SmoothingFunction().method3):
        eval_dataloader = torch.utils.data.DataLoader(self.valid_dataset, num_workers=8,
                        collate_fn=tempest_collate, batch_size=20, shuffle=False, drop_last=True)
        sentences, references = [], []
        scores_weights = { str(gram): [1/gram] * gram for gram in range(1, ngram+1)  }
        scores = { str(gram): 0 for gram in range(1, ngram+1)  }
        mf_loss = 0
        tmf_loss = 0
        mf_cnt = 0
        start_t = time()
        user2product = self.id_mapping['user2product']

        actual = []
        predicted = []
        with torch.no_grad():
            for batch in eval_dataloader:
                users = batch['users']
                non_empty_users = users != -1 
                if non_empty_users.sum() > 0:
                    users = users[non_empty_users].cuda()
                    user_embeddings = self.model.user_embedding( users )
                    for user_embed, user_id in zip(user_embeddings, users):
                        rank = torch.argsort((user_embed * self.prod_embeddings.weight).mean(1), descending=True)
                        actual.append( user2product[user_id.item()] )
                        predicted.append(rank.cpu().numpy())

        if len(predicted) > 10 and writer != None:
            writer.add_scalar('Rec/Recall@5',recall_at_k(actual, predicted, 5), step)
            writer.add_scalar('Rec/Recall@10',recall_at_k(actual, predicted, 10), step)

            writer.add_scalar('Rec/Precision@5',precision_at_k(actual, predicted, 5), step)
            writer.add_scalar('Rec/Precision@10',precision_at_k(actual, predicted, 10), step)

            writer.add_scalar('Rec/NDCG@5',ndcg_k(actual, predicted, 5), step)
            writer.add_scalar('Rec/NDCG@10',ndcg_k(actual, predicted, 10), step)
        # print('recomend eval end : ', time()-start_t)
        # print('Evaluate bleu scores', scores)
        start_t = time()
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

                non_empty_users = (batch['users'] != -1) & (batch['prods'] != -1)
                if non_empty_users.sum() > 0:
                    prods = batch['prods'][non_empty_users]
                    users = batch['users'][non_empty_users]
                    # neg_prods = batch['neg_prods'][non_empty_users]
                    users, prods = users.cuda(), prods.cuda()

                    user_embed = self.model.user_embedding( users )
                    prod_embed  = self.prod_embeddings( prods )
                    prediction = (user_embed*prod_embed).sum(1)

                    rating = torch.ones(prediction.shape).float().cuda()
                    n_rating = (torch.randn(prediction.shape)*0.01 + torch.zeros(prediction.shape)).float().cuda() 

                    # n_prod_embed  = self.prod_embeddings( neg_prods )
                    # neg_prediction = (user_embed*n_prod_embed).sum(1)

                    mf_loss_ = self.mse_criterion(prediction, rating) #+ self.mse_criterion(neg_prediction, n_rating)
                    mf_loss += mf_loss_.item() 
                    mf_cnt += 1

                    # prediction = self.title_rec( prods,  target[ non_empty_users.cuda() ])
                    # n_prediction = self.title_rec( neg_prods,  target[ non_empty_users.cuda() ])

                    tmf_loss += (self.mse_criterion( prediction, rating ).item() )# + self.mse_criterion(n_prediction, n_rating)


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
                            smoothing_function=smoothing_function)

                if len(sentences) > size:
                    break
        # print('bleu eval end : ', time()-start_t)
        ref_name = '{}_reference.txt'.format(0)
        gen_name = '{}_generate.txt'.format(step)

        if self.args.evaluate:

            ref_name = 'eval_{}_reference.txt'.format(step)
            gen_name = 'eval_{}_generate.txt'.format(step)

        with open(os.path.join(self.save_path, ref_name), 'w') as f:
            for sent in references:
                f.write(' '.join(sent)+'\n')

        with open(os.path.join(self.save_path, gen_name), 'w') as f:
            for sent in sentences:
                f.write(' '.join(sent)+'\n')

        for key, weights in scores.items():
            scores[key] /= len(sentences)

        if writer != None:
            if mf_cnt > 0:
                writer.add_scalar('Val/mf', mf_loss/mf_cnt, step)
                writer.add_scalar('Val/text', tmf_loss/mf_cnt, step)

            for key, weights in scores.items():
                writer.add_scalar("Bleu/score-"+key, scores[key], step)
            writer.flush()
        return scores

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
        tmp_outputs, tmp_latent = self.model.encode_tmp(tmp[:, :-1], temperature=self.gumbel_temp)

        user_embeddings = self.model.user_embedding( users_filled )
        output_target, output_logits = self.model.decode(tmp_latent, desc_latent, user_embeddings, 
                desc_outputs, tmp_outputs,
                max_length=target.shape[1])

        self.temp = self.args.kl_weight

        nll_loss, kl_loss = self.model.cycle_template(tmp[:, :-1], tmp[:, 1:], temperature=self.gumbel_temp)
        self.tmp_opt.zero_grad()

        nll_loss.backward(retain_graph=True)
        kl_loss.backward(retain_graph=True)

        self.tmp_opt.step()

        self.text_opt.zero_grad()

        desc_kl_loss = self.KL_loss(desc_mean, desc_std)
        
        construct_loss = self.mle_criterion(output_target.view(-1, self.args.vocab_size), target.flatten())

        total_loss = construct_loss + desc_kl_loss

        total_loss.backward()
        self.text_opt.step()

        non_empty_users = (batch['users'] != -1) & (batch['prods'] != -1)
        mf_loss = 0
        tmf_loss = 0
        if non_empty_users.sum() > 0:
            prods = batch['prods'][non_empty_users]
            users = batch['users'][non_empty_users]
            neg_prods = batch['neg_prods'][non_empty_users]
            neg_prods = neg_prods.cuda()
            users, prods = users.cuda(), prods.cuda()

            # print(users.max(), prods.max(), prods.min(), users.min())
            self.rec_opt.zero_grad()

            user_embed = self.model.user_embedding( users )
            prod_embed  = self.prod_embeddings( prods )
            prediction = (user_embed*prod_embed).sum(1)
            rating = torch.ones(prediction.shape).float().cuda()

            rating = torch.ones(prediction.shape).float().cuda()
            n_rating = (torch.randn(prediction.shape)*0.01 + torch.zeros(prediction.shape)).float().cuda() 

            n_prod_embed  = self.prod_embeddings( neg_prods )
            neg_prediction = (user_embed*n_prod_embed).sum(1)

            mf_loss_ = self.mse_criterion(prediction, rating) + self.mse_criterion(neg_prediction, n_rating)
            L2_reg = torch.tensor(0., requires_grad=True).cuda()
            for name, param in self.prod_embeddings.named_parameters():
                if 'weight' in name:
                    L2_reg = L2_reg + torch.norm(param, 2).cuda()
            for name, param in self.model.user_embedding.named_parameters():
                if 'weight' in name:
                    L2_reg = L2_reg + torch.norm(param, 2).cuda()

            mf_loss = mf_loss_ + self.l2*L2_reg

            mf_loss.backward()
            self.rec_opt.step()

            mf_loss = mf_loss.item()

            self.trec_opt.zero_grad()
            prediction = self.title_rec( prods,  target[ non_empty_users.cuda() ])
            n_prediction = self.title_rec( neg_prods,  target[ non_empty_users.cuda() ])
            tmf_loss_ = self.mse_criterion( prediction, rating ) + self.mse_criterion(n_prediction, n_rating)
            tmf_loss_.backward()
            self.trec_opt.step()
            tmf_loss = tmf_loss_.item()

        return total_loss.item(), construct_loss.item(), desc_kl_loss.item(), nll_loss.item(), kl_loss.item(), mf_loss, tmf_loss


    def sample_latent(self):
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

        desc_outputs, desc_latent, _, _ = self.model.encode_desc(src_inputs)
        tmp_outputs, tmp_latent = self.model.encode_tmp(tmp[:, :-1], temperature=self.gumbel_temp)
        user_embeddings = self.model.user_embedding( users_filled )
        return tmp_latent, desc_latent, user_embeddings, desc_outputs, tmp_outputs, target


    def gan_step(self, i):
        with torch.no_grad():
            tmp_latent1, desc_latent1, user_embeddings1, desc_outputs1, tmp_outputs1, target1 = self.sample_latent()
            tmp_latent2, desc_latent2, user_embeddings2, desc_outputs2, tmp_outputs2, target2 = self.sample_latent()

        fake_target1, _, _ = self.model.decode(tmp_latent2, desc_latent1, user_embeddings1,
                desc_outputs1, tmp_outputs2,
                max_length=target1.shape[1], gumbel=True, temperature=self.temperature)
        fake_target2, _, _ = self.model.decode(tmp_latent1, desc_latent2, user_embeddings2, 
                desc_outputs2, tmp_outputs1,
                max_length=target2.shape[1], gumbel=True, temperature=self.temperature)

        D_fake1 = self.discriminator(fake_target1.detach()).mean()
        D_real1 = self.discriminator(target1, is_discrete=True).mean()
        D_fake2 = self.discriminator(fake_target2.detach()).mean()
        D_real2 = self.discriminator(target2, is_discrete=True).mean()

        D_loss = -self.args.dis_weight * ((D_real1 - D_fake1) + (D_real2 - D_fake2))

        self.dis_opt.zero_grad()
        D_loss.backward()
        self.dis_opt.step()

        G_wgan1 = -self.discriminator(fake_target1).mean()
        G_wgan2 = -self.discriminator(fake_target2).mean()
        G_wgan = self.args.gen_weight * (G_wgan1 + G_wgan2) / 2

        self.gen_opt.zero_grad()
        G_wgan.backward()
        self.gen_opt.step()

        return {
            'GAN/g_loss': G_wgan.item(),
            'GAN/g_wgan1': G_wgan1.item(),
            'GAN/g_wgan2': G_wgan2.item(),
            'GAN/d_loss': D_loss.item(),
            'GAN/d_fake': D_fake1.item()+D_fake2.item(),
            'GAN/d_real': D_real1.item()+D_real2.item(),
        }


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
        save_path = 'save/gan_tempest_{}-{}'.format(self.args.name, cur_time)
        os.makedirs(save_path, exist_ok=True)
        copyfile('module/vmt.py', os.path.join(save_path, 'vmt.py'))
        copyfile('module/vae.py', os.path.join(save_path, 'vae.py'))
        self.save_path = save_path
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/gan_tempest_{}-{}'.format(self.args.name, cur_time))

        i = 0
        self.temp = self.args.kl_weight
        self.gumbel_temp = 1.0
        # self.gumbel_temp = np.minimum(self.args.gumbel_max - (self.gumbel_temp * np.exp(-self.gumbel_anneal_rate * i)), 0.5)
        prev_mf_loss = 0
        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                self.model.train()

                dis_losses = self.gan_step(i)
                total_loss, construct_loss, desc_kl_loss, nll_loss, kl_loss, mf_loss, tmf_loss = self.step(i)
                

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
                        'prod_title_align': tmf_loss,
                    }
                    for key, value in loss_val.items():
                        writer.add_scalar('G/'+key, value, i)

                    for key, value in dis_losses.items():
                        writer.add_scalar(key, value, i)

                    writer.add_scalar('temp/gumbel', self.gumbel_temp, i)
                    writer.add_scalar('temp/temp', self.temperature, i)

                if i % self.args.bleu_iter == 0:
                    self.model.eval()
                    self.calculate_bleu(writer, i)
                    self.model.train()

                if i % 100 == 0:
                    self.model.eval(), self.prod_embeddings.eval(), self.title_rec.eval()
                    self.sample_results(writer, i)
                    self.model.train(), self.prod_embeddings.train(), self.title_rec.train()
                    self.gumbel_temp = np.maximum(self.gumbel_temp * np.exp(-self.gumbel_anneal_rate * i), 0.1)
                    self.temperature = np.maximum(np.exp(-self.args.anneal_rate * i), 0.1)
                    # self.gumbel_temp = np.maximum(self.args.gumbel_max ** (self.gumbel_anneal_rate * i), 0.00005)

                if i % args.check_iter == 0:
                    torch.save({
                        'model': self.model.state_dict(),
                        'title_rec': self.title_rec.state_dict(),
                        'prod_embeddings': self.prod_embeddings.state_dict(),
                    }, os.path.join(save_path,'checkpoint_{}.pt'.format(i)))

                    torch.save({
                        'text_opt': self.text_opt,
                        'tmp_opt': self.tmp_opt,
                        'mf_opt': self.rec_opt,
                        'gen_opt': self.gen_opt,
                        'dis_opt': self.dis_opt,
                        # 'tmp_adv_opt': self.tmp_adv_opt.state_dict(),
                    }, os.path.join(save_path,'optimizers.pt'))

                    torch.save({
                        'D': self.discriminator,
                        # 'tmp_adv_opt': self.tmp_adv_opt.state_dict(),
                    }, os.path.join(save_path,'discriminator.pt'))


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
    parser.add_argument('--bleu-iter', type=int, default=1000, help='bleu evaluation step')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)
    parser.add_argument('--tokenize', '-t', type=str, default='word', choices=['word', 'char'])

    parser.add_argument('--name', type=str, default='rec')

    parser.add_argument('--tmp-latent-dim', type=int, default=16)
    parser.add_argument('--tmp-cat-dim', type=int, default=10)

    parser.add_argument('--desc-latent-dim', type=int, default=32)
    parser.add_argument('--user-latent-dim', type=int, default=48)
    parser.add_argument('--gen-embed-dim', type=int, default=128)

    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--dis-num-layers', type=int, default=5)
    parser.add_argument('--max-seq-len', type=int, default=64)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gumbel-max', type=float, default=5)

    parser.add_argument('--anneal-rate', type=float, default=0.00002)

    parser.add_argument('--text-lr', type=float, default=0.001)
    parser.add_argument('--rec-lr', type=float, default=0.001)
    parser.add_argument('--trec-lr', type=float, default=0.001)
    parser.add_argument('--gen-lr', type=float, default=0.0001)
    parser.add_argument('--dis-lr', type=float, default=0.0005)


    parser.add_argument('--grad-penalty', type=str2bool, nargs='?',
                        default=False, help='Apply gradient penalty')
    parser.add_argument('--full-text', type=str2bool, nargs='?',
                        default=False, help='Dataset return full max length')
    parser.add_argument('--update-latent', type=str2bool, nargs='?',
                        default=True, help='Update latent assignment every epoch?')
    parser.add_argument('--biset',type=str2bool, nargs='?',
                        default=False, help='Use BiSET module to fuse article/template feature')
    parser.add_argument('--evaluate', type=str2bool, nargs='?',
                        default=False, help='Update latent assignment every epoch?')
    parser.add_argument('-ckpt','--checkpoint', type=str,
                        default='', help='Update latent assignment every epoch?')

    parser.add_argument('--dis-weight', type=float, default=0.1)
    parser.add_argument('--gen-weight', type=float, default=0.1)
    parser.add_argument('--kl-weight', type=float, default=1.0)
    parser.add_argument('--l2-weight', type=float, default=1e-3)
    # parser.add_argument('--opt-level', type=str, default='O1')
    # parser.add_argument('--cycle-weight', type=float, default=0.2)
    # parser.add_argument('--re-weight', type=float, default=0.5)
    # parser.add_argument('--gp-weight', type=float, default=10)
    # parser.add_argument('--bin-weight', type=float, default=0.5)
    # parser.add_argument('--loss-type', type=str, default='rsgan', 
    #                     choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = TemplateTrainer(args)
    trainer.gan_step(1)
    # trainer.pretrain(5)
    # trainer.sample_results(None)
    # trainer.step(1)
    # trainer.calculate_bleu(None, size=1000)
    # trainer.test()
    if args.evaluate:
        trainer.evaluate(args.checkpoint)
    else:
        trainer.train()
