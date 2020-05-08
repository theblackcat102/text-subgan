import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.relgan_d import RelSeqGAN_D
from module.cluster import VAE_Cluster
from module.vmt import VMT
from module.vae import GaussianKLLoss
from dataset import KKDayUser, seq_collate
from constant import Constants
import fasttext
from utils import get_fixed_temperature, get_losses
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks
from sklearn.manifold import SpectralEmbedding
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from shutil import copyfile
from apex import amp


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()


class TemplateTrainer():

    def __init__(self, args):
        self.dataset2 = KKDayUser(-1, 'data/kkday_dataset/user_data', 
            'data/kkday_dataset/matrix_factorized_64.pkl',
            prefix='item_graph', embedding=None, max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, 
            token_level=args.tokenize, is_train=True)
        self.dataset1 = KKDayUser(-1, 'data/kkday_dataset/user_data', 
            'data/kkday_dataset/matrix_factorized_64.pkl',
            prefix='item_graph', embedding=None, max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, 
            token_level=args.tokenize, is_train=True)
        self.val_dataset = KKDayUser(-1, 'data/kkday_dataset/user_data', 
            'data/kkday_dataset/matrix_factorized_64.pkl',
            prefix='item_graph', is_train=False,embedding=None, max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, 
            token_level=args.tokenize)
        
        self.model = VMT(args.gen_embed_dim, self.dataset2.vocab_size,
            enc_hidden_size=128, dec_hidden_size=128, 
            tmp_latent_dim=args.tmp_latent_dim, desc_latent_dim=args.desc_latent_dim, user_latent_dim=args.user_latent_dim,

            max_seq_len=args.max_seq_len-1, gpu=True)
        
        self.C = VAE_Cluster(64, 64, k_bins=10, output_embed_dim=args.user_latent_dim)
        self.C.cuda()
        self.model.cuda()
    
        args.vocab_size = self.dataset1.vocab_size
        self.args = args

        # self.cluster_opt = optim.Adam(self.C.parameters(), lr=args.dis_lr)
        self.gen_opt  = optim.Adam(self.model.parameters(), lr=args.gen_lr)

        self.gen_adv_opt  = optim.Adam(list(self.model.parameters()) + list(self.C.parameters()), lr=args.gen_adv_lr, betas=(0.5, 0.999))

        opt_level = args.opt_level
        # [self.model, self.C], self.gen_adv_opt = amp.initialize([self.model, self.C], self.gen_adv_opt, opt_level=opt_level)

        self.dataloader1 = torch.utils.data.DataLoader(self.dataset1, num_workers=6,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True, drop_last=True)

        self.dataloader2 = torch.utils.data.DataLoader(self.dataset2, num_workers=6,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True, drop_last=True)

        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.KL_loss = GaussianKLLoss()

        self.data_iterator1 = data_iter(self.dataloader1)
        self.data_iterator2 = data_iter(self.dataloader2)
        self.init_sample_inputs()

    def pretrain(self, epochs, writer=None):
        iter_ = 0
        pretrain_dataloader = torch.utils.data.DataLoader(self.dataset2, num_workers=6,
                        collate_fn=seq_collate, batch_size=args.pre_batch_size, shuffle=True, drop_last=True)
        for epoch in range(epochs):
            print('Epoch '+ str(epoch))
            with tqdm(total=len(pretrain_dataloader), dynamic_ncols=True) as pbar:
                for batch in pretrain_dataloader:
                    src_inputs = batch['src']
                    tmp = batch['tmp']
                    items, users = batch['items'], batch['users']
                    inputs, target = batch['tmp'][:, :-1], batch['tmp'][:, 1:]

                    if cfg.CUDA:
                        inputs, items, users = inputs.cuda(), items.cuda(), users.cuda()
                        src_inputs = src_inputs.cuda()
                        inputs = inputs.cuda()
                        target = target.cuda()
                        tmp = tmp.cuda()

                    nll_loss, kl_loss = self.model.cycle_template(inputs, target)
                    loss = nll_loss+kl_loss
                    self.gen_opt.zero_grad()
                    loss.backward()
                    self.gen_opt.step()
                    iter_ += 1
                    pbar.update(1)
                    pbar.set_description('loss: {:.4f}'.format(loss.item()))

                    if writer:
                        writer.add_scalar('pretrain/loss', loss.item(), iter_)




    def step(self, i):
        batch1 = next(self.data_iterator1)
        src_inputs1 = batch1['src']
        tmp1 = batch1['tmp']
        items1, users1 = batch1['items'], batch1['users']
        item_ids1 = batch1['item_ids']
        inputs1, target1 = batch1['tgt'][:, :-1], batch1['tgt'][:, 1:]

        if cfg.CUDA:
            inputs1, items1, users1 = inputs1.cuda(), items1.cuda(), users1.cuda()
            src_inputs1 = src_inputs1.cuda()
            item_ids1 = item_ids1.cuda()
            inputs1 = inputs1.cuda()
            target1 = target1.cuda()
            tmp1 = tmp1.cuda()
        

        batch2 = next(self.data_iterator2)
        src_inputs2 = batch2['src']
        tmp2 = batch2['tmp']
        items2, users2 = batch2['items'], batch2['users']
        item_ids2 = batch2['item_ids']
        inputs2, target2 = batch2['tgt'][:, :-1], batch2['tgt'][:, 1:]

        if cfg.CUDA:
            inputs2, items2, users2 = inputs2.cuda(), items2.cuda(), users2.cuda()
            src_inputs2 = src_inputs2.cuda()
            item_ids2 = item_ids2.cuda()
            inputs2 = inputs2.cuda()
            target2 = target2.cuda()
            tmp2 = tmp2.cuda()

        _, user1_embed = self.C(items1, users1)
        _, user2_embed = self.C(items2, users2)

        nll_loss1, kl_loss1 = self.model.cycle_template(tmp1[:, :-1], tmp1[:, 1:])
        nll_loss2, kl_loss2 = self.model.cycle_template(tmp2[:, :-1], tmp2[:, 1:])

        cycle_nll = (nll_loss1+nll_loss2)/2
        cycle_kl = (kl_loss1+kl_loss2)/2

        nll_loss1, kl_loss1 = self.model.cycle_template(inputs1, tmp1[:, 1:])
        nll_loss2, kl_loss2 = self.model.cycle_template(inputs2, tmp2[:, 1:])
        
        title_cycle_nll = (nll_loss1+nll_loss2)/2
        title_cycle_kl = (kl_loss1+kl_loss2)/2

        desc1_outputs, desc1_latent, desc1_mean, desc1_std = self.model.encode_desc(src_inputs1)
        _, tmp1_latent, tmp1_mean, tmp1_std = self.model.encode_tmp(tmp1)
        output1_target, output1_logits = self.model.decode(tmp1_latent, desc1_latent, user1_embed, desc1_outputs,
                max_length=target1.shape[1])
        

        desc2_outputs, desc2_latent, desc2_mean, desc2_std = self.model.encode_desc(src_inputs2)
        _, tmp2_latent, tmp2_mean, tmp2_std = self.model.encode_tmp(tmp2)
        output2_target, output2_logits = self.model.decode(tmp2_latent, desc2_latent, user2_embed, desc2_outputs,
            max_length=target2.shape[1])


        desc_kl_loss = self.KL_loss(desc2_mean, desc2_std) +  self.KL_loss(desc1_mean, desc1_std)
        tmp_kl_loss = self.KL_loss(tmp2_mean, tmp2_std) +  self.KL_loss(tmp1_mean, tmp1_std)

        construct1_loss = self.mle_criterion(output1_target.view(-1, self.args.vocab_size), target1.flatten())
        construct2_loss = self.mle_criterion(output2_target.view(-1, self.args.vocab_size), target2.flatten())

        reconstruct_loss = (construct1_loss+construct2_loss)/2

        total_kl_loss = (tmp_kl_loss+desc_kl_loss) + cycle_kl
        cycle_nll_loss = cycle_nll

        title_loss = (title_cycle_kl+title_cycle_nll) #* self.temp

        total_loss = reconstruct_loss*self.args.re_weight + cycle_nll_loss*self.args.cycle_weight + total_kl_loss*self.args.kl_weight + title_loss



        self.gen_adv_opt.zero_grad()
        # with amp.scale_loss(total_loss, self.gen_adv_opt) as scaled_loss:
        #     scaled_loss.backward()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_norm)
        torch.nn.utils.clip_grad_norm_(self.C.parameters(), cfg.clip_norm)
        # torch.nn.utils.clip_grad_norm_(amp.master_params(self.gen_adv_opt), cfg.clip_norm)

        self.gen_adv_opt.step()

        return total_loss.item(), total_kl_loss.item(), reconstruct_loss.item(), cycle_nll_loss.item(), title_cycle_nll.item(), cycle_nll.item()

    def sample_results(self, writer, step=0):
        sample_size = 5
        logits, embed = self.C(self.items[:sample_size,], self.users[:sample_size,] )

        desc1_outputs, desc1_latent, _, _ = self.model.encode_desc(self.descripion[:sample_size])
        _, tmp1_latent, _, _ = self.model.encode_tmp(self.template[:sample_size])
        _, tmp2_latent, _, _ = self.model.encode_tmp(self.template2[:sample_size])
        _, output_title = self.model.decode(tmp1_latent, desc1_latent, embed, desc1_outputs,
                max_length=self.args.max_seq_len)
        _, output_title2 = self.model.decode(tmp2_latent, desc1_latent, embed, desc1_outputs,
                max_length=self.args.max_seq_len)

        # output_title = torch.argmax(output1_logits, dim=-1)
        # output_title2 = torch.argmax(output2_logits, dim=-1)
        samples, new_sample = '', ''
        with torch.no_grad():
            for idx, sent in enumerate(output_title):

                sentence = []
                for token in self.template[idx][1:]:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.dataset1.idx2word[token.item()])
                samples += str(idx) + '. [tmp]: ' +' '.join(sentence) + '\n\n'

                sentence = []
                for token in sent:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.dataset1.idx2word[token.item()])
                samples += '       [out]: ' +' '.join(sentence[:30]) + '\n\n'

                sentence = []
                for token in self.template2[idx][1:]:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.dataset1.idx2word[token.item()])
                new_sample += str(idx) + '. [tmp]: ' +' '.join(sentence) + '\n\n'

                sentence = []
                for token in output_title2[idx]:
                    if token.item() == Constants.EOS:
                        break
                    sentence.append(  self.dataset1.idx2word[token.item()])
                new_sample += '       [mod]: ' +' '.join(sentence[:30]) + '\n\n'

        if writer != None:
            writer.add_text("Text", samples, step)
            writer.flush() 

            writer.add_text("Transfer", new_sample, step)
            writer.flush() 


    def calculate_bleu(self, writer, step=0, size=1000, ngram=4):
        eval_dataloader = torch.utils.data.DataLoader(self.val_dataset, num_workers=8,
                        collate_fn=seq_collate, batch_size=20, shuffle=False)
        sentences, references = [], []
        scores_weights = { str(gram): [1/gram] * gram for gram in range(1, ngram+1)  }
        scores = { str(gram): 0 for gram in range(1, ngram+1)  }

        # print('Evaluate bleu scores', scores)
        with torch.no_grad():
            for batch in eval_dataloader:
                src_inputs = batch['src']
                tmp = batch['tmp']
                items, users = batch['items'], batch['users']
                inputs, target1 = batch['tgt'][:, :-1], batch['tgt'][:, 1:]

                if cfg.CUDA:
                    inputs, items, users = inputs.cuda(), items.cuda(), users.cuda()
                    src_inputs = src_inputs.cuda()
                    inputs = inputs.cuda()
                    target1 = target1.cuda()
                    tmp = tmp.cuda()

                batch_size = src_inputs.shape[0]
                logits, embed = self.C(items, users)
                desc_outputs, desc_latent, _, _ = self.model.encode_desc(src_inputs)
                _, tmp_latent, _, _ = self.model.encode_tmp(tmp)
                _, output_title = self.model.decode(tmp_latent, desc_latent, embed, desc_outputs,
                        max_length=self.args.max_seq_len)
                
                # output_title = torch.argmax(output_logits, dim=-1)
                for idx, sent_token in enumerate(batch['tgt'][:, 1:]):
                    reference = []
                    for token in sent_token:
                        if token.item() == Constants.EOS:
                            break
                        reference.append(self.val_dataset.idx2word[token.item()] )
                    references.append(reference)

                    sent = output_title[idx]
                    sentence = []
                    for token in sent:
                        if token.item() == Constants.EOS:
                            break
                        sentence.append(  self.val_dataset.idx2word[token.item()])
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

    def init_sample_inputs(self):
        batch1 = next(self.data_iterator1)
        src_inputs1 = batch1['src']
        tmp1 = batch1['tmp']
        items1, users1 = batch1['items'], batch1['users']
        inputs1, target1 = batch1['tgt'][:, :-1], batch1['tgt'][:, 1:]

        if cfg.CUDA:
            inputs1, items1, users1 = inputs1.cuda(), items1.cuda(), users1.cuda()
            src_inputs1 = src_inputs1.cuda()
            inputs1 = inputs1.cuda()
            target1 = target1.cuda()
            tmp1 = tmp1.cuda()

        self.title_input = inputs1
        self.items, self.users = items1, users1
        self.descripion = src_inputs1
        self.template = tmp1

        batch2 = next(self.data_iterator1)
        tmp2 = batch2['tmp']
        if cfg.CUDA:
            tmp2 = tmp2.cuda()
        self.template2 = tmp2


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
        save_path = 'save/temp_{}-{}'.format(self.args.name, cur_time)
        os.makedirs(save_path, exist_ok=True)
        copyfile('module/vmt.py', os.path.join(save_path, 'vmt.py'))
        copyfile('module/vae.py', os.path.join(save_path, 'vae.py'))
        self.save_path = save_path
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/temp_{}-{}'.format(self.args.name, cur_time))
        self.temp = args.temperature_min 

        # self.pretrain(1, writer=writer)

        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                self.model.train(), self.C.train()
                total_loss, total_kl_loss, reconstruct_loss, cycle_nll_loss, title_cycle_nll, tmp_cycle_nll = trainer.step(i)

                if i % self.args.bleu_iter == 0:
                    self.model.eval(), self.C.eval()
                    self.calculate_bleu(writer, i)
                    self.model.train(), self.C.train()

                if i % cfg.adv_log_step == 0 and writer != None:
                    writer.add_scalar('G/loss', total_loss, i)
                    writer.add_scalar('G/kl_loss', total_kl_loss, i)
                    writer.add_scalar('G/reconstruct_loss', reconstruct_loss, i)
                    writer.add_scalar('G/cycle_nll_loss', cycle_nll_loss, i)

                if i % 100 == 0:
                    self.model.eval(), self.C.eval()
                    self.sample_results(writer, i)
                    self.model.train(), self.C.train()
                self.temp = self.update_temp(i, args.iterations)
                
                if i % args.check_iter == 0:
                    torch.save({
                        'model': self.model.state_dict(),
                        'C': self.C.state_dict(),
                        # 'amp': amp.state_dict()
                    }, os.path.join(save_path,'amp_checkpoint_{}.pt'.format(i)))

                    torch.save({
                        'gen_opt': self.gen_opt,
                        'gen_adv_opt': self.gen_adv_opt.state_dict(),
                    }, os.path.join(save_path,'optimizers.pt'))

                pbar.update(1)
                pbar.set_description(
                    'g_loss: %.4f, c_loss: %.4f, cycle: %.4f' % (total_loss, reconstruct_loss, cycle_nll_loss))

    def update_temp(self, i, N):
        # temperature = np.maximum( np.exp(-self.args.anneal_rate * i), self.args.temperature_min)
        return get_fixed_temperature(self.args.temperature, i, N, cfg.temp_adpt)


if __name__ == "__main__":
    
    import argparse
    # args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, args.gen_hidden_dim
    # args.dis_embed_dim, args.max_seq_len, args.num_rep
    # args.gen_lr args.gen_adv_lr, args.dis_lr
    parser = argparse.ArgumentParser(description='KKDay users')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pre-batch-size', type=int, default=48)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--pretrain-embeddings', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--check-iter', type=int, default=1000, help='checkpoint every 1k')
    parser.add_argument('--bleu-iter', type=int, default=400, help='bleu evaluation step')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)
    parser.add_argument('--tokenize', '-t', type=str, default='word', choices=['word', 'char'])

    parser.add_argument('--name', type=str, default='rec_gan')

    parser.add_argument('--tmp-latent-dim', type=int, default=32)
    parser.add_argument('--desc-latent-dim', type=int, default=128)
    parser.add_argument('--user-latent-dim', type=int, default=32)
    parser.add_argument('--gen-embed-dim', type=int, default=128)

    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=64)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1)

    parser.add_argument('--anneal-rate', type=float, default=0.00002)

    parser.add_argument('--gen-lr', type=float, default=0.0001)
    parser.add_argument('--gen-adv-lr', type=float, default=0.0001)
    parser.add_argument('--dis-lr', type=float, default=0.001)
    parser.add_argument('--grad-penalty', type=str2bool, nargs='?',
                        default=False, help='Apply gradient penalty')
    parser.add_argument('--full-text', type=str2bool, nargs='?',
                        default=False, help='Dataset return full max length')
    parser.add_argument('--update-latent', type=str2bool, nargs='?',
                        default=True, help='Update latent assignment every epoch?')

    parser.add_argument('--kl-weight', type=float, default=0.1)
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--cycle-weight', type=float, default=0.2)
    parser.add_argument('--re-weight', type=float, default=0.5)
    parser.add_argument('--gp-weight', type=float, default=10)
    parser.add_argument('--bin-weight', type=float, default=0.5)
    parser.add_argument('--loss-type', type=str, default='rsgan', 
                        choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = TemplateTrainer(args)
    # trainer.sample_results(None)
    # trainer.calculate_bleu(None, size=1000)
    trainer.train()
    # trainer.pretrain(args.pretrain_epochs)

    # for _ in range(10000):
    #     total_loss, total_kl_loss, reconstruct_loss, cycle_nll_loss, title_cycle_nll, tmp_cycle_nll = trainer.step()
    #     print(total_loss, total_kl_loss, reconstruct_loss, cycle_nll_loss, title_cycle_nll,  tmp_cycle_nll)