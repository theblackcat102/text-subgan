import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.relgan_d import RelSpaceGAN_D
from module.cluster import VAE_Cluster
from module.relgan_g import RelGAN_Seq2Seq
from module.seq2seq import LuongAttention
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


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()

K_BINS = 20

class SubSpaceRelGANTrainer():


    def __init__(self, args):
        self.dataset = KKDayUser(-1, 'data/kkday_dataset/user_data', 
            'data/kkday_dataset/matrix_factorized_64.pkl',
            prefix='item_graph', embedding=None, max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, 
            token_level=args.tokenize)

        dataset = self.dataset
        self.dataloader = data_iter(torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True))

        self.pretrain_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size*4, shuffle=True)

        self.G = RelGAN_Seq2Seq(args.gen_embed_dim, 
            args.gen_hidden_dim, dataset.vocab_size,
            latent_dim=args.gen_latent_dim, noise_dim=args.gen_noise_dim,
            max_seq_len=args.max_seq_len-1, padding_idx=Constants.PAD, gpu=True)

        self.D = RelSpaceGAN_D(args.dis_embed_dim, args.max_seq_len-1, args.num_rep, dataset.vocab_size, Constants.PAD,
            kbins=K_BINS, gpu=True, dropout=0.25)

        self.C = VAE_Cluster(64, 64, k_bins=K_BINS, output_embed_dim=100)
        self.C.cuda()
        self.G.cuda()
        self.D.cuda()

        args.vocab_size = dataset.vocab_size
        self.args = args

        # self.cluster_opt = optim.Adam(self.C.parameters(), lr=args.dis_lr)
        self.gen_opt  = optim.Adam(self.G.parameters(), lr=args.gen_lr)

        self.gen_adv_opt  = optim.Adam(list(self.G.parameters()) + list(self.C.parameters()), lr=args.gen_adv_lr, betas=(0.5, 0.999))
        self.dis_opt  = optim.Adam(self.D.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=1e-7)

        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.KL_criterion = nn.KLDivLoss(reduction='batchmean')
        self.dis_criterion = nn.CrossEntropyLoss()

        self.data_iterator = data_iter(self.dataloader)

    def adv_train_generator(self, g_step=1):
        total_loss = 0
        total_g_loss, total_bin_loss, total_c_loss = 0, 0, 0
        g_step = min(g_step, 1)

        for step in range(g_step):
            batch = next(self.data_iterator)
            src_inputs = batch['src']
            items, users = batch['items'], batch['users']
            inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]

            if cfg.CUDA:
                inputs, items, users = inputs.cuda(), items.cuda(), users.cuda()
                src_inputs = src_inputs.cuda()
                inputs = inputs.cuda()
                target = target.cuda()

            batch_size = inputs.shape[0]
            real_samples = F.one_hot(target, self.args.vocab_size).float()

            d_out_real, kbins_real, embed_real = self.D(real_samples)

            # Train cluster module
            _c_logits, _ = self.C(items, users)
            c_logits = F.softmax(_c_logits, 1)
            c_bins = torch.argmax(c_logits, 1).long()

            gen_samples = self.G.sample(src_inputs, batch_size, batch_size, one_hot=True)
            # backprop discriminator gradient back to generator via gumbel softmax
            d_out_fake, kbins_fake, embed_fake = self.D(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, self.args.loss_type)
            bin_loss = self.dis_criterion(kbins_fake, c_bins)
            _c_logits = F.log_softmax(_c_logits, dim=1)
            # c_loss = self.KL_criterion(_c_logits, norm_kbins_)

            loss = self.args.g_weight*g_loss #+ self.args.c_weight*c_loss + bin_loss * 0.5
            # if self.args.bin_weight > 0:
            #     loss += bin_loss * self.args.bin_weight

            self.gen_adv_opt.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), cfg.clip_norm)
            self.gen_adv_opt.step()

            total_g_loss += g_loss.item()
            # total_bin_loss += bin_loss.item()
            total_loss += loss.item()
            # total_c_loss += c_loss.item()

        return total_loss / g_step, total_g_loss / g_step #, total_bin_loss / g_step, total_c_loss / g_step

    def adv_train_discriminator(self, d_step=1):
        total_loss = 0
        total_bin_loss, total_d_loss = 0, 0
        d_step = min(d_step, 1)
        for step in range(d_step):
            batch = next(self.data_iterator)
            src_inputs = batch['src']
            items, users = batch['items'], batch['users']
            inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]

            if cfg.CUDA:
                inputs, items, users = inputs.cuda(), items.cuda(), users.cuda()
                src_inputs = src_inputs.cuda()
                inputs = inputs.cuda()
                target = target.cuda()

            batch_size = inputs.shape[0]
            real_samples = F.one_hot(target, self.args.vocab_size).float()
            d_out_real, kbins_real, embed_real = self.D(real_samples)
            c_logits, _ = self.C(items, users)
            c_logits = F.softmax(c_logits, 1)
            c_bins = torch.argmax(c_logits, 1).long()

            with torch.no_grad():
                gen_samples = self.G.sample(src_inputs, batch_size, batch_size, one_hot=True)


            # ===Train===
            d_out_fake, kbins_fake, embed_fake = self.D(gen_samples.detach())
            _, d_loss = get_losses(d_out_real, d_out_fake, self.args.loss_type)

            if self.args.grad_penalty:
                d_gp_loss = gradient_penalty(self.D, real_samples, gen_samples.detach())
                d_loss += self.args.gp_weight*d_gp_loss

            bin_loss = self.dis_criterion(kbins_real, c_bins)

            loss = d_loss + bin_loss * 0.5
            # if self.args.bin_weight > 0:
            #     loss += bin_loss * self.args.bin_weight

            self.dis_opt.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), cfg.clip_norm)
            self.dis_opt.step()

            total_loss += loss.item()
            # total_bin_loss += bin_loss.item()
            total_d_loss += d_loss.item()

        return total_loss / d_step, total_d_loss / d_step #, total_bin_loss / d_step

    def sample_results(self, writer, step=0):
        results = self.G.sample(self.src_sample[:10], 10, 10)
        samples = ''
        for idx, sent in enumerate(results):
            sentence = []
            for token in sent:
                if token.item() == Constants.EOS:
                    break
                sentence.append(  self.dataset.idx2word[token.item()])
            samples += str(idx) + '. :' +' '.join(sentence) + '\n\n'
        # print(samples)
        if writer != None:
            writer.add_text("Text", samples, step)
            writer.flush()

    def pretrain_generator(self, epochs, writer=None):
        iter_ = 0
        for epoch in range(epochs):
            print('Epoch '+ str(epoch))
            with tqdm(total=len(self.pretrain_dataloader), dynamic_ncols=True) as pbar:
                for batch in self.pretrain_dataloader:
                    batch = next(self.data_iterator)
                    src_inputs = batch['src']
                    items, users = batch['items'], batch['users']
                    inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]

                    if cfg.CUDA:
                        inputs, items, users = inputs.cuda(), items.cuda(), users.cuda()
                        src_inputs = src_inputs.cuda()
                        inputs = inputs.cuda()
                        target = target.cuda()

                    batch_size = inputs.shape[0]
                    encoder_outputs, hidden = self.G.init_hidden(src_inputs, batch_size=batch_size)
                    pred = self.G.forward(inputs, hidden, encoder_outputs, need_hidden=False)
                    self.gen_opt.zero_grad()
                    loss = self.mle_criterion(pred, target.flatten())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), cfg.clip_norm)
                    self.gen_opt.step()
                    iter_ += 1
                    pbar.update(1)
                    pbar.set_description('loss: {:.4f}'.format(loss.item()))
                    if iter_ % cfg.pre_log_step == 0 and writer is not None:
                        writer.add_scalar('pretrain/loss', loss.item(), iter_)

            torch.save(self.G.state_dict(), 'save/user_prod_relgan_G_pretrained_{}.pt'.format(self.args.tokenize))


    def calculate_bleu(self, writer, step=0, size=5000, ngram=4):
        '''
            writer: tensorboardX writer
            step: which iteration step
            size: compare a total of 1k sentences
        '''
        eval_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=8,
                        collate_fn=seq_collate, batch_size=20, shuffle=False)
        sentences, references = [], []
        scores_weights = { str(gram): [1/gram] * gram for gram in range(1, ngram+1)  }
        scores = { str(gram): 0 for gram in range(1, ngram+1)  }

        # print('Evaluate bleu scores', scores)
        with torch.no_grad():
            for batch in eval_dataloader:
                src_inputs = batch['src']
                inputs, target = batch['tgt'][:, :-1], batch['tgt'][:, 1:]

                if cfg.CUDA:
                    src_inputs = src_inputs.cuda()
                    # inputs = inputs.cuda()
                    # target = target.cuda()

                batch_size = src_inputs.shape[0]                
                results = self.G.sample(src_inputs, batch_size, batch_size)

                for idx, sent_token in enumerate(batch['tgt'][:, 1:]):
                    reference = []
                    for token in sent_token:
                        if token.item() == Constants.EOS:
                            break
                        reference.append(self.dataset.idx2word[token.item()] )
                    references.append(reference)

                    sent = results[idx]
                    sentence = []
                    for token in sent:
                        if token.item() == Constants.EOS:
                            break
                        sentence.append(  self.dataset.idx2word[token.item()])
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

    def _test(self):
        print(">start test")
        from time import time
        batch = next(self.data_iterator)
        src = batch['src']
        self.src_sample = src.cuda()
        print(self.G.lstm2out.weight.shape)
        self.G.lstm2out.weight.data.copy_(self.G.embeddings.weight.data)
        self.pretrain_generator(100)
        self.sample_results(None)
        self.adv_train_generator()
        self.adv_train_discriminator()
        self.calculate_bleu(None)
        

    def train(self):
        # initialize misc stuff, tensorboard, save path etc
        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = 'save/sub_{}-{}'.format(self.args.name, cur_time)
        os.makedirs(save_path, exist_ok=True)
        copyfile('module/relgan_d.py', os.path.join(save_path, 'relgan_d.py'))
        copyfile('module/relgan_g.py', os.path.join(save_path, 'relgan_g.py'))
        self.save_path = save_path
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/sub_{}-{}'.format(self.args.name, cur_time))

        # print('Pretrain stage....')
        # if args.pretrain_gen is not None:
        #     gen_dict = torch.load(args.pretrain_gen)
        #     self.G.load_state_dict(gen_dict)
        # else:
        #     self.pretrain_generator(args.pretrain_epochs, writer=writer)

        if self.args.pretrain_embeddings != None:
            model = fasttext.load_model(self.args.pretrain_embeddings)
            embedding_weight = self.G.embeddings.cpu().weight.data
            hit = 0
            for word, idx in self.dataset.word2idx.items():
                embedding_weight[idx] = torch.from_numpy(model[word]).float()
                hit += 1
            embedding_weight = embedding_weight.cuda()
            self.G.embeddings.weight.data.copy_(embedding_weight)
            self.G.embeddings.cuda()
            self.G.lstm2out.weight.data.copy_(self.G.embeddings.weight.data)

        if self.args.gen_embed_dim == self.args.dis_embed_dim:
            self.D.embeddings.weight.data.copy_(self.G.embeddings.weight.data.T)

        # fix our latent feature for sampling
        batch = next(self.data_iterator)
        src = batch['src']
        # fix latent feature
        self.src_sample = src.cuda()

        if args.pretrain_gen is not None:
            gen_dict = torch.load(args.pretrain_gen)
            self.G.load_state_dict(gen_dict)
        else:
            self.pretrain_generator(args.pretrain_epochs, writer=writer)

        # resample_freq = len(self.dataset) // self.args.batch_size
        # print(resample_freq)
        # if writer != None:
        #     writer.add_histogram('K_Bins', self.dataset.p, 0)
        #     writer.add_histogram('Latent', self.dataset.latent, 0)

        # start training...
        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                # update discriminator
                self.D.train()
                d_t_loss, d_loss = self.adv_train_discriminator(d_step=self.args.dis_steps)

                # update generator
                self.G.train()
                g_t_loss, g_loss = self.adv_train_generator(g_step=self.args.gen_steps)


                self.update_temp(i, args.iterations)
                pbar.set_description(
                    'g_loss: %.4f, d_loss: %.4f, temp: %.4f' % (g_loss, d_loss, self.G.temperature))

                if i % args.check_iter == 0:
                    torch.save({
                        'model': self.G.state_dict(),
                    }, os.path.join(save_path,'relgan_G_{}.pt'.format(i)))

                    torch.save(self.D.state_dict(), os.path.join(save_path,'relgan_D.pt'))
                    torch.save({
                        'gen_opt': self.gen_opt,
                        'gen_adv_opt': self.gen_adv_opt,
                        'dis_opt': self.dis_opt,
                    }, os.path.join(save_path,'optimizers.pt'))

                if i % cfg.adv_log_step == 0 and writer != None:
                    writer.add_scalar('G/loss', g_t_loss, i)
                    writer.add_scalar('G/g_loss', g_loss, i)
                    writer.add_scalar('G/temp', self.G.temperature, i)

                    writer.add_scalar('D/loss', d_t_loss, i)
                    writer.add_scalar('D/d_loss', d_loss, i)


                pbar.update(1)
                if i % 100 == 0:
                    curr_temp = self.G.temperature
                    self.G.temperature = 1.0
                    self.G.eval()
                    self.sample_results(writer, i)
                    self.G.temperature = curr_temp
                    self.G.train()

                if i % self.args.bleu_iter == 0:
                    curr_temp = self.G.temperature
                    self.G.temperature = 1.0
                    self.G.eval(), self.D.eval(), self.C.eval()
                    self.calculate_bleu(writer, i)
                    self.G.temperature = curr_temp
                    self.G.train(), self.D.train(), self.C.train()
                
                # if i % resample_freq == 0 and i > 0 and self.args.update_latent:
                #     resample_freq = len(self.dataset) // self.args.batch_size
                #     self.update_latent(writer, i)

        writer.flush()

    def update_temp(self, i, N):
        # temperature = np.maximum( np.exp(-self.args.anneal_rate * i), self.args.temperature_min)
        self.G.temperature = get_fixed_temperature(self.args.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        """Add clip_grad_norm_"""
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()



if __name__ == "__main__":
    import argparse
    # args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, args.gen_hidden_dim
    # args.dis_embed_dim, args.max_seq_len, args.num_rep
    # args.gen_lr args.gen_adv_lr, args.dis_lr
    parser = argparse.ArgumentParser(description='KKDay users')
    parser.add_argument('--batch-size', type=int, default=32)
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

    parser.add_argument('--gen-noise-dim', type=int, default=100)
    parser.add_argument('--gen-latent-dim', type=int, default=K_BINS)
    parser.add_argument('--gen-embed-dim', type=int, default=128)
    parser.add_argument('--gen-hidden-dim', type=int, default=128)
    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=cfg.temperature)

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

    parser.add_argument('--c-weight', type=float, default=1)
    parser.add_argument('--g-weight', type=float, default=1)
    parser.add_argument('--gp-weight', type=float, default=10)
    parser.add_argument('--bin-weight', type=float, default=0.5)
    parser.add_argument('--loss-type', type=str, default='rsgan', 
                        choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = SubSpaceRelGANTrainer(args)
    # trainer.update_latent()
    # trainer.pretrain_generator(args.pretrain_epochs)
    trainer.train()
    # trainer._test()