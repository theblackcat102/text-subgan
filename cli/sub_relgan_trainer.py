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
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks
from sklearn.manifold import SpectralEmbedding


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()

K_BINS = 5

class SubSpaceRelGANTrainer():


    def __init__(self, args):
        self.dataset = TextSubspaceDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, 
            max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, k_bins=K_BINS)
        dataset = self.dataset
        self.dataloader = data_iter(torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True))

        self.pretrain_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size*4, shuffle=True)

        self.G = RelSpaceG(args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, 
            args.gen_hidden_dim, dataset.vocab_size,
            k_bins=5, latent_dim=args.gen_latent_dim, noise_dim=args.gen_noise_dim,
            max_seq_len=args.max_seq_len-1, padding_idx=Constants.PAD, gpu=True)
        self.D = RelSpaceGAN_D(args.dis_embed_dim, args.max_seq_len-1, args.num_rep, dataset.vocab_size, Constants.PAD,
            kbins=K_BINS, gpu=True, dropout=0.25)
        self.C = Cluster(self.D.embeddings, args.dis_embed_dim, embed_dim=100, k_bins=K_BINS, output_embed_dim=100)
        self.k_bins = K_BINS
        self.C.cuda()
        self.G.cuda()
        self.D.cuda()

        args.vocab_size = dataset.vocab_size
        self.args = args

        # self.cluster_opt = optim.Adam(self.C.parameters(), lr=args.dis_lr)
        self.gen_opt  = optim.Adam(self.G.parameters(), lr=args.gen_lr)
        self.gen_adv_opt  = optim.Adam(list(self.G.parameters()) + list(self.C.parameters()), lr=args.gen_adv_lr)
        self.dis_opt  = optim.Adam(self.D.parameters(), lr=args.dis_lr)

        self.mle_criterion = nn.NLLLoss()
        self.KL_criterion = nn.KLDivLoss()
        self.dis_criterion = nn.CrossEntropyLoss()

        self.data_iterator = data_iter(self.dataloader)

        self.bins_weight = self.dataset.calculate_stats().cuda()


    def pretrain_generator(self, epochs, writer=None):
        iter_ = 0
        for epoch in range(epochs):
            print('Epoch '+ str(epoch))
            with tqdm(total=len(self.pretrain_dataloader), dynamic_ncols=True) as pbar:
                for batch in self.pretrain_dataloader:
                    inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                    kbins, latent = batch['bins'], batch['latents']
                    kbins = F.one_hot(kbins, self.k_bins).float()

                    inputs, target = inputs.cuda(), target.cuda()
                    kbins, latent = kbins.cuda(), latent.cuda()
                    
                    batch_size = inputs.shape[0]
                    hidden = self.G.init_hidden(kbins, latent, batch_size=batch_size)
                    pred = self.G.forward(inputs, hidden, need_hidden=False)
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

            torch.save(self.G.state_dict(), 'save/subspace_relgan_G_pretrained.pt')

    def adv_train_generator(self, g_step=1):
        total_loss = 0
        total_g_loss, total_bin_loss, total_c_loss = 0, 0, 0
        g_step = min(g_step, 1)

        for step in range(g_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            kbins, latent = batch['bins'], batch['latents']
            kbins_ = F.one_hot(kbins, self.k_bins).float()
            if cfg.CUDA:
                kbins_, latent = kbins_.cuda(), latent.cuda()
                kbins = kbins.cuda()
                target = target.cuda()

            batch_size = inputs.shape[0]
            real_samples = F.one_hot(target, self.args.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            norm_kbins_ = F.softmax(kbins_ / self.bins_weight.expand(batch_size, self.k_bins), dim=1)

            d_out_real, kbins_real, embed_real = self.D(real_samples)

            # Train cluster module
            c_logits, _ = self.C(real_samples, embed_real)

            gen_samples = self.G.sample(kbins_, latent, batch_size, batch_size, one_hot=True)

            d_out_fake, kbins_fake, embed_fake = self.D(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, self.args.loss_type)
            bin_loss = self.dis_criterion(kbins_fake, kbins)
            c_logits = F.log_softmax(c_logits)
            c_loss = self.KL_criterion(c_logits, norm_kbins_)

            loss = g_loss + bin_loss + c_loss

            self.gen_adv_opt.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), cfg.clip_norm)
            self.gen_adv_opt.step()

            total_g_loss += g_loss.item()
            total_bin_loss += bin_loss.item()
            total_loss += loss.item()
            total_c_loss += c_loss.item()

        return total_loss / g_step, total_g_loss / g_step, total_bin_loss / g_step, total_c_loss / g_step

    def adv_train_discriminator(self, d_step=1):
        total_loss = 0
        total_bin_loss, total_d_loss = 0, 0
        d_step = min(d_step, 1)
        for step in range(d_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            kbins, latent = batch['bins'], batch['latents']
            kbins_ = F.one_hot(kbins, self.k_bins).float()
            if cfg.CUDA:
                kbins_, latent = kbins_.cuda(), latent.cuda()
                kbins = kbins.cuda()
                target = target.cuda()

            batch_size = inputs.shape[0]
            real_samples = F.one_hot(target, self.args.vocab_size).float()
            with torch.no_grad():
                gen_samples = self.G.sample(kbins_, latent, batch_size, batch_size, one_hot=True)

            if cfg.CUDA:
                real_samples = real_samples.cuda()

            # ===Train===
            d_out_real, kbins_real, embed_real = self.D(real_samples)
            d_out_fake, kbins_fake, embed_fake = self.D(gen_samples.detach())
            _, d_loss = get_losses(d_out_real, d_out_fake, self.args.loss_type)

            if self.args.grad_penalty:
                d_gp_loss = gradient_penalty(self.D, real_samples, gen_samples.detach())
                d_loss += self.args.gp_weight*d_gp_loss

            bin_loss = self.dis_criterion(kbins_real, kbins)

            loss = d_loss + bin_loss
            self.dis_opt.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), cfg.clip_norm)
            self.dis_opt.step()

            total_loss += loss.item()
            total_bin_loss += bin_loss.item()
            total_d_loss += d_loss.item()

        return total_loss / d_step, total_d_loss / d_step, total_bin_loss / d_step

    def sample_results(self, writer, step=0):
        results = self.G.sample(self.z_bins[:10, :], self.z_latents[:10, :], 10, 10)
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

    def _test(self):
        print(">start test")
        from time import time
        # self.sample_results(None)
        

    def update_latent(self):
        print('Updating latent feature')
        eval_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=8,
                        collate_fn=seq_collate, batch_size=64, shuffle=False)
        latents, P = [], []
        self.D.eval(), self.C.eval()
        iter_ = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, dynamic_ncols=True):
                inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                kbins, latent = batch['bins'], batch['latents']
                target = target.cuda()
                real_samples = F.one_hot(target, self.args.vocab_size).float()
                if cfg.CUDA:
                    real_samples = real_samples.cuda()
                _, _, embed_real = self.D(real_samples)
                # Train cluster module
                logits, embed = self.C(real_samples, embed_real)
                latents.append(embed.cpu())
                P.append(torch.argmax(logits, 1).cpu())
                iter_ += 1
                # if iter_ > 10:
                #     break

        latents = torch.cat(latents, 0).data.numpy()
        P = torch.cat(P).data.numpy()

        context = []
        print('Calculate eigen latent features')
        # 40000 use roughly 40G of memory so reduce this if your memory is lower than 64G
        for latents_ in tqdm(chunks(latents, 40000), dynamic_ncols=True):
            embeddings = SpectralEmbedding(n_components=self.k_bins, affinity='rbf').fit_transform(latents_)
            context.append(embeddings)
        context = np.concatenate(context, axis=0)
        assert self.dataset.latent.shape == context.shape
        assert self.dataset.p.shape == P.shape
        # update latent space
        self.dataset.latent = context
        self.dataset.p = P
        # update P weights
        self.bins_weight = self.dataset.calculate_stats().cuda()
        self.D.train(), self.C.train()

        # update sample kbins and latent feature
        batch = next(self.data_iterator)
        z_bins, z_latents = batch['bins'], batch['latents']
        z_bins = F.one_hot(z_bins, self.k_bins).float()

        # fix latent feature
        self.z_bins, self.z_latents = z_bins.cuda(), z_latents.cuda()

    def train(self):
        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = 'save/sub{}-{}'.format(self.args.name, cur_time)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/sub{}-{}'.format(self.args.name, cur_time))
        print('Pretrain stage....')
        if args.pretrain_gen is not None:
            gen_dict = torch.load(args.pretrain_gen)
            self.G.load_state_dict(gen_dict)
        else:
            self.pretrain_generator(args.pretrain_epochs, writer=writer)
        batch = next(self.data_iterator)
        z_bins, z_latents = batch['bins'], batch['latents']
        z_bins = F.one_hot(z_bins, self.k_bins).float()

        # fix latent feature
        self.z_bins, self.z_latents = z_bins.cuda(), z_latents.cuda()
        resample_freq = 1000
        print(resample_freq)
        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                self.D.train()
                d_t_loss, d_loss, d_bin_loss = self.adv_train_discriminator(d_step=self.args.dis_steps)
                self.G.train()
                g_t_loss, g_loss, g_bin_loss, c_loss = self.adv_train_generator(g_step=self.args.gen_steps)

                self.update_temp(i, args.iterations)
                pbar.set_description(
                    'g_loss: %.4f, d_loss: %.4f, temp: %.4f' % (g_loss, d_loss, self.G.temperature))

                if i % args.check_iter == 0:
                    torch.save(self.G.state_dict(), os.path.join(save_path,'relgan_G_{}.pt'.format(i)))
                    torch.save(self.D.state_dict(), os.path.join(save_path,'relgan_D.pt'))
                    torch.save({
                        'gen_opt': self.gen_opt,
                        'gen_adv_opt': self.gen_adv_opt,
                        'dis_opt': self.dis_opt
                    }, os.path.join(save_path,'optimizers.pt'))

                if i % cfg.adv_log_step == 0 and writer != None:
                    writer.add_scalar('G/loss', g_t_loss, i)
                    writer.add_scalar('G/g_loss', g_loss, i)
                    writer.add_scalar('G/bin_loss', g_bin_loss, i)
                    writer.add_scalar('G/temp', self.G.temperature, i)
                    writer.add_scalar('G/c_loss', c_loss, i)

                    writer.add_scalar('D/loss', d_t_loss, i)
                    writer.add_scalar('D/d_loss', d_loss, i)
                    writer.add_scalar('D/bin_loss', d_bin_loss, i)


                pbar.update(1)
                if i % 100 == 0:
                    curr_temp = self.G.temperature
                    self.G.temperature = 1.0
                    self.G.eval()
                    self.sample_results(writer, i)
                    self.G.temperature = curr_temp
                    self.G.train()
                
                if i % resample_freq == 0 and i > 0:
                    resample_freq = len(self.dataset) // self.args.batch_size
                    self.update_latent()

        writer.flush()

    def update_temp(self, i, N):
        # temperature = np.maximum( np.exp(-self.args.anneal_rate * i), self.args.temperature_min)
        self.G.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--check-iter', type=int, default=1000, help='checkpoint every 1k')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)

    parser.add_argument('--name', type=str, default='relgan')

    parser.add_argument('--head-size', type=int, default=256)
    parser.add_argument('--mem-slots', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--gen-noise-dim', type=int, default=100)
    parser.add_argument('--gen-latent-dim', type=int, default=5)
    parser.add_argument('--gen-embed-dim', type=int, default=64)
    parser.add_argument('--gen-hidden-dim', type=int, default=128)
    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.1)
    parser.add_argument('--anneal-rate', type=float, default=0.00002)

    parser.add_argument('--gen-lr', type=float, default=0.0001)
    parser.add_argument('--gen-adv-lr', type=float, default=0.0001)
    parser.add_argument('--dis-lr', type=float, default=0.0003)
    parser.add_argument('--grad-penalty', type=str2bool, nargs='?',
                        default=False, help='Apply gradient penalty')
    parser.add_argument('--full-text', type=str2bool, nargs='?',
                        default=False, help='Dataset return full max length')

    parser.add_argument('--gp-weight', type=float, default=10)

    parser.add_argument('--loss-type', type=str, default='rsgan', 
                        choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = SubSpaceRelGANTrainer(args)
    # trainer.update_latent()
    # trainer.pretrain_generator(args.pretrain_epochs)
    trainer.train()
    # trainer._test()