import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg
from module.vae import VariationalAutoEncoder, GaussianKLLoss
from module.seq2seq import CycleLoss
from module.generator import VAE_Generator
from module.discriminator import VAE_Discriminator
from module.cluster import VAE_Cluster
# from module.relgan_g import RelSpaceG
from dataset import TextSubspaceDataset, seq_collate
from constant import Constants
from utils import get_fixed_temperature, get_losses
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool, chunks, data_iter
from sklearn.manifold import SpectralEmbedding
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from shutil import copyfile


K_BINS = 20

class VAE_GAN_Trainer():

    def __init__(self, args):
        self.dataset = TextSubspaceDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, 
            max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text, k_bins=K_BINS, token_level=args.tokenize)
        dataset = self.dataset
        self.dataloader = data_iter(torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True))
        self.k_bins = K_BINS

        self.G = VAE_Generator(args.gen_noise_dim, K_BINS, args.gen_latent_dim, 
            args.gen_num_layers, args.block_dim)
        self.D = VAE_Discriminator(k_bins=K_BINS, block_dim=args.block_dim, n_layers=args.dis_num_layers)
        self.C = VAE_Cluster( args.block_dim, embed_dim=args.block_dim, k_bins=K_BINS, output_embed_dim=K_BINS)
        self.V = VariationalAutoEncoder(dataset.vocab_size, 
            dec_hidden_size=args.block_dim, bin_latent_dim=K_BINS, k_bins=K_BINS,
            max_seq_len=args.max_seq_len, noise_dim=args.gen_noise_dim)

        self.G.cuda()
        self.V.cuda()
        self.C.cuda()
        self.D.cuda()

        self.gen_opt = optim.Adam(self.V.parameters(), lr=args.vae_lr)
        self.gen_adv_opt  = optim.Adam(list(self.G.parameters()) + list(self.C.parameters())
            , lr=args.gen_adv_lr, betas=(0.5, 0.999))
        self.dis_opt  = optim.Adam(self.D.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=1e-7)

        self.bins_weight = self.dataset.calculate_stats().cuda()        

        args.vocab_size = dataset.vocab_size
        self.args = args
        self.data_iterator = data_iter(self.dataloader)
        batch = next(self.data_iterator)
        while len(batch['bins']) < 10: # resample if length is under 10
            batch = next(self.data_iterator)
        z_bins, z_latents = batch['bins'], batch['latents']
        z_bins = F.one_hot(z_bins, self.k_bins).float()
        # fix latent feature
        self.z_bins, self.z_latents = z_bins.cuda(), z_latents.cuda()
        self.criteria = nn.BCEWithLogitsLoss()
        self.KL_criterion = nn.KLDivLoss(reduction='batchmean')
        self.dis_criterion = nn.CrossEntropyLoss()
        self.bins_weight = self.dataset.calculate_stats().cuda()
        self.kmeans = KMeans(n_clusters=K_BINS, random_state=666)
        self.dataset.p = self.kmeans.fit_predict(self.dataset.latent)


    def sample_results(self, writer, step=0):
        latent = self.G.sample(self.z_bins[:10, :], self.z_latents[:10, :], 10)
        results = self.V.sample_text(latent)
        samples = ''
        for idx, sent in enumerate(results):
            sentence = []
            for token in sent:
                if token.item() == Constants.EOS:
                    break
                sentence.append(  self.dataset.idx2word[token.item()])
            samples += str(idx) + '. ' +' '.join(sentence) + '\n\n'
        # print(samples)
        if writer != None:
            writer.add_text("Text", samples, step)
            writer.flush()

    def pretrain_vae(self, epochs, writer=None):
        iter_ = 0
        self.pretrain_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4,
                collate_fn=seq_collate, batch_size=self.args.batch_size, shuffle=True)

        criterion_KL = GaussianKLLoss()
        criterion_cycle = CycleLoss()

        for epoch in range(epochs):
            print('Epoch '+ str(epoch))
            with tqdm(total=len(self.pretrain_dataloader), dynamic_ncols=True) as pbar:
                for batch in self.pretrain_dataloader:
                    inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                    kbins, latent = batch['bins'], batch['latents']
                    kbins = F.one_hot(kbins, self.k_bins).float()
                    if cfg.CUDA:
                        inputs, target = inputs.cuda(), target.cuda()
                        kbins, latent = kbins.cuda(), latent.cuda()

                    outputs, latent, mean, std, de_outputs = self.V(kbins, latent, inputs, 
                        device='cuda', max_length=target.shape[1])
                    self.gen_opt.zero_grad()

                    loss_NLL = criterion_cycle(de_outputs, target)
                    loss_KL = criterion_KL(mean, std)
                    loss = loss_NLL + loss_KL
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.V.parameters(), cfg.clip_norm)
                    self.gen_opt.step()
                    iter_ += 1
                    pbar.update(1)
                    pbar.set_description('loss: {:.4f}'.format(loss.item()))

                    if iter_ % cfg.pre_log_step == 0 and writer is not None:
                        writer.add_scalar('pretrain/loss', loss.item(), iter_)
                        writer.add_scalar('pretrain/kl_loss', loss_KL.item(), iter_)
                        writer.add_scalar('pretrain/nll_loss', loss_NLL.item(), iter_)

                    # print(loss.item(), loss_NLL.item(), loss_KL.item())

            torch.save(self.V.state_dict(), 'save/vae_{}_{}.pt'.format(self.k_bins, self.args.tokenize))

    def adv_train_generator(self, g_step=1):
        total_loss, total_g_loss, total_bin_loss, total_c_loss = 0, 0, 0, 0
        g_step = min(g_step, 1)

        for step in range(g_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            batch_size = inputs.shape[0]

            kbins, latent = batch['bins'], batch['latents']
            kbins_ = F.one_hot(kbins, self.k_bins).float()
            real_label = torch.full((batch_size, 1), 1)

            if cfg.CUDA:
                kbins_, latent = kbins_.cuda(), latent.cuda()
                kbins = kbins.cuda()
                target = target.cuda()
                real_label = real_label.cuda()

            norm_kbins_ = F.softmax(kbins_ / self.bins_weight.expand(batch_size, self.k_bins), dim=1)
            self.gen_adv_opt.zero_grad()
            
            fake_latent = self.G.sample(kbins_, latent, batch_size)
            real_latent = self.V.encode(kbins_, latent, target, 'cuda')


            d_out_fake, kbins_fake, embed_fake = self.D(fake_latent)
            _, kbins_real, embed_real = self.D(real_latent)

            _c_logits, _ = self.C(real_latent, embed_real.detach())

            # which source should we use for kbins ?
            c_bins = torch.argmax(norm_kbins_.detach(), 1).long()

            _c_logits = F.log_softmax(_c_logits, dim=1)

            c_loss = self.KL_criterion(_c_logits, norm_kbins_)

            g_loss = self.criteria(d_out_fake, real_label)

            bin_loss = self.dis_criterion(kbins_fake, c_bins)

            loss = self.args.g_weight*g_loss + self.args.c_weight*c_loss
            if self.args.bin_weight > 0:
                loss += bin_loss * self.args.bin_weight

            loss.backward()

            self.gen_adv_opt.step()


            total_loss += loss.item()
            total_g_loss += g_loss.item()
            total_c_loss += c_loss.item()
            total_bin_loss += bin_loss.item()

        return total_loss / g_step, total_g_loss / g_step, total_bin_loss / g_step, total_c_loss / g_step

    def adv_train_discriminator(self, d_step=1):
        total_loss, total_bin_loss, total_d_loss = 0, 0, 0
        d_step = min(d_step, 1)
        for step in range(d_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            batch_size = inputs.shape[0]
            real_label = torch.full((batch_size, 1), 1)
            fake_label = torch.full((batch_size, 1), 0)

            kbins, latent = batch['bins'], batch['latents']
            kbins_ = F.one_hot(kbins, self.k_bins).float()
            if cfg.CUDA:
                kbins_ = kbins_.cuda()
                real_label, fake_label = real_label.cuda(), fake_label.cuda()
                latent = latent.cuda()
                kbins = kbins.cuda()
                target = target.cuda()

            norm_kbins_ = F.softmax(kbins_ / self.bins_weight.expand(batch_size, self.k_bins), dim=1)

            with torch.no_grad():
                fake_latent = self.G.sample(kbins_, latent, batch_size)
                real_latent = self.V.encode(kbins_, latent, target, 'cuda')

            self.dis_opt.zero_grad()

            d_out_fake, kbins_fake, embed_fake = self.D(fake_latent)
            d_out_real, kbins_real, embed_real = self.D(real_latent)

            real_loss = self.criteria(d_out_real, real_label)
            fake_loss = self.criteria(d_out_fake, fake_label)
            d_loss = fake_loss+real_loss

            if self.args.grad_penalty:
                d_gp_loss = gradient_penalty(self.D, real_samples, gen_samples.detach())
                d_loss += self.args.gp_weight*d_gp_loss
            loss = d_loss

            bin_loss = 0
            if self.args.bin_weight > 0:
                c_bins = torch.argmax(norm_kbins_, dim=1).detach()
                bin_loss = self.dis_criterion(kbins_real, c_bins)
                loss += bin_loss * self.args.bin_weight

            loss.backward()

            self.dis_opt.step()
            
            total_d_loss += d_loss.item()
            total_bin_loss += float(bin_loss)
            total_loss += loss.item()

        return total_loss / d_step, total_d_loss / d_step, total_bin_loss / d_step

    def calculate_bleu(self,writer, step=0, size=5000, ngram=4):
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
                inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                batch_size = inputs.shape[0]
                kbins, latent = batch['bins'], batch['latents']
                kbins = F.one_hot(kbins, self.k_bins).float()
                if cfg.CUDA:
                    kbins, latent = kbins.cuda(), latent.cuda()

                norm_kbins_ = F.softmax(kbins / self.bins_weight.expand(batch_size, self.k_bins), dim=1)                
                fake_latent = self.G.sample(norm_kbins_, latent, batch_size)
                results = self.V.sample_text(fake_latent)

                for idx, sent_token in enumerate(target):
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

    def update_latent(self, writer=None, iter=0):

        # print('Updating latent feature')
        eval_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=8,
                        collate_fn=seq_collate, batch_size=64, shuffle=False)
        latents, P = [], []
        self.D.eval(), self.C.eval()
        iter_ = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                kbins, latent = batch['bins'], batch['latents']
                kbins = F.one_hot(kbins, self.k_bins).float()
                if cfg.CUDA:
                    kbins, latent, target = kbins.cuda(), latent.cuda(), target.cuda()

                real_latent = self.V.encode(kbins, latent, target, 'cuda')

                _, _, embed_real = self.D(real_latent)
                # Train cluster module
                logits, embed = self.C(real_latent, embed_real)
                latents.append(embed.cpu())
                P.append(torch.argmax(logits, 1).cpu())
                iter_ += 1

        latents = torch.cat(latents, 0).data.numpy()
        P = torch.cat(P).data.numpy()

        context = []
        print('Calculate eigen latent features')
        # 40000 use roughly 40G of memory so reduce this if your memory is lower than 64G
        for latents_ in chunks(latents, 40000):
            embeddings = SpectralEmbedding(n_components=self.k_bins, affinity='rbf').fit_transform(latents_)
            context.append(embeddings)

        context = np.concatenate(context, axis=0)
        ss = sklearn.preprocessing.StandardScaler(copy=False)
        context = ss.fit_transform(context)
        assert self.dataset.latent.shape == context.shape
        # assert self.dataset.p.shape == P.shape
        # update latent space
        self.dataset.latent = context
        self.dataset.p = self.kmeans.fit_predict(self.dataset.latent)
        if writer != None:
            writer.add_histogram('K_Bins', self.dataset.p, iter)
            writer.add_histogram('Latent', self.dataset.latent, iter)
        # update P weights
        self.bins_weight = self.dataset.calculate_stats().cuda()
        self.D.train(), self.C.train()

        # update sample kbins and latent feature
        batch = next(self.data_iterator)
        while len(batch['bins']) < 10: # resample if length is under 10
            batch = next(self.data_iterator)
        z_bins, z_latents = batch['bins'], batch['latents']
        z_bins = F.one_hot(z_bins, self.k_bins).float()

        # fix latent feature
        self.z_bins, self.z_latents = z_bins.cuda(), z_latents.cuda()



    def train(self):
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

        print('Pretrain stage....')
        if args.pretrain_gen is not None:
            vae_dict = torch.load(args.pretrain_gen)
            self.V.load_state_dict(vae_dict)
        else:
            self.pretrain_vae(args.pretrain_epochs, writer=writer)

        # fix our latent feature for sampling
        batch = next(self.data_iterator)
        z_bins, z_latents = batch['bins'], batch['latents']
        z_bins = F.one_hot(z_bins, self.k_bins).float()
        # fix latent feature
        self.z_bins, self.z_latents = z_bins.cuda(), z_latents.cuda()
        resample_freq = len(self.dataset) // self.args.batch_size
        print(resample_freq)
        if writer != None:
            writer.add_histogram('K_Bins', self.dataset.p, 0)
            writer.add_histogram('Latent', self.dataset.latent, 0)

        # start training...
        with tqdm(total=args.iterations+1, dynamic_ncols=True) as pbar:
            for i in range(args.iterations+1):
                # update discriminator
                d_t_loss, d_loss, d_bin_loss = self.adv_train_discriminator(d_step=self.args.dis_steps)

                # update generator
                g_t_loss, g_loss, g_bin_loss, c_loss = self.adv_train_generator(g_step=self.args.gen_steps)

                # self.update_temp(i, args.iterations)
                pbar.set_description(
                    'g_loss: %.4f, d_loss: %.4f' % (g_loss, d_loss))

                if i % args.check_iter == 0:
                    torch.save({
                        'model': self.G.state_dict(),
                        'p': self.dataset.p,
                        'latent': self.dataset.latent,
                    }, os.path.join(save_path,'relgan_G_{}.pt'.format(i)))
                    torch.save({
                        'kmeans': self.kmeans,
                        'p': self.dataset.p,
                        'latent': self.dataset.latent,
                    }, os.path.join(save_path,'latest_cluster_assignment.pt'))
                    torch.save(self.D.state_dict(), os.path.join(save_path,'relgan_D.pt'))
                    torch.save({
                        'gen_opt': self.gen_opt,
                        'gen_adv_opt': self.gen_adv_opt,
                        'dis_opt': self.dis_opt,
                    }, os.path.join(save_path,'optimizers.pt'))

                if i % cfg.adv_log_step == 0 and writer != None:
                    writer.add_scalar('G/loss', g_t_loss, i)
                    writer.add_scalar('G/g_loss', g_loss, i)
                    writer.add_scalar('G/bin_loss', g_bin_loss, i)
                    writer.add_scalar('G/c_loss', c_loss, i)

                    writer.add_scalar('D/loss', d_t_loss, i)
                    writer.add_scalar('D/d_loss', d_loss, i)
                    writer.add_scalar('D/bin_loss', d_bin_loss, i)

                if i % 100 == 0:
                    self.G.eval()
                    self.sample_results(writer, i)
                    self.G.train()

                if i % self.args.bleu_iter == 0:
                    self.G.eval(), self.D.eval(), self.C.eval()
                    self.calculate_bleu(writer, i)
                    self.G.train(), self.D.train(), self.C.train()

                if i % resample_freq == 0 and i > 0 and self.args.update_latent:
                    resample_freq = len(self.dataset) // self.args.batch_size
                    self.update_latent(writer, i)

                pbar.update(1)


    def _test(self):
        print(">start test")
        from time import time

        # latent = self.G.sample(self.z_bins[:10, :], self.z_latents[:10, :], 10, 10)
        # samples = self.V.sample_text(latent)
        # logits, k_bins, latent = self.D(latent)
        # self.sample_results(None)
        self.pretrain_vae(100)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--check-iter', type=int, default=1000, help='checkpoint every 1k')
    parser.add_argument('--bleu-iter', type=int, default=400, help='bleu evaluation step')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)
    parser.add_argument('--tokenize', '-t', type=str, default='word', choices=['word', 'char'])
    parser.add_argument('--max-seq-len', type=int, default=56)

    parser.add_argument('--name', type=str, default='vae-gan')
    parser.add_argument('--gen-noise-dim', type=int, default=100)
    parser.add_argument('--gen-latent-dim', type=int, default=K_BINS)
    parser.add_argument('--block-dim', type=int, default=600)
    parser.add_argument('--gen-num-layers', type=int, default=5)
    parser.add_argument('--dis-num-layers', type=int, default=5)

    parser.add_argument('--temperature-min', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=cfg.temperature)

    parser.add_argument('--anneal-rate', type=float, default=0.00002)

    parser.add_argument('--vae-lr', type=float, default=0.00001)
    parser.add_argument('--gen-adv-lr', type=float, default=0.0001)
    parser.add_argument('--dis-lr', type=float, default=0.0004)
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
    trainer = VAE_GAN_Trainer(args)
    trainer.train()
    # trainer.calculate_bleu(None)
    # trainer.update_latent()
    # trainer.pretrain_generator(args.pretrain_epochs)
    # trainer.adv_train_generator(1)
    # trainer.adv_train_discriminator(1)

    # trainer._test()