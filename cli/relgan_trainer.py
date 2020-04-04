import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, glob, json
import config as cfg

from module.relgan_d import RelGAN_D
from module.relgan_g import RelGAN_G
from dataset import TextSubspaceDataset, seq_collate
from constant import Constants
from utils import get_fixed_temperature, get_losses
import numpy as np
from tensorboardX import SummaryWriter
from utils import gradient_penalty, str2bool
from nltk.translate.bleu_score import sentence_bleu


def data_iter(dataloader):
    def function():
        while True:
            for batch in dataloader:
                yield batch
    return function()

class RelGANTrainer():


    def __init__(self, args):
        self.dataset = TextSubspaceDataset(-1, 'data/kkday_dataset/train_title.txt', prefix='train_title', embedding=None, 
            max_length=args.max_seq_len, force_fix_len=args.grad_penalty or args.full_text)
        dataset = self.dataset
        self.dataloader = data_iter(torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size, shuffle=True))

        self.pretrain_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=args.batch_size*4, shuffle=True)

        self.G = RelGAN_G(args.mem_slots, args.num_heads, args.head_size, args.gen_embed_dim, 
            args.gen_hidden_dim, dataset.vocab_size,
            max_seq_len=args.max_seq_len-1, padding_idx=Constants.PAD, gpu=True, model_type=args.gen_model_type)
        self.D = RelGAN_D(args.dis_embed_dim, args.max_seq_len-1, args.num_rep, dataset.vocab_size, Constants.PAD,
            gpu=True, dropout=0.25)
        self.G.cuda()
        self.D.cuda()
        args.vocab_size = dataset.vocab_size
        
        self.args = args
        self.args.name = args.name + '-' + args.gen_model_type
        self.gen_opt  = optim.Adam(self.G.parameters(), lr=args.gen_lr)
        self.gen_adv_opt  = optim.Adam(self.G.parameters(), lr=args.gen_adv_lr)
        self.dis_opt  = optim.Adam(self.D.parameters(), lr=args.dis_lr)

        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()

        self.data_iterator = data_iter(self.dataloader)


    def pretrain_generator(self, epochs, writer=None):
        iter_ = 0
        for epoch in range(epochs):
            print('Epoch '+ str(epoch))
            with tqdm(total=len(self.pretrain_dataloader), dynamic_ncols=True) as pbar:
                for batch in self.pretrain_dataloader:
                    inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                    inputs, target = inputs.cuda(), target.cuda()
                    batch_size = inputs.shape[0]
                    hidden = self.G.init_hidden(batch_size=batch_size)
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

            torch.save(self.G.state_dict(), 'save/{}-relgan_G_pretrained.pt'.format(self.args.gen_model_type))


    def adv_train_generator(self, g_step=1):
        total_loss = 0
        for step in range(g_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            batch_size = inputs.shape[0]

            real_samples = F.one_hot(target, args.vocab_size).float()
            gen_samples = self.G.sample(batch_size, batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            d_out_real = self.D(real_samples)
            d_out_fake = self.D(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, self.args.loss_type)

            self.gen_adv_opt.zero_grad()
            g_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), cfg.clip_norm)
            self.gen_adv_opt.step()

            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step=1):
        total_loss = 0
        for step in range(d_step):
            batch = next(self.data_iterator)
            inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
            batch_size = inputs.shape[0]
            real_samples = F.one_hot(target, args.vocab_size).float()
            with torch.no_grad():
                gen_samples = self.G.sample(batch_size, batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # ===Train===
            d_out_real = self.D(real_samples)
            d_out_fake = self.D(gen_samples.detach())
            _, d_loss = get_losses(d_out_real, d_out_fake, self.args.loss_type)

            if self.args.grad_penalty:
                d_gp_loss = gradient_penalty(self.D, real_samples, gen_samples.detach())
                d_loss += self.args.gp_weight*d_gp_loss

            self.dis_opt.zero_grad()
            d_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), cfg.clip_norm)
            self.dis_opt.step()

            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def sample_results(self, writer, step=0):
        results = self.G.sample(10, 10)
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

    def calculate_bleu(self, writer, step=0, size=2000):
        '''
            writer: tensorboardX writer
            step: which iteration step
            size: compare a total of 1k sentences
        '''
        eval_dataloader = torch.utils.data.DataLoader(self.dataset, num_workers=4,
                        collate_fn=seq_collate, batch_size=20, shuffle=False)
        sentences, references = [], []
        scores_weights = { str(gram): [1/gram] * gram for gram in range(1, 4)  }
        scores = { str(gram): 0 for gram in range(1, 4)  }
        # print('Evaluate bleu scores', scores)
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs, target = batch['seq'][:, :-1], batch['seq'][:, 1:]
                batch_size = inputs.shape[0]
                results = self.G.sample(batch_size, batch_size)

                for idx, sent_token in enumerate(batch['seq'][:, 1:]):
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
                        scores[key] += sentence_bleu([reference], sentence, weights)
                if len(sentences) > size:
                    break

        with open(os.path.join(self.save_path, '{}_reference.txt'.format(step)), 'w') as f:
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
        # self.sample_results(None)
        

    def train(self):
        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = 'save/{}-{}'.format(self.args.name, cur_time)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(vars(self.args), f)
        writer = SummaryWriter('logs/{}-{}'.format(self.args.name, cur_time))
        print('Pretrain stage....')
        if args.pretrain_gen is not None:
            gen_dict = torch.load(args.pretrain_gen)
            self.G.load_state_dict(gen_dict)
        else:
            self.pretrain_generator(args.pretrain_epochs, writer=writer)

        with tqdm(total=args.iterations, dynamic_ncols=True) as pbar:
            for i in range(args.iterations):
                self.D.train()
                d_loss = self.adv_train_discriminator(d_step=self.args.dis_steps)
                self.G.train()
                g_loss = self.adv_train_generator(g_step=self.args.gen_steps)

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
                    writer.add_scalar('G/loss', g_loss, i)
                    writer.add_scalar('D/loss', d_loss, i)
                    writer.add_scalar('G/temp', self.G.temperature, i)

                if i % self.args.bleu_iter == 0:
                    curr_temp = self.G.temperature
                    self.G.temperature = 1.0
                    self.G.eval()
                    self.calculate_bleu(writer, i)
                    self.G.temperature = curr_temp
                    self.G.train()

                pbar.update(1)
                if i % 100 == 0:
                    curr_temp = self.G.temperature
                    self.G.temperature = 1.0
                    self.G.eval()
                    self.sample_results(writer, i)
                    self.G.temperature = curr_temp
                    self.G.train()

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
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--check-iter', type=int, default=1000, help='checkpoint every 1k')
    parser.add_argument('--bleu-iter', type=int, default=500, help='bleu evaluation step')
    parser.add_argument('--pretrain-gen', type=str, default=None)
    parser.add_argument('--gen-steps', type=int, default=1)
    parser.add_argument('--dis-steps', type=int, default=1)

    parser.add_argument('--name', type=str, default='relgan')

    parser.add_argument('--head-size', type=int, default=256)
    parser.add_argument('--mem-slots', type=int, default=1)

    parser.add_argument('--num-heads', type=int, default=2)

    parser.add_argument('--gen-embed-dim', type=int, default=64)
    parser.add_argument('--gen-hidden-dim', type=int, default=128)
    parser.add_argument('--dis-embed-dim', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=50)
    parser.add_argument('--num-rep', type=int, default=64)

    parser.add_argument('--temperature-min', type=float, default=0.1)
    parser.add_argument('--anneal-rate', type=float, default=0.00002)
    parser.add_argument('--gen-model-type', type=str, default='LSTM',
         choices=['LSTM', 'RMC'], help='Generator module type')

    parser.add_argument('--gen-lr', type=float, default=0.0001)
    parser.add_argument('--gen-adv-lr', type=float, default=0.0001)
    parser.add_argument('--dis-lr', type=float, default=0.0003)
    parser.add_argument('--grad-penalty', type=str2bool, nargs='?',
                        default=False, help='Apply gradient penalty')
    parser.add_argument('--full-text', type=str2bool, nargs='?',
                        default=False, help='Dataset return full max length')

    parser.add_argument('--gp-weight', type=float, default=10)

    parser.add_argument('--loss-type', type=str, default='rsgan', choices=['rsgan', 'wasstestein', 'hinge'])

    args = parser.parse_args()
    trainer = RelGANTrainer(args)
    # trainer.pretrain_generator(args.pretrain_epochs)
    trainer.train()
    # trainer._test()