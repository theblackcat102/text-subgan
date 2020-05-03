# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from module.generator import LSTMGenerator
from module.seq2seq import Encoder, LuongAttention
from module.relational_rnn_general import RelationalMemory

class RelGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 gpu=False, model_type='RMC'):
        super(RelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'

        self.temperature = 1.0  # init value is 1.0
        self.model_type = model_type
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if model_type == 'LSTM':
            # LSTM
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)
        else:
            # RMC
            self.hidden_dim = mem_slots * num_heads * head_size
            self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                         num_heads=num_heads, return_all_outputs=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()
        # pass

    def init_hidden(self, batch_size=cfg.batch_size):
        if self.model_type == 'LSTM':
            h = torch.zeros(1, batch_size, self.hidden_dim)
            c = torch.zeros(1, batch_size, self.hidden_dim)

            if self.gpu:
                return h.cuda(), c.cuda()
            else:
                return h, c
        else:
            """init RMC memory"""
            memory = self.lstm.initial_state(batch_size)
            memory = self.lstm.repackage_hidden(memory)  # detch memory at first
            return memory.cuda() if self.gpu else memory

    def step(self, inp, hidden):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o

    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t



class RelSpaceG(RelGAN_G):
    def __init__( self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                k_bins=5, latent_dim=100, noise_dim=100,
                 gpu=False):
        super(RelSpaceG, self).__init__(mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 gpu, 'LSTM')
        self.latent_proj = nn.Sequential(
            nn.Linear(noise_dim+latent_dim+k_bins, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.noise_dim = noise_dim

    def init_hidden(self, kbins, latents, batch_size=cfg.batch_size):
        noise = torch.randn(batch_size, self.noise_dim)
        if self.gpu:
            noise = noise.cuda()
        kbins = kbins
        latents = latents
        latent = torch.cat([noise, kbins, latents], axis=1)
        latents = self.latent_proj(latent).unsqueeze(0).repeat(2, 1, 1)
        h, c = latents[[0], :, :], latents[[1], :, :]
        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def sample(self, kbins, latents, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            hidden = self.init_hidden(kbins, latents, batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples


class RelGAN_Seq2Seq(RelGAN_G):
    def __init__( self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
            k_bins=5, latent_dim=100, noise_dim=100, enc_layers=2, enc_bidirect=True,
            attention=True,
            gpu=False):
        super(RelGAN_Seq2Seq, self).__init__(-1, -1, -1, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
            gpu, 'LSTM')
        self.encoder = Encoder( embedding_dim, hidden_dim,  num_layers=enc_layers, dropout=0.1, 
            bidirectional=enc_bidirect, cell=nn.LSTM )
        self.attention = LuongAttention(hidden_dim, hidden_dim)
        if attention:
            self.down_sample = nn.Linear(hidden_dim*2, hidden_dim)
        self.lstm2out = nn.Linear(hidden_dim, self.vocab_size, bias=False)



    def init_hidden(self, input_seq, batch_size=cfg.batch_size):
        inputs = self.embeddings(input_seq)
        outputs, hidden = self.encoder(inputs)
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        h, c = hidden[[0], :, :], hidden[[1], :, :]
        return outputs, (h, c)

    def forward(self, inp, hidden, encoder_outputs=None, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """

        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        if self.attention is not None:
            out, attn_weight = self.attention(out, encoder_outputs)
            out = out.contiguous().view(-1, self.hidden_dim*2)  # out: (batch_size * len) * hidden_dim
            out = self.down_sample(out)
        else:
            out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim

        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred        

    def step(self, inp, hidden, encoder_outputs=None):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)

        if self.attention is not None:
            out, attn_weight = self.attention(out, encoder_outputs)
            out = self.down_sample(out)

        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o


    def sample(self, inputs, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            encoder_outputs, hidden = self.init_hidden(inputs, batch_size)

            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step(inp, hidden, encoder_outputs)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples


if __name__ == "__main__":
    from utils import gradient_penalty
    # G = RelSpaceG(mem_slots=5, num_heads=12, head_size=16, embedding_dim=18, hidden_dim=18, vocab_size=3600, max_seq_len=128, padding_idx=0,
    #              gpu=False)#, module_type='LSTM')
    attention = LuongAttention(18, 18)
    G = RelGAN_Seq2Seq(embedding_dim=18, hidden_dim=18, vocab_size=3600, max_seq_len=128, padding_idx=0,
              gpu=True, attention=attention)#, module_type='LSTM')
    data = torch.randint(0, 3600, (32, 128))
    # latents = torch.randn(32, 100)
    # kbins = torch.randn(32, 5)
    # hidden = G.init_hidden(kbins, latents,batch_size=32)
    # # print(hidden.shape)
    # pred = G.forward(data, hidden, need_hidden=False)
    # print(pred.shape)
    
    inputs = torch.randint(0, 3600, (32, 128))
    G.cuda()
    inputs, data = inputs.cuda(), data.cuda()
    outputs, hidden = G.init_hidden(inputs, batch_size=32)
    pred = G.forward(data, hidden, need_hidden=False, encoder_outputs=outputs)
    G.sample(inputs, 32, 32)
