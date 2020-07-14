import torch
import torch.nn as nn
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from module.conv import ResBlock
import config as cfg

from utils import truncated_normal_

class CNNDiscriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False,
                 dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


class GRUDiscriminator(nn.Module):

    def __init__(self, embedding_dim, vocab_size, hidden_dim, feature_dim, max_seq_len, padding_idx,
                 gpu=False, dropout=0.2):
        super(GRUDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, feature_dim)
        self.feature2out = nn.Linear(feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp):
        """
        Get final feature of discriminator
        :param inp: batch_size * seq_len
        :return pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        hidden = self.init_hidden(inp.size(0))

        emb = self.embeddings(input)  # batch_size * seq_len * embedding_dim
        emb = emb.permute(1, 0, 2)  # seq_len * batch_size * embedding_dim
        _, hidden = self.gru(emb, hidden)  # 4 * batch_size * hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size * 4 * hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))  # batch_size * 4 * hidden_dim
        feature = torch.tanh(out)  # batch_size * feature_dim

        return feature

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


# Classifier
class CNNClassifier(CNNDiscriminator):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, filter_sizes, num_filters, padding_idx,
                 gpu=False, dropout=0.25):
        super(CNNClassifier, self).__init__(embed_dim, vocab_size, filter_sizes, num_filters, padding_idx,
                                            gpu, dropout)

        self.k_label = k_label
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])  # vanilla
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
        #     zip(num_filters, filter_sizes)
        # ])  # RelGAN

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, k_label)  # vanilla
        # self.out2logits = nn.Linear(num_rep * 100, k_label) # RelGAN
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        # vanilla
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        # RelGAN
        # cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        # pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]

        pred = torch.cat(pools, 1)  # batch_size * feature_dim
        # pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)    # RelGAN
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway, same dim

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(self.dropout(pred)).squeeze(1)  # vanilla, batch_size * k_label
        # logits = self.out2logits(self.dropout(pred.view(inp.size(0), -1))).squeeze(1)  # RelGAN, batch_size * k_label

        return logits


class CNNClassifierModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, k_label=1):
        super(CNNClassifierModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.block = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, k_label)
        )

    def forward(self, inputs, is_discrete=False):
        """
            inputs: float tensor, shape = [B, T, vocab_size]
        """
        if is_discrete:
            inputs = self.embedding(inputs)
        else:
            batch_size, length, vocab_size = inputs.size()
            inputs = inputs.contiguous().view(-1, vocab_size)
            inputs = torch.mm(inputs, self.embedding.weight)
            inputs = inputs.view(batch_size, length, -1)
        inputs = inputs.transpose(1, 2)     # (B, H, T)
        outputs = self.block(inputs)        # (B, H, T)
        outputs = self.maxpool(outputs)     # (B, H, 1)
        outputs = outputs.squeeze(-1)
        outputs = self.linear(outputs)      # (B, 1)
        return outputs


class RNNClassifier(nn.Module):
    def __init__(self, embedding, hidden_size, dropout=0.1, num_layers=1,
                 bidirectional=True, cell=nn.GRU):
        super(RNNClassifier, self).__init__()
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = cell(
            embedding.embedding_dim, hidden_size,
            dropout=(0 if num_layers == 1 else dropout),
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, is_discrete=False):
        """
        Args:
            inputs: float tensor, shape = [B x T x vocab_size], probility
            is_discrete: boolean, if True, the inputs shape is [B x T]

        Returns:
            loss: float tensor, scalar, binomial probility
        """
        if is_discrete:
            inputs = self.embedding(inputs)
        else:
            batch_size, length, vocab_size = inputs.size()
            inputs = inputs.contiguous().view(-1, vocab_size)
            inputs = torch.mm(inputs, self.embedding.weight)
            inputs = inputs.view(batch_size, length, -1)
        _, hidden = self.rnn(inputs)
        if type(hidden) is tuple:
            hidden = hidden[0]
        hidden = hidden.view(
            self.num_layers, self.num_directions, -1, self.hidden_size)
        if self.num_directions == 2:
            outputs = torch.cat(
                [hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=-1)
        else:
            outputs = hidden[-1, -1, :, :]
        outputs = self.linear(outputs)
        outputs = self.sigmoid(outputs)
        return outputs


class CNNCritic(nn.Module):
    def __init__(self, embedding, max_length):
        super(CNNCritic, self).__init__()
        self.embedding = embedding
        self.block = nn.Sequential(
            ResBlock(embedding.embedding_dim),
            ResBlock(embedding.embedding_dim),
            ResBlock(embedding.embedding_dim),
            ResBlock(embedding.embedding_dim),
            ResBlock(embedding.embedding_dim),
        )
        self.maxpool = nn.MaxPool1d(max_length)
        self.linear = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, inputs, is_discrete=False):
        """
            inputs: float tensor, shape = [B, T, vocab_size]
        """
        if is_discrete:
            inputs = self.embedding(inputs)
        else:
            batch_size, length, vocab_size = inputs.size()
            inputs = inputs.contiguous().view(-1, vocab_size)
            inputs = torch.mm(inputs, self.embedding.weight)
            inputs = inputs.view(batch_size, length, -1)
        inputs = inputs.transpose(1, 2)     # (B, H, T)
        outputs = self.block(inputs)        # (B, H, T)
        outputs = self.maxpool(outputs)     # (B, H, 1)
        outputs = outputs.squeeze(-1)       # (B, H)
        outputs = self.linear(outputs)      # (B, 1)
        return outputs

from module.block import Block


class VAE_Discriminator(nn.Module):

    def __init__(self, k_bins=20, n_layers=5, block_dim=600, gpu=True):
        super().__init__()
        self.block_dim = block_dim
        self.gpu = gpu

        self.net = nn.Sequential(
            *[Block(block_dim) for _ in range(n_layers)]
        )
        self.k_bins = nn.Linear(block_dim, k_bins)
        self.logits = nn.Linear(block_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        latent = self.net(x)
        latent = self.relu(latent)
        return self.logits(latent), self.k_bins(latent), latent