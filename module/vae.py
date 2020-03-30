import torch
import torch.nn as nn

from transfer.module.seq2seq import Encoder, Decoder, LuongAttention
from transfer.constant import Constants


class VariationalEncoder(nn.Module):
    def __init__(self, embedding, latent_dim, hidden_size,
                 num_layers=1, dropout=0.1, bidirectional=False, cell=nn.GRU):
        super().__init__()
        self.encoder = Encoder(
            embedding.embedding_dim, hidden_size, num_layers, dropout,
            bidirectional, cell)
        self.embedding = embedding
        self.style_embedding = nn.Embedding(2, hidden_size)
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.direction_num = 2 if bidirectional else 1
        self.cell = cell
        self.hidden2mean = nn.Linear(hidden_size, latent_dim)
        self.hidden2std = nn.Linear(hidden_size, latent_dim)

    def forward(self, inputs, style, device):
        # [B x T x embedding_dim]
        inputs = self.embedding(inputs)
        # [1 x B x hidden_size]
        hidden = self.style_embedding(style.view(-1)).unsqueeze(0)
        # [(num_layers x direction_num) x B x hidden_size]
        hidden = hidden.expand(self.num_layers * self.direction_num, -1, -1)
        hidden = hidden.contiguous()
        if self.cell == nn.LSTM:
            hidden = (hidden, hidden)
        outputs, hidden = self.encoder(inputs, hidden)
        mean = self.hidden2mean(hidden)
        std = self.hidden2std(hidden)
        latent = torch.randn([inputs.size(0), self.latent_dim]).to(device)
        latent = latent * std + mean
        return outputs, latent, mean, std


class VariationalDecoder(nn.Module):
    def __init__(self, embedding, style_dim, latent_dim, hidden_size,
                 num_layers=1, dropout=0.1, st_mode=False, cell=nn.GRU):
        super().__init__()
        self.decoder = Decoder(
            embedding.embedding_dim, embedding.num_embeddings,
            hidden_size, num_layers, dropout, st_mode, cell,
            LuongAttention(hidden_size, hidden_size, 'general'))
        self.embedding = embedding
        self.style_embedding = nn.Embedding(2, style_dim)
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.cell = cell
        self.latent2hidden = nn.Linear(
            latent_dim + style_dim, hidden_size)

    def forward(self, latent, style, encoder_outputs, max_length,
                device, temperture=None, embed_style=None):
        """
        Args:
            latent: float tensor, shape = [B x latent_dim]
            style: long tensor, shape = [B]
            encoder_outputs: float tensor, shape = [B x T x H]
            max_length: int
            temperture: float
        """
        # [B x 1]
        inputs = torch.ones(latent.size(0), 1).fill_(Constants.BOS).long()
        # [B x 1 x embedding_dim]
        inputs = self.embedding(inputs.to(device))
        # [B x dim_v]
        if embed_style is None:
            embed_style = self.style_embedding(style.view(-1))
        else:
            embed_style = torch.from_numpy(embed_style).float().to(device)
        hidden = self.latent2hidden(torch.cat([latent, embed_style], dim=-1))
        hidden = hidden.unsqueeze(0).expand(self.num_layers, -1, -1)
        hidden = hidden.contiguous()
        if self.cell == nn.LSTM:
            hidden = (hidden, hidden)
        outputs = []
        for _ in range(max_length):
            output, hidden = self.decoder(
                inputs, hidden, encoder_outputs, temperture)
            outputs.append(output)
            topi = torch.argmax(output, dim=-1)
            inputs = self.embedding(topi.detach())
        outputs = torch.cat(outputs, dim=1)
        return outputs


class GaussianKLLoss(nn.Module):
    def forward(self, mean, std):
        loss = 0.5 * (mean.pow(2) + std.pow(2) - torch.log(std.pow(2)) - 1)
        return torch.mean(loss)