import torch
import torch.nn as nn

from module.seq2seq import Encoder, Decoder, LuongAttention
from constant import Constants

import torch.nn.functional as F
class VariationalEncoder(nn.Module):
    def __init__(self, embedding, latent_dim, hidden_size,
                 num_layers=1, dropout=0.1, bidirectional=False, cell=nn.GRU, gpu=True):
        super().__init__()
        self.encoder = Encoder(
            embedding.embedding_dim, hidden_size, num_layers, dropout,
            bidirectional, cell)
        self.embedding = embedding
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.direction_num = 2 if bidirectional else 1
        self.cell = cell
        self.gpu = gpu
        self.hidden2mean = nn.Linear(hidden_size, latent_dim)
        self.hidden2std = nn.Linear(hidden_size, latent_dim)

    def forward(self, inputs, device):
        # [B x T x embedding_dim]
        inputs = self.embedding(inputs)
        # [1 x B x hidden_size]
        outputs, hidden = self.encoder(inputs, None)
        mean = self.hidden2mean(hidden)
        std = self.hidden2std(hidden)
        latent = torch.randn([inputs.size(0), self.latent_dim]).to(device)
        latent = latent * std + mean
        return outputs, latent, mean, std
    


class VariationalDecoder(nn.Module):
    def __init__(self, embedding, hidden_size,
                 num_layers=1, dropout=0.1, st_mode=False, cell=nn.GRU, attention=None):
        super().__init__()
        self.decoder = Decoder(
            embedding.embedding_dim, embedding.num_embeddings,
            hidden_size, num_layers, dropout, st_mode, cell,
            attention=attention)
        self.embedding = embedding
        self.vocab_size = embedding.num_embeddings
        self.num_layers = num_layers
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.cell = cell

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=True):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())

        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t


    def forward(self, latent, encoder_outputs, max_length,
                device, temperature=1, gumbel=False):
        """
        Args:
            latent: float tensor, shape = [B x latent_dim]
            encoder_outputs: float tensor, shape = [B x T x H]
            max_length: int
            temperature: float
        """
        # [B x 1]
        inputs = torch.ones(latent.size(0), 1).fill_(Constants.BOS).long()
        # [B x 1 x embedding_dim]
        inputs = self.embedding(inputs.to(device))
        # [B x dim_v]
        hidden = latent
        hidden = hidden.unsqueeze(0).expand(self.num_layers, -1, -1)
        hidden = hidden.contiguous()
        if self.cell == nn.LSTM:
            hidden = (hidden, hidden)
        outputs = []
        inp = []

        if gumbel:
            batch_size = latent.size(0)
            all_preds = torch.zeros(batch_size, max_length, self.vocab_size).to(device)

        gumbel_outputs = []
        for idx in range(max_length):
            output, hidden = self.decoder(
                inputs, hidden, encoder_outputs)

            gumbel_t = self.add_gumbel(output)
            pred = F.softmax(gumbel_t * temperature, dim=-1)  # batch_size * vocab_size
            if gumbel:
                all_preds[:, idx, :] = pred.squeeze(1)

            output = self.log_softmax(output)

            outputs.append(output)


            # multinomial sampling seems better than argmax method

            # next_tokens = []
            # for batch in range(output.shape[0]):
            #     next_token = torch.multinomial(torch.exp(output[batch]), 1)
            #     next_tokens.append(next_token)

            # # [Bx1]
            # next_tokens = torch.cat(next_tokens, dim=0)

            # argmax method
            next_tokens = torch.argmax(output, dim=-1)

            # print(next_tokens.shape)


            inp.append(next_tokens.squeeze(0))
            inputs = self.embedding(next_tokens)

        outputs = torch.cat(outputs, dim=1)
        inp = torch.stack(inp).squeeze(-1).transpose(0,1)
        if gumbel:
            return outputs, inp, all_preds

        return outputs, inp


class GaussianKLLoss(nn.Module):
    def forward(self, mean, std):
        loss = 0.5 * (mean.pow(2) + std.pow(2) - torch.log(std.pow(2)) - 1)
        return torch.mean(loss)


class VariationalAutoEncoder(nn.Module):
    def  __init__(self, vocab_size, embed_dim=64, enc_hidden_size=256, dec_hidden_size=256, # latent dim
                latent_dim=512, max_seq_len=50,
                k_bins=12, bin_latent_dim=100,noise_dim=100,
                 num_layers=2, dropout=0.1, bidirectional=False, cell=nn.GRU, gpu=True,
                 st_mode=False):
        super().__init__()
        self.gpu = gpu
        self.max_seq_len = max_seq_len
        self.dec_hidden_size = dec_hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.latent2hidden = nn.Linear(latent_dim, dec_hidden_size)

        self.encoder = VariationalEncoder(self.embedding, latent_dim, enc_hidden_size, 
            k_bins=k_bins, bin_latent_dim=bin_latent_dim,noise_dim=noise_dim, 
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, cell=cell).cuda()

        self.decoder = VariationalDecoder(self.embedding, dec_hidden_size,
            num_layers=num_layers, dropout=dropout, st_mode=False, cell=cell).cuda()

    def forward(self, inputs, device, max_length):
        outputs, hidden, mean, std = self.encoder(inputs, device=device)
        latent = self.latent2hidden(hidden)
        de_outputs = self.decoder(latent, None, max_length=max_length, device=device)
        return outputs, latent, mean, std, de_outputs
    
    def encode(self, kbins, latent, inputs, device):
        outputs, hidden, mean, std = self.encoder(inputs,(kbins, latent), device=device)
        latent = self.latent2hidden(hidden)
        return latent

    def sample_text(self, latents, start_letter=Constants.BOS, device='cuda'):
        batch_size = latents.shape[0]
        latent_dim = latents.shape[1]
        assert latent_dim == self.dec_hidden_size

        num_samples = batch_size

        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        # Generate sentences with multinomial sampling strategy
        inp = torch.LongTensor([start_letter] * batch_size)
        hidden = latents
        if self.gpu:
            inp = inp.cuda()
            hidden = hidden.cuda()

        outputs = self.decoder(latents, None, self.max_seq_len, device)
        samples = torch.argmax(outputs, dim=2)
        samples = samples[:num_samples]

        return samples




if __name__ == "__main__":

    vae = VariationalAutoEncoder(vocab_size=1000).cuda()
    inputs = torch.randint(0, 300, (10, 20)).cuda()
    kbins = torch.randn((10, 12)).cuda()
    latent = torch.randn((10, 100)).cuda()
    outputs, latent, mean, std, de_outputs = vae(kbins, latent, inputs, device='cuda')
    print(latent.shape, de_outputs.shape)