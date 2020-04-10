import torch
import torch.nn as nn

from module.seq2seq import Encoder, Decoder, LuongAttention
from constant import Constants


class VariationalEncoder(nn.Module):
    def __init__(self, embedding, latent_dim, hidden_size,
                k_bins=12, bin_latent_dim=100,noise_dim=100,
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
        self.noise_dim = noise_dim
        self.latent_proj = nn.Linear(k_bins+bin_latent_dim+noise_dim, hidden_size)

    def forward(self, inputs, init_latent, device):
        # [B x T x embedding_dim]
        inputs = self.embedding(inputs)
        # [1 x B x hidden_size]
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(*init_latent, batch_size=batch_size)
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
    
    def init_hidden(self, kbins, latents, batch_size=64):
        noise = torch.randn(batch_size, self.noise_dim)
        if self.gpu:
            noise = noise.cuda()
        kbins = kbins
        latents = latents
        latent = torch.cat([noise, kbins, latents], axis=1)
        latent = self.latent_proj(latent)
        return latent


class VariationalDecoder(nn.Module):
    def __init__(self, embedding, hidden_size,
                 num_layers=1, dropout=0.1, st_mode=False, cell=nn.GRU):
        super().__init__()
        self.decoder = Decoder(
            embedding.embedding_dim, embedding.num_embeddings,
            hidden_size, num_layers, dropout, st_mode, cell,
            None)
        self.embedding = embedding
        self.num_layers = num_layers
        self.cell = cell

    def forward(self, latent, encoder_outputs, max_length,
                device, temperature=None):
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
        for _ in range(max_length):
            output, hidden = self.decoder(
                inputs, hidden, encoder_outputs, temperature)
            outputs.append(output)
            topi = torch.argmax(output, dim=-1)
            inputs = self.embedding(topi.detach())
        outputs = torch.cat(outputs, dim=1)
        return outputs


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

    def forward(self, kbins, latent, inputs, device, max_length):
        outputs, hidden, mean, std = self.encoder(inputs,(kbins, latent), device=device)
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