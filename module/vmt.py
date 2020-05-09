import torch
import torch.nn as nn
from module.vae_gumbel import VAE_Gumbel
from module.vae import VariationalDecoder, VariationalEncoder, GaussianKLLoss
from module.seq2seq import LuongAttention
from constant import Constants

from module.discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [64, 64, 64, 64]


class TemplateD(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size,padding_idx, latent_dim, gpu=False, dropout=0.25):
        super(TemplateD, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single))) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
        pred = self.feature2out(self.dropout(pred))
        return pred

class VariationalAutoEncoder(nn.Module):
    def  __init__(self, embeddding, embed_dim=64, enc_hidden_size=256, dec_hidden_size=256, # latent dim
                latent_dim=512, max_seq_len=50,
                 num_layers=2, dropout=0.1, bidirectional=False, cell=nn.GRU, gpu=True,
                 st_mode=False):
        super().__init__()
        self.gpu = gpu
        self.max_seq_len = max_seq_len
        self.dec_hidden_size = dec_hidden_size
        self.embedding = embeddding
        self.latent2hidden = nn.Linear(latent_dim, dec_hidden_size)

        self.encoder = VariationalEncoder(self.embedding, latent_dim, enc_hidden_size, 
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, cell=cell)

        self.decoder = VariationalDecoder(self.embedding, dec_hidden_size,
            num_layers=num_layers, dropout=dropout, st_mode=False, cell=cell)
        self.gpu = gpu

        if gpu:
            self.decoder = self.decoder.cuda()
            self.encoder = self.encoder.cuda()


    def forward(self, inputs, max_length, device='cpu'):
        if self.gpu:
            device = 'cuda'
        outputs, hidden, mean, std = self.encoder(inputs, device=device)
        latent = self.latent2hidden(hidden)
        de_outputs, _ = self.decoder(latent, None, max_length=max_length, device=device)
        return outputs, latent, mean, std, de_outputs


class VMT(nn.Module):
    def __init__( self, embedding_dim, vocab_size, max_seq_len, padding_idx=Constants.PAD,
            tmp_hidden_size=64, tmp_dec_hidden_size=64, tmp_latent_dim=16, tmp_category=10, desc_latent_dim=87, user_latent_dim=128,
            enc_hidden_size=128, enc_layers=2, enc_bidirect=True, dropout=0.1, dec_layers=2, dec_hidden_size=128, 
            attention=True, 
            gpu=False):
        super(VMT, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.template_vae = VAE_Gumbel(self.embedding, embedding_dim=embedding_dim, 
            enc_hidden=tmp_hidden_size, categorical_dim=tmp_category, dec_hidden=tmp_dec_hidden_size, 
            latent_dim=tmp_latent_dim)

        self.content_encoder = VariationalEncoder(self.embedding, desc_latent_dim, enc_hidden_size, 
            num_layers=enc_layers, dropout=dropout, bidirectional=enc_bidirect, cell=nn.LSTM, gpu=gpu)

        self.attention = LuongAttention(dec_hidden_size, dec_hidden_size)

        self.title_decoder = VariationalDecoder(self.embedding, dec_hidden_size,
            num_layers=dec_layers, dropout=dropout, st_mode=False, cell=nn.LSTM, attention=self.attention)

        self.merge_proj = nn.Linear(desc_latent_dim+(tmp_latent_dim*tmp_category)+user_latent_dim, dec_hidden_size)

        self.max_seq_len = max_seq_len
        self.user_latent_dim = user_latent_dim
        self.vocab_size = vocab_size
        self.softmax = nn.LogSoftmax(dim=-1)
        self.kl_loss_ = GaussianKLLoss()
        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.gpu = gpu

    def cycle_template(self, src, output, temperature=1):
        decoder_output, latent, inp = self.template_vae(src, max_length=output.shape[1], temperature=temperature)
        nll_loss = self.mle_criterion(decoder_output.view(-1, self.vocab_size),
                output.flatten())
        log_ratio = torch.log(latent * self.template_vae.categorical_dim + 1e-20)
        kl_loss = torch.sum(latent * log_ratio, dim=-1).mean()
        return nll_loss, kl_loss

    def encode_desc(self, desc, device='cpu'):
        if self.gpu:
            device = 'cuda'
        outputs, hidden, mean, std = self.content_encoder(desc,  device=device)
        return outputs, hidden, mean, std

    def encode_tmp(self, template, device='cpu'):
        if self.gpu:
            device = 'cuda'
        encoder_output, latent = self.template_vae.encode(template, device=device)
        return encoder_output, latent

    def decode(self, tmp_latent, desc_latent, user_feature, desc_outputs, max_length, device='cpu', temperature=None):
        if self.gpu:
            device = 'cuda'

        latent = self.merge_proj(torch.cat([ tmp_latent, desc_latent, user_feature ], axis=1))
        output_feat, output_logits  = self.title_decoder(latent, desc_outputs, max_length, device=device, temperature=temperature)
        return output_feat, output_logits

if __name__ == "__main__":
    import torch.nn.functional as F
    inputs = torch.randint(0, 1800, (32, 128)).long()
    user_feat = torch.randn((32, 128)).float()
    vmt = VMT(128, 3200, 128,  tmp_latent_dim=89, desc_latent_dim=87,
        max_seq_len=128, gpu=False)

    desc_outputs, desc_latent, mean, std = vmt.encode_desc(inputs, 'cpu')
    print('description: ',desc_latent.shape)
    print('template: ',tmp_latent.shape)
    outputs, tmp_latent, mean, std = vmt.encode_tmp(inputs, 'cpu')
    output_feat = vmt.decode(tmp_latent, desc_latent, user_feat, desc_outputs)

    nll_loss, kl_loss = vmt.cycle_template(inputs, inputs)
    print(desc_outputs.shape)
