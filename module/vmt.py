import torch
import torch.nn as nn
from module.vae_gumbel import VAE_Gumbel
from module.vae import VariationalDecoder, VariationalEncoder, GaussianKLLoss
from module.seq2seq import LuongAttention
from constant import Constants
import torch.nn.functional as F
from module.discriminator import CNNDiscriminator
from module.biset import BiSET, MultiBiSET

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [64, 64, 64, 32]

class QHead(nn.Module):
    def __init__(self, dis_latent, category_dim,latent_dim):
        super(QHead, self).__init__()
        self.feature_dim = dis_latent
        self.encode = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.feature2mean = nn.Linear(128, latent_dim)
        self.feature2var = nn.Linear(128, latent_dim)
        self.feature2cat = nn.Linear(128, category_dim)

    def forward(self, latent):
        mean_, vars_ = self.feature2mean(latent), self.feature2var(latent), 
        cat = self.feature2cat(latent)
        return mean_, vars_, cat

class ProductHead(nn.Module):
    def __init__(self, dis_latent, latent_dim):
        super(ProductHead, self).__init__()
        self.feature_dim = dis_latent
        self.feature2latent = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, latent):
        return self.feature2latent(latent)


class TemplateD(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, category_dim,  padding_idx=Constants.PAD, 
        to_latent=False, gpu=False, dropout=0.25):
        super(TemplateD, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        self.num_rep = num_rep

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.out2logits = nn.Linear(self.feature_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_params()
        self.to_latent = to_latent

        if to_latent:
            self.proj = nn.Linear(self.feature_dim, category_dim)

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
        logits = self.out2logits(pred)

        latent = torch.sum(pred.view(-1, self.num_rep, self.feature_dim), dim=1)
        if self.to_latent:
            latent = self.proj(latent)
        return logits, latent

class VariationalAutoEncoder(nn.Module):
    '''
        Not to be confused with VariationalAutoEncoder from vae.py, that one is bit more complicated
    '''
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
            attention=True, biset=True, user_embedding=False, user_size=-1,
            gpu=False):
        super(VMT, self).__init__()

        if user_embedding:
            self.user_embedding = nn.Embedding(user_size, user_latent_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.template_vae = VAE_Gumbel(nn.Embedding(vocab_size, embedding_dim), embedding_dim=embedding_dim, 
            enc_hidden=tmp_hidden_size, categorical_dim=tmp_category, dec_hidden=tmp_dec_hidden_size, 
            latent_dim=tmp_latent_dim)

        self.content_encoder = VariationalEncoder(self.embedding, desc_latent_dim, enc_hidden_size, 
            num_layers=enc_layers, dropout=dropout, bidirectional=enc_bidirect, cell=nn.LSTM, gpu=gpu)
        self.content_dropout = nn.Dropout(0.3)
        self.attention = None
        if attention:
            self.attention = LuongAttention(dec_hidden_size, dec_hidden_size)

        self.title_decoder = VariationalDecoder(self.embedding, dec_hidden_size,
            num_layers=dec_layers, dropout=dropout, st_mode=False, cell=nn.LSTM, attention=self.attention)

        self.merge_proj = nn.Linear(desc_latent_dim+(tmp_latent_dim*tmp_category)+user_latent_dim, dec_hidden_size)

        self.biset = None
        if biset:
            self.biset = BiSET(article_hidden_size=enc_hidden_size, template_hidden_size= tmp_hidden_size, att_type='general')

        self.max_seq_len = max_seq_len
        self.user_latent_dim = user_latent_dim
        self.vocab_size = vocab_size
        self.softmax = nn.LogSoftmax(dim=-1)
        self.kl_loss_ = GaussianKLLoss()
        self.mle_criterion = nn.NLLLoss(ignore_index=Constants.PAD)
        self.gpu = gpu

    def cycle_template(self, src, output, temperature=1):
        decoder_output, latent, latent_y = self.template_vae(src, max_length=output.shape[1], temperature=temperature)
        nll_loss = self.mle_criterion(decoder_output.view(-1, self.vocab_size),
                output.flatten())
        log_ratio = torch.log(latent_y * self.template_vae.categorical_dim + 1e-20)
        kl_loss = torch.sum(latent_y * log_ratio, dim=-1).mean()
        return nll_loss, kl_loss

    def encode_desc(self, desc, device='cpu'):
        if self.gpu:
            device = 'cuda'
        outputs, hidden, mean, std = self.content_encoder(desc,  device=device)
        return outputs, self.content_dropout(hidden), mean, std

    def encode_tmp(self, template, temperature=1,device='cpu'):
        if self.gpu:
            device = 'cuda'
        encoder_output, latent, _ = self.template_vae.encode(template, device=device, temperature=temperature)
        return encoder_output, latent

    def decode(self, tmp_latent, desc_latent, user_feature, desc_outputs, tmp_outputs, max_length, device='cpu', temperature=1, gumbel=False):
        if self.gpu:
            device = 'cuda'

        latent = self.merge_proj(torch.cat([ tmp_latent, desc_latent, user_feature ], axis=1))
        if self.biset != None:
            desc_outputs = self.biset(desc_outputs, tmp_outputs)

        if gumbel:
            output_feat, output_logits, one_hot  = self.title_decoder(latent, desc_outputs, max_length, device=device, temperature=temperature, gumbel=True)
            return output_feat, output_logits, one_hot
        output_feat, output_logits  = self.title_decoder(latent, desc_outputs, max_length, device=device, temperature=temperature)
        return output_feat, output_logits

if __name__ == "__main__":
    import torch.nn.functional as F
    inputs = torch.randint(0, 1800, (32, 32)).long().cuda()
    user_feat = torch.randn((32, 32)).float().cuda()
    vmt = VMT(128, 3200, 32,  tmp_latent_dim=10,desc_latent_dim=87,
        user_latent_dim=32,
        gpu=False, biset=True).cuda()

    desc_outputs, desc_latent, mean, std = vmt.encode_desc(inputs, device='cuda')
    print('description: ',desc_latent.shape)
    outputs, tmp_latent = vmt.encode_tmp(inputs, device='cuda')
    print('template: ',tmp_latent.shape)
    output_feat = vmt.decode(tmp_latent, desc_latent, user_feat, desc_outputs, outputs, 
        max_length=32, device='cuda')

    nll_loss, kl_loss = vmt.cycle_template(inputs, inputs)
    print(desc_outputs.shape)
