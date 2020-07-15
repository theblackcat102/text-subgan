import torch
import torch.nn as nn
import pickle
from module.seq2seq import Encoder, Decoder, LuongAttention
from constant import Constants
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device='cuda'):
    y = logits + sample_gumbel(logits.size(), device=device)
    return torch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, device='cuda'):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class VAE_Gumbel(nn.Module):

    def __init__(self, embedding, latent_dim=32, categorical_dim=10, enc_hidden=64, 
        embedding_dim=32, dec_hidden=64, enc_layers=1, enc_dropout=0.1, dec_layers=1, dec_dropout=0.1 ):
        super(VAE_Gumbel, self).__init__()

        self.encoder = Encoder(embedding_dim, enc_hidden, 
            enc_layers, enc_dropout, True, cell=nn.LSTM)
        self.decoder = Decoder(embedding.embedding_dim, embedding.num_embeddings, 
            hidden_size=dec_hidden, num_layers=dec_layers, dropout=dec_dropout, 
                st_mode=False, cell=nn.LSTM)
        self.num_layers = dec_layers

        self.embedding = embedding
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.hidden2latent = nn.Sequential(
            nn.BatchNorm1d(enc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(enc_hidden, latent_dim*categorical_dim),
        )

        self.cell = nn.LSTM

        self.latent2hidden = nn.Sequential(
            nn.BatchNorm1d(latent_dim*categorical_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim*categorical_dim,  dec_hidden),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode(self, inputs, device='cuda', temperature=1, st_mode=False):
        inputs = self.embedding(inputs)
        output, hidden = self.encoder(inputs)
        latent = self.hidden2latent(hidden)
        latent_y = latent.view(latent.size(0), self.latent_dim, self.categorical_dim)
        z = gumbel_softmax(latent_y, temperature, st_mode, device=device).view(-1, self.latent_dim * self.categorical_dim)
        return output, z, latent

    def decode(self, latent, max_length, device='cuda'):
        # [B x 1]
        inputs = torch.ones(latent.size(0), 1).fill_(Constants.BOS).long()
        # [B x 1 x embedding_dim]
        inputs = self.embedding(inputs.to(device))
        # [B x dim_v]
        hidden = self.latent2hidden(latent)
        hidden = hidden.unsqueeze(0).expand(self.num_layers, -1, -1)
        hidden = hidden.contiguous()
        if self.cell == nn.LSTM:
            hidden = (hidden, hidden)
        outputs = []
        inp = []        

        for _ in range(max_length):
            output, hidden = self.decoder(
                inputs, hidden, None, None)
            output = self.log_softmax(output)
            outputs.append(output)

            # argmax method
            next_tokens = torch.argmax(output, dim=-1)

            inp.append(next_tokens.squeeze(0))
            inputs = self.embedding(next_tokens)

        outputs = torch.cat(outputs, dim=1)
        inp = torch.stack(inp).squeeze(-1).transpose(0,1)

        return outputs, inp
    

    def forward(self, inputs, max_length, temperature=1, st_mode=False, device='cuda'):
        encoder_output, z, latent = self.encode(inputs, device=device, 
            st_mode=st_mode, temperature=temperature)
        decoder_output, inp = self.decode(z, max_length, device=device)
        latent_y = latent.view(latent.size(0), self.latent_dim, self.categorical_dim)
        return decoder_output, torch.softmax(latent_y, dim=-1).reshape(*latent.size()), inp



if __name__ == "__main__":
    from dataset import KKDayUser, seq_collate
    dataset = KKDayUser(-1, 'data/kkday_dataset/user_data', 
        'data/kkday_dataset/matrix_factorized_64.pkl',
        prefix='item_graph', embedding=None, max_length=128, 
        token_level='word', is_train=True)
    val_dataset = KKDayUser(-1, 'data/kkday_dataset/user_data', 
        'data/kkday_dataset/matrix_factorized_64.pkl',
        prefix='item_graph', embedding=None, max_length=128, 
        token_level='word', is_train=False)

    # 32 latent feature x 10 category
    categorical_dim = 10
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.embedding = nn.Embedding(dataset.vocab_size, embedding_dim=32)
            self.vae = VAE_Gumbel(self.embedding, latent_dim=8, categorical_dim=categorical_dim)

    model = Model().cuda()
    pretrain_dataloader = torch.utils.data.DataLoader(dataset, num_workers=6,
                        collate_fn=seq_collate, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=6,
                        collate_fn=seq_collate, batch_size=48, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    nll_criterion = nn.NLLLoss(ignore_index=Constants.PAD)

    iter_ = 0
    max_temp = 1.0
    temp_min = 0.00005
    temp = 1.0
    N = len(pretrain_dataloader) * 10
    anneal_rate = max_temp / N

    for e in range(10):
        for batch in pretrain_dataloader:
            inputs, target = batch['tmp'][:, :-1], batch['tmp'][:, 1:]
            title = batch['tgt'][:, :-1]
            title = title.cuda()
            inputs = inputs.cuda()
            target = target.cuda()
            decoder_output, latent, inp = model.vae(inputs, max_length=inputs.shape[1], temperature=temp)


            nll_loss = nll_criterion(decoder_output.view(-1, dataset.vocab_size),
                target.flatten())

            if iter_ % 100 == 1:
                temp = np.maximum(temp * np.exp(-anneal_rate * iter_), temp_min)

            log_ratio = torch.log(latent * categorical_dim + 1e-20)
            kl_loss = torch.sum(latent * log_ratio, dim=-1).mean()
            optimizer.zero_grad()
            loss = kl_loss + nll_loss
            loss.backward()
            optimizer.step()

            decoder_output, latent, inp = model.vae(title, max_length=title.shape[1], temperature=temp)
            nll_loss = nll_criterion(decoder_output.view(-1, dataset.vocab_size),
                target.flatten())
            if iter_ % 100 == 1:
                temp = np.maximum(temp * np.exp(-anneal_rate * iter_), temp_min)
            log_ratio = torch.log(latent * categorical_dim + 1e-20)
            kl_loss = torch.sum(latent * log_ratio, dim=-1).mean()
            optimizer.zero_grad()
            loss = kl_loss + nll_loss
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                print('loss: {:.4f}'.format(loss.item()))

    
            if iter_ % 1000 == 0 and iter_ != 0:
                model.eval()
                with torch.no_grad():
                    print('sample latent')
                    clusters = {}
                    for batch in val_dataloader:
                        inputs, target = batch['tmp'][:, :-1], batch['tmp'][:, 1:]
                        inputs = inputs.cuda()
                        target = target.cuda()
                        decoder_output, latents, inp = model.vae(inputs, max_length=inputs.shape[1], temperature=temp)
                        for idx, sent in enumerate(batch['tmp'][:, 1:]):
                            sentence = []
                            for token in sent:
                                if token.item() == Constants.EOS:
                                    break
                                sentence.append( dataset.idx2word[token.item()])
                            sent = ' '.join(sentence)
                            latent = latents[idx].cpu().detach().numpy()

                            clusters[sent] = latent

                    with open(f'logs/vae_gumbel/cluster_{iter_}.pkl', 'wb') as f:
                        pickle.dump(clusters, f)

                    torch.save(model, f'logs/vae_gumbel/model_{iter_}.pt')
                model.train()
            
            iter_ += 1
