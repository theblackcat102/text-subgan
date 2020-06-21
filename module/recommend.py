import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import os, glob, json


class SimpleLM(nn.Module):
    def __init__(self, vocab_size, n_factors, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 
                                               embed_dim,
                                               padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, n_factors, num_layers=2, batch_first=True, dropout=0.2)
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        if len(inputs.shape) > 3:
            inputs = inputs.squeeze(2)
        output, hidden = self.rnn(inputs)
        return output


class SimpleMF(nn.Module):
    
    def __init__(self, user_size, vocab_size, n_factors, user_embedding=None, embed_dim=32):
        super().__init__()
        if user_embedding != None:
            self.user_factors = user_embedding
        else:
            self.user_factors = torch.nn.Embedding(user_size, 
                                               n_factors)

        self.user_offset = torch.nn.Embedding(user_size, 1)
        self.word_factors = SimpleLM(vocab_size, n_factors, embed_dim)

    def forward(self, user, item):
        sent_context = self.word_factors(item).mean(1)
        user_context = self.user_factors(user)

        return (user_context * sent_context ).sum(1) + self.user_offset(user).squeeze(1)



