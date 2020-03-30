import torch
import torch.nn as nn

from transfer.module.conv import ResBlock


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(CNNCritic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.block = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )
        self.maxpool = nn.MaxPool1d(max_length)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

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
        outputs = self.sigmoid(outputs)
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