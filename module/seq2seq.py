'''
    define module layers
'''
import torch
import torch.nn as nn

from utils import softmax, gumbel_softmax
from constant import Constants


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate
            the noise. Relative means that it will be multiplied by the
            magnitude of the value your are adding the noise to. This means
            that sigma can be the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable
            before computing the scale of the noise. If `False` then the scale
            of the noise won't be seen as a constant but something to optimize:
            this will bias the network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            if torch.cuda.is_available():
                self.noise = self.noise.cuda()
            if self.is_relative_detach:
                scale = self.sigma * x.detach()
            else:
                scale = self.sigma * x
            sampled_noise = \
                self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
            del sampled_noise
        return x


class Encoder(nn.Module):
    """A general Encoder, keep it as general as possible."""
    def __init__(self, inputs_size, hidden_size, num_layers, dropout,
                 bidirectional, cell):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # self.noise_layer = GaussianNoise(sigma=0.1)
        self.rnn = cell(
            inputs_size, hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=bidirectional,
            batch_first=True)

    def forward(self, inputs, hidden=None):
        """
        Args:
            inputs: int tensor, shape = [B x T x inputs_size]
            hidden: float tensor,
                shape = shape = [num_layers * num_directions x B x hidden_size]
            is_discrete: boolean, if False, inputs shape is
                [B x T x vocab_size]

        Returns:
            outputs: float tensor, shape = [B x T x (hidden_size x dir_num)]
            hidden: float tensorf, shape = [B x (hidden_size x dir_num)]
        """
        outputs, hidden = self.rnn(inputs, hidden)
        if self.bidirectional:
            outputs = outputs.view(-1, outputs.size(1), 2, self.hidden_size)
            hidden = outputs[:, -1, 0] + outputs[:, 0, 1]
            outputs = outputs[:, :, 0] + outputs[:, :, 1]
        else:
            hidden = outputs[:, -1]
        return outputs, hidden


class Decoder(nn.Module):
    """A general Decoder, keep it as general as possible."""
    def __init__(self, inputs_size, vocab_size, hidden_size,
                 num_layers, dropout, st_mode, cell, attention=None):
        super(Decoder, self).__init__()

        self.attention = attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.st_mode = st_mode

        self.rnn = cell(
            inputs_size, hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True)
        if attention is not None:
            self.outputs2vocab = nn.Linear(hidden_size * 2, vocab_size)
        else:
            self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden, encoder_outputs=None, temperature=1):
        """
        Args:
            inputs: float tensor, shape = [B x T x inputs_size]
            hidden: float tensor, shape = [num_layers x B x H]
            encoder_outputs: float tensor, shape = [B x Tin x H]
            temperature: float, gumbel softmax

        Returns:
            outputs: float tesor, shape = [B x T x vocab_size], probability
            hidden: float tensor, shape = [num_layers, B x H]
        """
        outputs, hidden = self.rnn(inputs, hidden)
        if self.attention is not None:
            outputs, attn_weight = self.attention(outputs, encoder_outputs)
        outputs = self.outputs2vocab(outputs)
        if self.training:
            outputs = softmax(outputs, temperature, st_mode=self.st_mode)
        else:
            outputs = softmax(outputs, temperature=1, st_mode=False)
        return outputs, hidden


class LuongAttention(nn.Module):
    """Implementation of Luong Attention

    reference:
        Effective Approaches to Attention-based Neural Machine Translation
        Minh-Thang Luong, Hieu Pham, Christopher D. Manning
        https://arxiv.org/abs/1508.04025

    """
    def __init__(self, encoder_hidden_size, decoder_hidden_size, score='dot'):
        super(LuongAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.score = score
        if score == 'dot':
            assert(encoder_hidden_size == decoder_hidden_size)
        elif score == 'general':
            self.linear = nn.Linear(decoder_hidden_size, encoder_hidden_size)
        else:
            assert(False)

    def compute_energy(self, decoder_outputs, encoder_outputs):
        if self.score == 'dot':
            # [B x Tou x H_decoder] x [B x Tin x H_encoder] -> [B x Tou x Tin]
            attn_weight = torch.bmm(
                decoder_outputs, encoder_outputs.transpose(1, 2))
        if self.score == 'general':
            # [B x Tou x H_encoder]
            decoder_outputs = self.linear(decoder_outputs)
            # [B x Tou x H_decoder] x [B x Tin x H_encoder] -> [B x Tou x Tin]
            attn_weight = torch.bmm(
                decoder_outputs, encoder_outputs.transpose(1, 2))
        return attn_weight

    def forward(self, decoder_outputs, encoder_outputs):
        """Support batch operation.

        Output size of encoder and decoder must be equal.

        Args:
            decoder_outputs: float tensor, shape = [B x Tou x H_decoder]
            encoder_outputs: float tensor, shape = [B x Tin x H_encoder]

        Returns:
            output: float tensor, shape = [B x Tou x (2 x H_decoder)]
            attn_weight: float tensor, shape = [B x Tou x Tin]
        """
        attn_weight = self.compute_energy(decoder_outputs, encoder_outputs)
        attn_weight = self.softmax(attn_weight)
        # [B x Tou x Tin] * [B x Tin x H] -> [B, Tou, H]
        attn_encoder_outputs = torch.bmm(attn_weight, encoder_outputs)
        # concat [B x Tou x H], [B x Tou x H] -> [B x Tou x (2 x H)]
        output = torch.cat([decoder_outputs, attn_encoder_outputs], dim=-1)

        return output, attn_weight


class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = nn.NLLLoss(ignore_index=Constants.PAD)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: float tensor, shape = [B x T x vocab_size], probility
            targets: int tensor, shape = [B x T],
                0 <= targets[i][j] < vocab_size

        Returns:
            loss: float tensor, scalar
        """
        batch_size, max_length, vocab_size = inputs.size()
        inputs = torch.log(inputs + 1e-10).view(-1, vocab_size)
        targets = targets.flatten()
        loss = self.criteria(inputs, targets)
        return loss