import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResBlock(nn.Module):
    def __init__(self, channel, kernel=5, alpha=0.5):
        super(ResBlock, self).__init__()
        assert(kernel % 2 == 1)
        padding = kernel // 2
        self.alpha = alpha
        self.res_block = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channel, channel, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channel, channel, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(channel),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + self.alpha * output