import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channel, kernel=5, alpha=0.3):
        super(ResBlock, self).__init__()
        assert(kernel % 2 == 1)
        padding = kernel // 2
        self.alpha = alpha
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(channel, channel, kernel_size=kernel, padding=padding),
            nn.ReLU(True),
            nn.Conv1d(channel, channel, kernel_size=kernel, padding=padding),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + self.alpha * output