import torch
import torch.nn as nn



class Block(nn.Module):
    
    def __init__(self, block_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(block_dim, block_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(block_dim),
            nn.Linear(block_dim, block_dim),
        )
    
    def forward(self, x):
        return self.net(x) + x
