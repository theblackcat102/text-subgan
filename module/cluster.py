import torch
import torch.nn as nn
import torch.nn.functional as F
from module.conv import ResBlock



class Cluster(nn.Module):

    def __init__(self, embedding, embedding_dim, embed_dim=100, k_bins=5, output_embed_dim=100):
        super(Cluster, self).__init__()
        self.output_embed_dim = output_embed_dim
        self.embed_dim = embed_dim
        self.k_bins = k_bins
        self.embedding = embedding

        self.block = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(embedding_dim+embed_dim, 256)
        self.logits = nn.Linear(256, k_bins)
        self.output = nn.Linear(256, output_embed_dim)
        self.relu = nn.ReLU()

    def forward(self, text, d_embedding):
        """
            inputs: float tensor, shape = [B, T, vocab_size]
        """
        inputs = self.embedding(text)
        inputs = inputs.transpose(1, 2)     # (B, H, T)
        outputs = self.block(inputs)        # (B, H, T)
        outputs = self.maxpool(outputs).squeeze(-1)     # (B, H, 1)
        latent = torch.cat([outputs, d_embedding], axis=1)
        out = self.linear(latent)
        out = self.relu(out)
        logits = self.logits(out)
        embed = self.output(out)
        return logits, embed


if __name__ == "__main__":
    inputs = torch.randint(0, 3200, (32, 128)).long()
    latent = torch.rand((32, 100))
    embedding = nn.Embedding(3201, 100)
    model = Cluster(embedding, 100)
    logits, embed = model(inputs, latent)

