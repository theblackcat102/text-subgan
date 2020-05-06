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
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(embedding_dim+embed_dim, 256)
        self.output2embed = nn.Sequential(
            nn.Linear(256, output_embed_dim),
            nn.BatchNorm1d(output_embed_dim)
        )
        self.logits = nn.Linear(output_embed_dim, k_bins)

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
        embed = self.output2embed(out)
        embed = self.relu(embed)
        logits = self.logits(embed)

        return logits, embed


class VAE_Cluster(nn.Module):

    def __init__(self, block_dim, embed_dim=100, k_bins=5, output_embed_dim=100):
        super(VAE_Cluster, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(block_dim+embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_embed_dim),
        )
        self.relu = nn.ReLU()
        self.logits = nn.Linear(output_embed_dim, k_bins)
        
    

    def forward(self, text_latent, d_embedding):
        latent = torch.cat([text_latent, d_embedding], axis=1)
        embed = self.linear(latent)
        embed_ = self.relu(embed)
        logits = self.logits(embed)
        return logits, embed


if __name__ == "__main__":
    inputs = torch.randint(0, 3200, (32, 128)).long()
    latent = torch.rand((32, 100))
    embedding = nn.Embedding(3201, 100)
    model = Cluster(embedding, 100)
    logits, embed = model(inputs, latent)

