import torch
import torch.nn as nn
from constant import Constants
import torch.nn.functional as F


class GATE(nn.Module):

    def __init__(self, user_embeddings, item_embeddings, drop_rate=0.0, hidden=512, text_feature=512):
        super(GATE, self).__init__()
        self.drop_rate = drop_rate

        self.user_hidden = user_embeddings.weight.shape[1]
        self.item_hidden = item_embeddings.weight.shape[1]

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        self.user_linear = nn.Sequential(
            nn.Linear(self.user_hidden, hidden),
            nn.Tanh(),
        )
        self.content_linear = nn.Sequential(
            nn.Linear(text_feature, hidden),
            nn.Tanh(),
        )


        self.item_linear = nn.Sequential(
            nn.Linear(self.item_hidden, hidden),
            nn.ReLU(),
        )

        self.gate_user = nn.Linear(hidden, hidden)
        self.gate_content = nn.Linear(text_feature, hidden)


    def forward(self, users, items, content_embedding):
        user_embedding = self.user_embeddings(users)
        item_embedding = self.item_embeddings(items)

        z_user = self.user_linear(user_embedding)
        z_context = self.content_linear(content_embedding)
        user_f = self.gate_user(z_user)
        text_f = self.gate_content(z_context)
        gate = torch.sigmoid(user_f +text_f)

        gated_f = (gate*user_f) + (1-gate)*(text_f)
        item_embedding = self.item_linear(item_embedding)
        #                 B x H x 1              B x 1 x H
        # print(gated_f.unsqueeze(-1).shape, item_embedding.unsqueeze(1).shape)
        pred = torch.bmm(gated_f.unsqueeze(1), item_embedding.unsqueeze(-1))
        return torch.sigmoid(pred.squeeze(-1))

if __name__ == "__main__":
    user_embedding = nn.Embedding(1200, 10)
    item_embedding = nn.Embedding(1200, 10)
    text_f = torch.randn(10, 512)
    gate = GATE(user_embedding, item_embedding, text_feature=512)
    users = torch.randint(0, 1100, (10,))
    items = torch.randint(0, 1100, (10,))

    print(gate(users, items, text_f).shape)

