import torch
import torch.nn as nn
from constant import Constants
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


def init_gate(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    


class GateModule(nn.Module):
    def __init__(self, hidden):
        super(GateModule, self).__init__()
        self.H = hidden
        self.gate_matrix1 = Variable(torch.zeros(self.H, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_matrix2 = Variable(torch.zeros(self.H, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_matrix1 = torch.nn.init.xavier_uniform_(self.gate_matrix1)
        self.gate_matrix2 = torch.nn.init.xavier_uniform_(self.gate_matrix2)
        self.gate_bias = Variable(torch.zeros(1, self.H).type(T.FloatTensor), requires_grad=True)
        self.gate_bias = torch.nn.init.xavier_uniform_(self.gate_bias)
    
    def forward(self, input1, input2):
        gate = torch.sigmoid(input1.mm(self.gate_matrix1) + input2.mm(self.gate_matrix2) + self.gate_bias)
        gated_embedding = gate * input1 + (1 - gate) * input2
        return gated_embedding

class Pointer(nn.Module):
    def __init__(self, user_embeddings, item_embeddings, drop_rate=0.0, hidden=128, text_feature=64):
        super(Pointer, self).__init__()
        self.drop_rate = drop_rate
        self.user_hidden = user_embeddings.weight.shape[1]
        self.item_hidden = item_embeddings.weight.shape[1]
        self.item_size = item_embeddings.weight.shape[0]

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.item_proj = nn.Linear(self.item_hidden, hidden)
        self.user_linear = nn.Linear(self.user_hidden, hidden, bias=False)
        self.content_linear = nn.Linear(text_feature, hidden, bias=True)
        self.user_linear.apply(init_gate)
        self.content_linear.apply(init_gate)
    
    def forward(self, users, items, content_embedding, temperature=1):
        user_embedding = self.user_embeddings(users)
        item_embedding = self.item_proj(self.item_embeddings(items))

        user_embedding = self.user_linear(user_embedding)
        content_embedding = self.content_linear(content_embedding)

        gated_f = torch.tanh(user_embedding + content_embedding)
        pred = torch.bmm(gated_f.unsqueeze(1), item_embedding.unsqueeze(-1))
        # print(pred.shape)
        return pred.squeeze(-1).squeeze(-1)        

    def inference(self, users, content_embedding):
        user_embedding = self.user_embeddings(users)
        item_embedding = self.item_embeddings.weight
        item_embedding = self.item_proj(item_embedding)

        user_embedding = self.user_linear(user_embedding)
        content_embedding = self.content_linear(content_embedding)

        gated_f = torch.tanh(user_embedding+content_embedding)
        # print(gated_f.repeat(self.item_size, 1, 1).shape, item_embedding.unsqueeze(-1).shape)
        # gated_f = z_user
        # #                 B x H x 1              B x 1 x H
        pred = torch.bmm(gated_f.repeat(self.item_size, 1, 1), item_embedding.unsqueeze(-1))
        rank = torch.argsort(pred.flatten(), descending=True)
        rank_scores_ = torch.sigmoid(pred.squeeze(-1)).flatten()
        # print(pred.shape, rank[:10], rank_scores_[rank][:10])

        return rank_scores_, rank

class GATE(nn.Module):

    def __init__(self, user_embeddings, item_embeddings, drop_rate=0.0, hidden=128, text_feature=64):
        super(GATE, self).__init__()
        self.drop_rate = drop_rate

        self.user_hidden = user_embeddings.weight.shape[1]
        self.item_hidden = item_embeddings.weight.shape[1]
        self.item_size = item_embeddings.weight.shape[0]

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.user_linear = nn.Sequential(
            nn.Linear(self.user_hidden, hidden),
            # nn.Dropout(drop_rate),
            nn.Tanh(),
        )
        self.user_linear.apply(init_gate)


        self.content_linear = nn.Sequential(
            nn.Linear(text_feature, hidden),
            nn.Tanh(),
        )
        self.content_linear.apply(init_gate)

        self.item_linear = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.item_hidden, hidden),
            nn.Tanh(),
        )
        self.pred_layer = nn.Linear(
            hidden*2, 1
        )
        self.item_linear.apply(init_gate)

        self.gate = GateModule(hidden)


    def forward(self, users, items, content_embedding, temperature=1):
        user_embedding = self.user_embeddings(users)
        item_embedding = self.item_embeddings(items)

        user_embedding = self.user_linear(user_embedding)
        item_embedding = self.item_linear(item_embedding)

        # gated_f = user_embedding
        # pred = torch.bmm(gated_f.unsqueeze(1), item_embedding.unsqueeze(-1))
        # return torch.sigmoid(pred.squeeze(-1)).squeeze(-1)

        content_embedding = self.content_linear(content_embedding)

        gated_f = self.gate(user_embedding, content_embedding)
        #                 B x H x 1              B x 1 x H
        # print(gated_f.unsqueeze(-1).shape, item_embedding.unsqueeze(1).shape)
        pred = torch.bmm(gated_f.unsqueeze(1), item_embedding.unsqueeze(-1))
        # print(pred.shape)
        return pred.squeeze(-1).squeeze(-1)
    
    def inference(self, user_id, content_embedding):

        user_embedding = self.user_embeddings(user_id)
        item_embedding = self.item_embeddings.weight

        item_embedding = self.item_linear(item_embedding)
        user_embedding = self.user_linear(user_embedding)
        # print(user_embedding.shape, item_embedding.shape)
        # gated_f = user_embedding
        # pred = torch.bmm(gated_f.repeat(self.item_size, 1, 1), item_embedding.unsqueeze(-1))
        # return pred.flatten(), torch.argsort(pred.flatten(), descending=True)

        content_embedding = self.content_linear(content_embedding)
        # print(user_embedding.shape, content_embedding.shape)
        gated_f = self.gate(user_embedding, content_embedding)
        # print(gated_f.repeat(self.item_size, 1, 1).shape, item_embedding.unsqueeze(-1).shape)
        # gated_f = z_user
        # #                 B x H x 1              B x 1 x H
        pred = torch.bmm(gated_f.repeat(self.item_size, 1, 1), item_embedding.unsqueeze(-1))
        # print(pred.shape)
        # print(pred.shape)
        rank = torch.argsort(pred.flatten(), descending=True)
        rank_scores_ = torch.sigmoid(pred.squeeze(-1)).flatten()
        # print(pred.shape, rank[:10], rank_scores_[rank][:10])

        return rank_scores_, rank

if __name__ == "__main__":
    user_embedding = nn.Embedding(1200, 10)
    item_embedding = nn.Embedding(1200, 10)
    text_f = torch.randn(10, 32).cuda()
    gate = Pointer(user_embedding, item_embedding, hidden=128, text_feature=32).cuda()
    # gate.apply(init_gate)
    users = torch.randint(0, 1100, (10,)).cuda()
    items = torch.randint(0, 1100, (10,)).cuda()

    print(gate(users, items, text_f).shape)
    pred, rank = gate.inference(users[0], text_f[[0]])
    print(rank.shape)
    print(pred[rank][:10], rank[:10])
