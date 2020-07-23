"""
copy from @yuyusica repo
Define RNN-based BISET encoders.
https://github.com/yuyusica/TemPEST_onmt

"""
import torch.nn as nn
import torch.nn.functional as F

import torch

class T2A(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.W = nn.Linear(dim,dim,bias=False)
        self.U = nn.Linear(dim,dim,bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, article_hidden, template_hidden):
        seq_len = template_hidden.shape[0]
        article_hidden = article_hidden[-1,:,:].repeat(seq_len,1,1)
        s = self.W(article_hidden)+self.U(template_hidden)+self.b
        s = template_hidden*torch.sigmoid(s)
        return s

class A2T(nn.Module):
    '''
    args:
        1,batch,dim
    return:
        scaler
    '''
    def __init__(self,dim,att_type='dot'):
        super().__init__()
        assert att_type in ['dot','general','mlp']
        self.att_type=att_type
        if att_type=='general':
            self.W = nn.Linear(dim,dim,bias=False)
        else:
            self.W = nn.Linear(dim*2,dim,bias=False)
            self.V = nn.Linear(dim,1,bias=False)

    def forward(self, x, y):
        # if self.att_type=='mlp':
        #     z=torch.cat([x,y],dim=2)
        #     z=F.tanh(self.W(z))
        #     z=self.V(z)
        #     return F.sigmoid(z)
        # print('1==============')
        # print(x.size())
        # print(y.size())

        x = x.transpose(0,1)
        y = y.permute(1,2,0)
        if self.att_type == 'general':
            x = self.W(x)
        return torch.sigmoid(torch.bmm(x,y))

class BiSET(nn.Module):
    def __init__(self, article_hidden_size, template_hidden_size, att_type='general'):
        '''
            hidden_size must be same as input hidden size
        '''
        super().__init__()
        self.t_proj = None
        if article_hidden_size != template_hidden_size:
            self.t_proj = nn.Linear(template_hidden_size, article_hidden_size)

        self.T2A=T2A(article_hidden_size)

        self.A2T=A2T(article_hidden_size, att_type=att_type)

    def forward(self, article_hidden, template_hidden):
        # torch.Size([18, 32, 64]) torch.Size([18, 32, 64])
        # T x B x hidden dim
        if self.t_proj != None:
            template_hidden = self.t_proj(template_hidden)

        hidden_dim = article_hidden.shape[-1]
        # batch first to time first
        template_hidden = template_hidden.transpose(0, 1)
        article_hidden = article_hidden.transpose(0, 1)

        s = self.A2T(template_hidden[-1:, :], article_hidden[-1:, :])

        s = s.repeat(1, article_hidden.shape[0], hidden_dim)
        s = s.transpose(0, 1)

        gate_memory_bank = self.T2A(template_hidden, article_hidden)

        memory_bank = (1 - s) * article_hidden + s * gate_memory_bank
        # time first to batch first

        memory_bank = memory_bank.transpose(0, 1)
        return memory_bank

class ConBiSET(nn.Module):
    def __init__(self, article_hidden_size, template_hidden_size, user_hidden_size, att_type='general'):
        '''
            hidden_size must be same as input hidden size
        '''
        super().__init__()
        self.t_proj = None
        if article_hidden_size != template_hidden_size:
            self.t_proj = nn.Linear(template_hidden_size, article_hidden_size)
        self.user_proj = nn.Linear(user_hidden_size, article_hidden_size)
        self.T2A=T2A(article_hidden_size)

        self.A2T=A2T(article_hidden_size, att_type=att_type)

    def forward(self, article_hidden, template_hidden, user_hidden):
        # torch.Size([18, 32, 64]) torch.Size([18, 32, 64])
        # T x B x hidden dim
        if self.t_proj != None:
            template_hidden = self.t_proj(template_hidden)

        hidden_dim = article_hidden.shape[-1]
        # batch first to time first
        template_hidden = template_hidden.transpose(0, 1)

        article_hidden = article_hidden.transpose(0, 1)

        user_hidden = self.user_proj(user_hidden).unsqueeze(1)
        user_bias = user_hidden.repeat(1, article_hidden.shape[0], 1)
        user_bias = user_bias.transpose(0, 1)

        s = self.A2T(template_hidden[-1:, :], article_hidden[-1:, :])

        s = s.repeat(1, article_hidden.shape[0], hidden_dim)
        s = s.transpose(0, 1)

        gate_memory_bank = self.T2A(template_hidden, article_hidden)

        memory_bank = (1 - s) * article_hidden + s * gate_memory_bank + user_bias
        # time first to batch first 

        memory_bank = memory_bank.transpose(0, 1)
        return memory_bank

if __name__ == "__main__":
    a_hidden = torch.randn(32, 10, 32)
    t_hidden = torch.randn(32, 10, 64)
    # biset = BiSET(32, 64, att_type='dot')
    # memory = biset(a_hidden, t_hidden)
    u_hidden = torch.randn(32, 32)
    biset = ConBiSET(32, 64, 32, att_type='dot')
    memory = biset(a_hidden, t_hidden, u_hidden)

    print(memory.shape)
