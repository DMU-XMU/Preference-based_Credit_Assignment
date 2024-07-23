import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
torch.autograd.set_detect_anomaly(True)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def AttenMatrix(n, l=10):
    indices = torch.arange(n)
    m = (indices.view(-1, 1) - indices.view(1, -1)).abs() > l

    return m


class RewardPredictor(nn.Module):
    def __init__(self, args):
        super(RewardPredictor, self).__init__()
        self.args = args
        self.device = torch.device(args.device if args.cuda else "cpu")
        self.obs_dims = np.sum(args.obs_dims) if self.args.act_type == 'continuous' else 64
        self.acts_dims = np.sum(args.acts_dims) if self.args.act_type == 'continuous' else 1
        self.hidden_dim = 32
        self.obs_embedding = nn.Linear(self.obs_dims, self.hidden_dim)
        self.acts_embedding = nn.Linear(self.acts_dims, self.hidden_dim)         
        
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.attention_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=2, dim_feedforward=self.hidden_dim*4, dropout=0.)
        self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=3)

        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
            )
        self.softmax = nn.Softmax(dim=-1)
        self.r_embedding = nn.Sequential(nn.Linear(1, self.hidden_dim))
        self.apply(init_params)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)



    def forward(self, obs, train=False):
        seq_obs = obs['po']
        seq_obs = pad_sequence(seq_obs, batch_first=True, padding_value=-10.).to(self.device).float()
        seq_obs = self.obs_embedding(seq_obs)
        seq_obs = self.pos_encoder(seq_obs)
        seq_acts = obs['pa']
        seq_acts = pad_sequence(seq_acts, batch_first=True, padding_value=-10.).to(self.device).float()
        seq_acts = self.acts_embedding(seq_acts)
        seq_acts = self.pos_encoder(seq_acts)

        obs_acts = torch.zeros((seq_obs.shape[0], seq_obs.shape[1]+seq_acts.shape[1], self.hidden_dim), dtype=torch.float32, device=self.device)
        obs_acts[:, 0::2, :] = seq_obs
        obs_acts[:, 1::2, :] = seq_acts
        obs_acts = obs_acts.transpose(0, 1)
        pre_len = obs['pre_len']

        embedding = obs_acts
        attn_mask = nn.Transformer.generate_square_subsequent_mask(embedding.shape[0], device=self.device)
        attn_mask = attn_mask != 0
        r = self.attention(embedding, mask=attn_mask, is_causal=True)
        r = self.output(r)
        r = r.transpose(0, 1)
        
        r_prob = self.softmax(r)
        if train:
            r_tra = []
            for i in range(len(pre_len)):
                tra_prob = torch.mean(r_prob[i][1:pre_len[i]*2+2:2], dim=0)
                r_tra += [tra_prob]
            r_tra = torch.cat(r_tra)
            r_tra = r_tra.view(-1, 3)
            r_sa = None
        else:
            r_tra = None
            r_sa = torch.cat([r_prob[i][pre_len[i]*2+1] for i in range(len(pre_len))]).view(-1, 3)

        return r_tra, r_sa

    def update(self, batch, r_prob, info):
        real_r = torch.tensor(np.array(batch['sub_rews']), device=self.device, dtype=torch.float32)
        target_label = real_r.clone()

        rews_setmax = info['set_max']
        rews_setmin = info['set_min']

        r_delta1 = self.args.delta1
        r_delta2 = self.args.delta2

        for i in range(len(real_r)):
            r = real_r[i]
            r = 2*(r - rews_setmin)/max(rews_setmax-rews_setmin, 0.1) - 1
            if(r > r_delta2):
                target_label[i] = 2
            elif(r < r_delta1):
                target_label[i] = 0
            else:
                target_label[i] = 1
        target_label = target_label.long()
        r_pos_num = len(target_label[target_label==2])
        r_neg_num = len(target_label[target_label==0])
        r_zero_num = len(target_label[target_label==1])
        loss_weight = torch.Tensor([
                                    1/r_neg_num if r_neg_num!=0 else 0, 
                                    1/r_zero_num if r_zero_num!=0 else 0, 
                                    1/r_pos_num if r_pos_num!=0 else 0
                                   ]).to(self.device)
        
        
        loss = F.cross_entropy(r_prob, target_label, loss_weight)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 50.)
        self.optimizer.step()
        
        return {'loss':loss.item()}