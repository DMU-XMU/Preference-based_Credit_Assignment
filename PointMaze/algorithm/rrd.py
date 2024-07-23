import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomizedReturnDecomposition(nn.Module):
    def __init__(self, args):
        super(RandomizedReturnDecomposition, self).__init__()
        self.args = args
        self.device = torch.device(args.device if args.cuda else "cpu")
        
        mlp_dims = np.sum(self.args.obs_dims)*2 + np.sum(self.args.acts_dims)
        self.hidden_size = 256
        self.mlp_rrd = nn.Sequential(
            nn.Linear(mlp_dims, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
    def forward(self, batch, train=True):
        if train:
            rrd_obs = torch.tensor(np.array(batch['rrd_obs']), dtype=torch.float32, device=self.device)
            rrd_acts = torch.tensor(np.array(batch['rrd_acts']), dtype=torch.float32, device=self.device)
            rrd_obs_next = torch.tensor(np.array(batch['rrd_obs_next']), dtype=torch.float32, device=self.device)
            
            rrd_inputs = torch.cat([rrd_obs, rrd_acts, rrd_obs-rrd_obs_next], dim=-1)
            rrd_rews_pred = self.mlp_rrd(rrd_inputs).squeeze()
            rrd = torch.mean(rrd_rews_pred, dim=-1).unsqueeze(1)
            return rrd_rews_pred, rrd
        else:
            with torch.no_grad():
                rrd_obs = torch.tensor(np.array(batch['obs']), dtype=torch.float32, device=self.device)
                rrd_acts = torch.tensor(np.array(batch['acts']), dtype=torch.float32, device=self.device)
                rrd_obs_next = torch.tensor(np.array(batch['obs_next']), dtype=torch.float32, device=self.device)
                done = torch.tensor(np.array(batch['done']), dtype=torch.float32, device=self.device).bool()
                # rews = torch.tensor(np.array(batch['rews']), dtype=torch.float32, device=self.device)
                # rews = torch.masked_fill(rews, ~done, 0.)
                
                rrd_inputs = torch.cat([rrd_obs, rrd_acts, rrd_obs-rrd_obs_next], dim=-1)
                rrd_rews_pred = self.mlp_rrd(rrd_inputs)
                # rrd_rews_pred = torch.masked_fill(rrd_rews_pred, done, 0.)
                
                # rrd_rews_pred = rrd_rews_pred + rews
                return rrd_rews_pred.cpu().numpy()
        
    def update(self, batch, rrd_rews_pred, rrd):
        rrd_rews = torch.tensor(np.array(batch['rrd_rews']), dtype=torch.float32, device=self.device)
        
        
        r_loss = torch.mean(torch.square(rrd-rrd_rews))
        if self.args.rrd_bias_correction:
            rrd_var_coef = torch.tensor(np.array(batch['rrd_var_coef']), dtype=torch.float32, device=self.device)
            n = rrd_rews_pred.shape[1]
            r_var_single = torch.sum(torch.square(rrd_rews_pred-torch.mean(rrd_rews_pred, dim=-1).unsqueeze(-1)), dim=-1).unsqueeze(-1) / (n-1)
            r_var = torch.mean(r_var_single*rrd_var_coef/n)
            
            r_total_loss = r_loss - r_var
        else:
            r_total_loss = r_loss
        self.optimizer.zero_grad()
        r_total_loss.backward()
        self.optimizer.step()
        if self.args.rrd_bias_correction:
            return {'r_loss': r_total_loss.item(), 'r_var': r_var.item()}
        else:
            return {'r_loss': r_total_loss.item()}