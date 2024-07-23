import copy
import numpy as np
import torch
import random

class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': [],
            'real_done': [],
            'predict_r_prob':[],
            'state_value':[],
            'predict_R':[]
        }
        self.length = 0
        self.sum_rews = 0
    
    def store_transition(self, info):
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['obs'].append(copy.deepcopy(info['obs_next']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(np.float32(info['done'])))
        self.ep['real_done'].append(copy.deepcopy(np.float32(info['real_done'])))
        self.ep['state_value'].append(copy.deepcopy(info['state_val']))
                
        self.length += 1
        self.sum_rews += info['rews']

    def sample(self):
        idx = np.random.randint(1, self.length)
        info = {
            'obs': copy.deepcopy(self.ep['obs'][idx]),
            'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])],
            'ep_rew': copy.deepcopy(self.sum_rews),
            'pre_obs': torch.tensor(np.array(self.ep['obs'][0: idx+1]), dtype=torch.float),
            'pre_acts': torch.tensor(np.array(self.ep['acts'][0: idx+1]), dtype=torch.float),
            'pre_len': idx,
            'sub_rews': self.sum_rews*(idx+1)/self.length,
            'ep_len':self.length
        }
        return info


class ReplayBuffer_Predictor:
    def __init__(self, args):
        self.args = args
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size
        self.ep = []
        self.ram_idx = []
        self.length = 0
        self.head_idx = 0
        self.rear_idx = -1
        self.in_head = True

        self.ep_idx = []
        self.orderRewIdx = {}
        self.ep_info={'set_max':1.0,'set_min':-7.3, 'max':0,'min':0,'avg':0,'std':0} #medium

    def update_ep_info(self, rew):
        rews = np.array([v[1] for v in self.orderRewIdx.items()]+[rew])
        self.ep_info['max'] =  np.max(rews)
        self.ep_info['min'] = np.min(rews)
        self.ep_info['avg'] = np.average(rews)
        self.ep_info['std'] = np.std(rews)

    def store_transition(self, info):
        if self.in_head:
            new_ep = Trajectory(info['obs'])
            self.ep.append(new_ep)
        self.ep[-1].store_transition(info)
        self.ram_idx.append(self.ep_counter)
        if len(self.ep_idx) == 0:
            self.ep_idx.append(0)
        self.length += 1

        if self.length>self.buffer_size:
            del_len = self.ep[0].length
            self.orderRewIdx.pop(self.ram_idx[0])
            self.ep.pop(0)
            self.head_idx += 1
            self.length -= del_len
            self.ram_idx = self.ram_idx[del_len:]
            self.ep_idx.pop(0)
            self.ep_idx = [i - del_len for i in self.ep_idx]
            

        self.step_counter += 1
        self.in_head = info['real_done']
        if info['real_done']:
            if self.ep[-1].length > 1:
                self.orderRewIdx[self.ep_counter] = self.ep[-1].sum_rews
            self.rear_idx = self.head_idx - 1 if self.head_idx - 1 > 0 else len(self.ep)-1
            self.update_ep_info(self.ep[-1].sum_rews)
            self.ep_counter += 1
            self.ep_idx.append(self.ep_idx[-1]+self.ep[-1].length)

    def sample_batch(self, batch_size=-1, predict=False):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[], ep_rew=[], pre_obs=[], pre_acts=[], pre_len=[], sub_rews=[], ep_len=[])
        if predict:
            orderRewIdx = np.array([[k, v] for k, v in sorted(self.orderRewIdx.items(), key=lambda x: x[1])])
            batch_idx = np.array([], dtype=np.int32)

            r_delta1 = self.args.delta1
            r_delta2 = self.args.delta2
            b_i = 0
            for i in range(len(orderRewIdx)):
                r = 2*(orderRewIdx[i][1] - self.ep_info['set_min'])/max(self.ep_info['set_max']-self.ep_info['set_min'], 0.1) - 1
                if  len(batch_idx)==0 and r  > r_delta1:
                    batch_idx = np.concatenate((batch_idx, np.random.choice(orderRewIdx[:i,0], size=int(batch_size*i/len(orderRewIdx)), replace=True).astype(np.int32)), dtype=np.int32)
                    b_i = i
                elif r  > r_delta2:
                    batch_idx = np.concatenate((batch_idx, np.random.choice(orderRewIdx[b_i:i,0], size=int(batch_size*(i-b_i)/len(orderRewIdx)), replace=True).astype(np.int32)), dtype=np.int32)
                    batch_idx = np.concatenate((batch_idx, np.random.choice(orderRewIdx[i:,0], size=batch_size-len(batch_idx), replace=True).astype(np.int32)), dtype=np.int32)
                    break
            if len(batch_idx) < batch_size:
                batch_idx = np.random.choice(orderRewIdx[:,0], size=batch_size, replace=True).astype(np.int32)
            for idx in batch_idx:
                idx -= self.head_idx
                info = self.ep[idx].sample()
                for key in info.keys():
                    batch[key].append(info[key])
        else:
            for i in range(batch_size):
                idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
                info = self.ep[idx].sample()
                for key in info.keys():
                    batch[key].append(info[key])

        return batch
