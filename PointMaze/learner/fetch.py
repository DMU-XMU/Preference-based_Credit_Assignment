import copy
import numpy as np
from envs import make_env
import torch
import os

class FetchLearner:
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.learner_info = []
        self.ep = []
        self.writter = args.writter
        self.pre = args.pre
        self.update_counter = 0
        self.update_flag = False



    def learn(self, args, env, agent, buffer, predictor=None):
        for _ in range(args.iterations):

            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                obs, reward, done, info = env.step(action)
                transition = {
                    'obs': obs_pre,
                    'obs_next': obs,
                    'acts': action,
                    'rews': reward,
                    'done': done if env.steps<args.test_timesteps else False,
                    'real_done': done,
                    'predict_r_prob': 0,
                    'state_val': 0
                }
                self.step_counter += 1
                self.ep.append(transition)
                if done:
                    self.ep_counter += 1
                    if self.ep_counter%8 == 0:
                        self.update_flag = True

                    for idx in range(len(self.ep)):
                        if self.pre == 'none':
                            self.ep[idx]['rews'] = self.ep[-1]['rews'] * pow(0.99, len(self.ep)-idx-1)
                        transition = copy.deepcopy(self.ep[idx])
                        
                        buffer.store_transition(transition)
                    self.ep = []
                    obs = env.reset()
            if self.pre == 'none' and buffer.step_counter>=args.warmup and self.update_flag:
                self.update_flag = False
                batch = buffer.sample_batch()
                batch['predict_R'] = batch['rews']
                info = agent.train(batch, self.target_count)

                self.writter.add_scalar('loss/critic_1', info['critic_1_loss'], self.step_counter)
                self.writter.add_scalar('loss/critic_2', info['critic_2_loss'], self.step_counter)
                self.writter.add_scalar('loss/policy', info['policy_loss'], self.step_counter)


            if self.pre == 'rrd' and buffer.step_counter>=args.warmup :
                self.update_flag = False

                for _ in range(args.train_batches):
                    
                    batch = buffer.sample_batch()
                    
                    r_rrd_pre, r_rrd = predictor(batch)
                    r_info = predictor.update(batch, r_rrd_pre, r_rrd)

                    rrd_pre = predictor(batch, train=False)
                    batch['predict_R'] = copy.deepcopy(rrd_pre)
                    self.writter.add_scalar('reward/pred_mean', np.mean(rrd_pre), self.step_counter)
                    self.writter.add_scalar('reward/pred_max', np.max(rrd_pre), self.step_counter)
                    self.writter.add_scalar('reward/pred_min', np.min(rrd_pre), self.step_counter)
                    self.writter.add_scalar('reward/pred_std', np.std(rrd_pre), self.step_counter)

                    info = agent.train(batch, self.target_count)
                    self.writter.add_scalar('loss/critic_1', info['critic_1_loss'], self.step_counter)
                    self.writter.add_scalar('loss/critic_2', info['critic_2_loss'], self.step_counter)
                    self.writter.add_scalar('loss/policy', info['policy_loss'], self.step_counter)

            
            if self.pre == 'ircr' and buffer.step_counter>=args.warmup and self.update_flag:
                batch = buffer.sample_batch()
                
                batch['predict_R'] = batch['rews']
                info = agent.train(batch, self.target_count)
                self.writter.add_scalar('loss/critic_1', info['critic_1_loss'], self.step_counter)
                self.writter.add_scalar('loss/critic_2', info['critic_2_loss'], self.step_counter)
                self.writter.add_scalar('loss/policy', info['policy_loss'], self.step_counter)
            
            if self.pre == 'ours' and buffer.step_counter>=args.warmup and self.update_flag:
                self.update_counter += 1
                critic_1_loss_sum = 0
                critic_2_loss_sum = 0
                policy_loss_sum = 0
                ent_loss_sum = 0
                alpha_sum = 0
                predict_loss = 0
                r_delta1 = args.delta1
                r_delta2 = args.delta2
     
                train_num = args.train_batches
                for _ in range(args.train_batches):

                    batch = buffer.sample_batch(predict=True)
                    predictor_batch = {'po':batch['pre_obs'], 'pa':batch['pre_acts'], 'pre_len':batch['pre_len']}
                    pre_prob, pre_rsa = predictor(predictor_batch, True)
                    r_info = predictor.update(batch, pre_prob, buffer.ep_info)
                    
                    batch = buffer.sample_batch()
                    predictor_batch = {'po':batch['pre_obs'], 'pa':batch['pre_acts'], 'pre_len':batch['pre_len']}
                    with torch.no_grad():            
                        pre_prob, pre_rsa = predictor(predictor_batch, train=False)                

                    r = pre_rsa[:,2] 

            
                    for i in range(len(pre_rsa)):
                        rew = batch['ep_rew'][i]
                        std_rew = 2*(rew - buffer.ep_info['set_min'])/max(buffer.ep_info['set_max']-buffer.ep_info['set_min'], 0.1) - 1

                        if(std_rew > r_delta2):
                            r[i] = pre_rsa[i][2] * rew 
                        elif(std_rew < r_delta1):
                            r[i] =  pre_rsa[i][0] * rew 
                        else:
                            r[i] =  pre_rsa[i][1] * rew 

                    batch['predict_R'] = r.unsqueeze(1).cpu().detach().numpy()
                    
                    info = agent.train(batch, self.target_count)
                    critic_1_loss_sum += info['critic_1_loss']
                    critic_2_loss_sum += info['critic_2_loss']
                    policy_loss_sum += info['policy_loss']
                    ent_loss_sum += info['ent_loss']
                    alpha_sum += info['alpha']

                self.update_flag = False

                critic_1_loss = critic_1_loss_sum / train_num
                critic_2_loss = critic_2_loss_sum / train_num
                policy_loss = policy_loss_sum / train_num
                predict_loss = predict_loss/train_num
               
                self.writter.add_scalar('loss/critic_1', critic_1_loss, self.step_counter)
                self.writter.add_scalar('loss/critic_2', critic_2_loss, self.step_counter)
                self.writter.add_scalar('loss/policy', policy_loss, self.step_counter)

        