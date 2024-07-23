import numpy as np
from envs import make_env
from utils.os_utils import make_dir

class Tester:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.info = []

        if args.save_rews:
            make_dir('log/rews', clear=False)
            self.rews_record = {}
            self.rews_record[args.env] = []

    def test_rollouts(self):
        rewards_sum = 0.0
        success_rate = 0
        rews_List, V_pred_List = [], []
        obs_list = []
        acts_list = []
        for _ in range(self.args.test_rollouts):
            rewards = 0.0
            obss =[]
            obs = self.env.reset()
            obss += [obs]
            acts = []
            for timestep in range(self.args.test_timesteps):
                action, info = self.args.agent.step(obs, explore=False, test_info=True)
                obss += [obs]
                acts += [action]
                if 'test_eps' in self.args.__dict__.keys():
                    if np.random.uniform(0.0, 1.0)<=self.args.test_eps:
                        action = self.env.action_space.sample()
                obs, reward, done, info = self.env.test_step(action)
                rewards += reward
                if done: break
            if self.args.env_category=='fetch' and reward > 0:
                success_rate += 1
            rewards_sum += rewards
            rews_List += [rewards]
            obs_list += [obss]
            acts_list += [acts]

        if self.args.save_rews:
            step = self.args.learner.step_counter
            update = self.args.learner.update_counter
            rews = rewards_sum/self.args.test_rollouts
            success_rate = success_rate / self.args.test_rollouts
            self.args.writter.add_scalar('success_rate', success_rate, step)
            print("test:"+str(rews), "success rate:"+str(success_rate))

            self.args.writter.add_scalar('rewards', rews, step)
            
            
            self.rews_record[self.args.env].append((step, update))

    def cycle_summary(self):
        self.test_rollouts()

    def epoch_summary(self):
        if self.args.save_rews:
            for key, acc_info in self.rews_record.items():
                log_folder = 'rews'
                if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
                self.args.logger.save_npz(acc_info, key, log_folder)

    def final_summary(self):
        pass
