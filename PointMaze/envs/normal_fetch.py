import gymnasium as gym
# import gym
import numpy as np
from utils.os_utils import remove_color

class FetchNormalEnv():
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env, max_episode_steps=100).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render = self.env.render

        self.acts_dims = list(self.action_space.shape)
        self.obs_dims = list(self.observation_space['observation'].shape
                             + self.observation_space['desired_goal'].shape
                             + self.observation_space['achieved_goal'].shape
                             )

        self.action_scale = np.array(self.action_space.high)
        for value_low, value_high in zip(list(self.action_space.low), list(self.action_space.high)):
            assert abs(value_low+value_high)<1e-3, (value_low, value_high)

        self.reset()
        self.env_info = {
            'Steps': self.process_info_steps, # episode steps
            'Rewards@green': self.process_info_rewards # episode cumulative rewards
        }

    def process_info_steps(self, obs, reward, info):
        self.steps += 1
        return self.steps

    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info(self, obs, reward, info):
        return {
            remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }

    def env_step(self, action):
        obs, reward, done, truncated, info = self.env.step(action*self.action_scale)
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        if reward>0:
            done=True
        elif self.steps==self.args.test_timesteps-1:
            reward = 0.5-pow((achieved_goal[0]-desired_goal[0])**2 + (achieved_goal[1]-desired_goal[1])**2, 0.5)
            done = True
      
        # done = done or truncated
        obs = np.concatenate(list(obs.copy().values()), axis=0)
        info = self.process_info(obs, reward, info)
        self.last_obs = obs.copy()
        
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self.env_step(action)

        return obs, reward, done, info
    
    def t_step(self, action):
        obs, reward, done, info = self.env_step(action)

        if done:
            info['success'] = reward > 0
        return obs, reward, done, info

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0
        obs = (self.env.reset())[0].copy()
        self.last_obs = np.concatenate(list(obs.values()), axis=0)

    def reset(self):
        self.reset_ep()
        return self.last_obs.copy()
    
    def get_obs(self):
        return self.last_obs.copy()
