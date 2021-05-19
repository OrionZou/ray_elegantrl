'''Environments used in simulation.'''
import gym
from gym.spaces.box import Box as Continuous
from gym.spaces.discrete import Discrete
import numpy as np


class DiscreteBanditEnv(gym.Env):
    '''
    A simple single-state environment with discrete actions.
    '''

    def __init__(self, avg_rewards, noise_std=1e-3, **kwargs):
        self.action_dim = len(avg_rewards)
        self.avg_rewards = avg_rewards
        self.obs_dim = 1
        self.noise_std = noise_std
        self.action_space = Discrete(self.action_dim)
        self.observation_space = Continuous(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32)
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = self.avg_rewards[int(action)] + self.noise_std * np.random.normal()
        done = True
        return obs, reward, done, {}


class SingleSmallPeakEnv(gym.Env):
    '''
    A simple single-state environment with continuous actions.
    Reward is 1.0 for a in (-1, -0.8), and 0 otherwise in each dimension.
    '''

    def __init__(self, noise_std=1e-1, action_dim=1):
        self.action_dim = action_dim
        self.obs_dim = 1
        self.noise_std = noise_std
        low = np.array([-1.5] * action_dim)
        high = np.array([1.5] * action_dim)
        self.action_space = Continuous(low=low, high=high, dtype=np.float32)
        self.observation_space = Continuous(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32)
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = self.noise_std * np.random.normal()
        action = np.array(action)
        assert len(action) == self.action_dim
        # Highest reward possible is 1.
        reward += ((action > -1.0) & (action < -0.8)).sum() / self.action_dim
        done = True
        return obs, reward, done, {}


class TwoPeakEnv(gym.Env):
    '''
    A simple single-state environment with 1D continuous actions.
    Reward function has two peaks at -2 and 1.
    '''

    def __init__(self, noise_std=1e-1):
        self.action_dim = 1
        self.obs_dim = 1
        self.noise_std = noise_std
        low = np.array([-5])
        high = np.array([5])
        self.action_space = Continuous(low=low, high=high, dtype=np.float32)
        self.observation_space = Continuous(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32)
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = 1.1 * np.exp(-1.2 * np.power(action - (-2), 2))
        reward += 0.9 * np.exp(-0.9 * np.power(action - (1), 2))
        reward = reward.sum()
        done = True
        return obs, reward, done, {}


# class OneDimGoalEnv(gym.Env):
#     '''
#     A simple single-state environment with 1D continuous actions.
#     Reward function has two peaks at -2 and 1.
#     '''
#
#     def __init__(self):
#         self.action_dim = 1  # 方向x, y; 速度v
#         self.obs_dim = 4
#
#         self.delta_time = 1.
#         self.goal = 10.
#         self.trap_time = 0.
#         self.trap_posi = 0.
#         self.our = 0.
#         self.v = 0.
#         self.reward_0 = 0.
#         self.reward_1 = 0.
#
#         self.action_space = Continuous(low=-np.ones(self.action_dim, dtype=np.float32),
#                                        high=np.ones(self.action_dim, dtype=np.float32),
#                                        shape=np.ones(self.action_dim).shape,
#                                        dtype=np.float32)
#         self.observation_space = Continuous(
#             low=-100 * np.ones(self.obs_dim, dtype=np.float32),
#             high=100 * np.ones(self.obs_dim, dtype=np.float32),
#             shape=np.ones(self.obs_dim).shape,
#             dtype=np.float32)
#         self.reset()
#
#     def reset(self):
#         self.goal = 10.
#         self.our = 0.
#         self.v = 0.
#         self.trap_posi = 8.
#         self.trap_time = 0.
#         self.reward_0 = 0.
#         self.reward_1 = 0.
#         obs = np.array([(self.goal - self.our),
#                         self.trap_time,
#                         self.trap_posi - self.our,
#                         self.v])
#         return obs
#
#     def go_next_goal(self):
#         self.goal = 10.
#         self.our = 0.
#         self.v = 0.
#         self.trap_posi = 8.
#         self.trap_time = 0.  # 多少时间后陷阱发生
#         obs = np.array([(self.goal - self.our),
#                         self.trap_time,
#                         self.trap_posi - self.our,
#                         self.v])
#         return obs
#
#     def dynamic(self, action):
#         self.v += action * 0.2 * self.delta_time  # 计算速度
#         self.v = np.clip(self.v, 0, 1)[0]  # 限制速度范围
#         self.new_our = self.our + self.v * self.delta_time  # 更新agent位置
#         if self.trap_time == 0:  # 更新陷阱
#             if np.random.uniform() > 0.8:  # 陷阱一定概率维持
#                 self.trap_time = 5. * self.delta_time
#         else:
#             self.trap_time -= 1. * self.delta_time
#         if (self.trap_time == 0.) and (self.new_our >= self.trap_posi) and (self.our <= self.trap_posi):
#             self.reward_1 = -1.
#         else:
#             self.reward_1 = 0.
#             if (self.new_our >= self.goal) and (self.our <= self.goal):
#                 self.reward_0 = 1.
#             else:
#                 self.reward_0 = 0.
#         self.our = self.new_our
#
#     def step(self, action):
#         self.dynamic(action)
#         obs = np.array([(self.goal - self.our),
#                         self.trap_time,
#                         self.trap_posi,
#                         self.v])
#         reward = self.reward_0 + self.reward_1
#         # done = 1 if self.reward_1 == -1. else 0
#         done = 0 if reward == 0. else 1
#         # if self.reward_0 == 1.:
#         #     obs = self.go_next_goal()
#         return obs, reward, done, {}
#
#
# if __name__ == '__main__':
#     env = OneDimGoalEnv()
#     action = np.array([1.])
#     iters = 1000
#     horizon_step = 1000
#     ret_list = []
#     for _ in range(iters):
#         obs = env.reset()
#         ret = 0
#         for _ in range(horizon_step):
#             if (env.trap_posi - env.goal) / (env.v + 1e-6) < env.trap_time:
#                 action = np.array([-1])
#             else:
#                 action = np.array([1])
#             next_obs, reward, done, _ = env.step(action)
#             if env.trap_time > 0:
#                 if (env.trap_posi - env.goal) / (env.v + 1e-6) < env.trap_time:
#                     print('gg')
#
#             ret += reward
#             if done:
#                 break
#         print(ret)
#         ret_list.append(ret)
#     print("###ENd###")
#     result = np.array(ret_list)
#     print(result)
#     print(result.mean())
#     print(result.std())
#     print(result.max())
#     print(result.min())



