import os
import time
from copy import deepcopy
import gym
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta)
        return logprob.sum(1)


class ActorDiscretePPO(ActorPPO):
    def __init__(self, *args):
        super().__init__(*args)
        self.soft_max = nn.Softmax(dim=1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)

    def get_action_prob(self, state):
        action_prob = self.soft_max(self.net(state))

        # action_dist = self.Categorical(action_prob)
        # action_int = action_dist.sample()
        # logprob = action_dist.log_prob(action_int)  # get_action_logprob

        action_int = torch.multinomial(action_prob, 1, True)
        # logits = torch.log(action_prob.clamp(min=1e-6, max=1 - 1e-6))  # prob_to_logit
        # value, log_pmf = torch.broadcast_tensors(action_int, logits)
        # logprob = log_pmf.gather(-1, value[:, :1]).squeeze(-1)
        return action_int.squeeze(1), action_prob

    def compute_logprob(self, state, action):
        policy_prob = self.soft_max(self.net(state))
        action_dist = self.Categorical(policy_prob)
        return action_dist.log_prob(action)


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.97  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.compute_reward = None

        self.act = None
        self.cri = None
        self.optimizer = None
        self.criterion = None
        self.device = None

    def init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if if_per_or_gae else self.compute_reward_raw

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        # self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
        #                                    {'params': self.cri.parameters(), 'lr': learning_rate}])
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        steps = 0
        while steps < target_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                trajectory_list.append((state, other))
                if done:
                    break
                state = next_state
            steps += step + 1
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times=8):
        buffer.update_now_len()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''compute reverse reward'''
        with torch.no_grad():  # Trajectory using reverse reward
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_logstd + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(self, buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for update_c in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            # obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            # self.optimizer.zero_grad()
            # obj_united.backward()
            # self.optimizer.step()
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()

        logging_tuple = (obj_critic.item(), obj_actor.item(), self.act.a_logstd.mean().item())
        return logging_tuple

    @staticmethod
    def compute_reward_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage


class AgentoffPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.97  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.compute_reward = None

        self.act = None
        self.cri = self.cri_target = None
        self.optimizer = None
        self.criterion = None
        self.device = None

    def init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act_target = deepcopy(self.act)
        self.criterion = torch.nn.SmoothL1Loss()
        # self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
        #                                    {'params': self.cri.parameters(), 'lr': learning_rate}])
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        steps = 0
        while steps < target_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))

                other = (reward * reward_scale, 0.0 if done else gamma, *action)
                trajectory_list.append((state, other))
                if done:
                    break
                state = next_state
            steps += step + 1
        return trajectory_list

    def update_net(self, buffer, target_step, batch_size, repeat_times=8):
        buffer.update_now_len()
        buf_len = buffer.now_len  # assert buf_len >= _target_step
        target_step = buf_len / target_step
        self.hard_update(self.act_target, self.act)
        for _ in range(int(target_step * repeat_times)):
            obj_critic, advantage, state, action = self.compute_reward_td(buffer, batch_size)
            logprob = self.act_target.compute_logprob(state, action)
            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()

        logging_tuple = (obj_critic.item(), obj_actor.item(), self.act.a_logstd.mean().item())
        return logging_tuple

    def compute_reward_td(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            next_s = torch.as_tensor(next_s, dtype=torch.float32, device=self.device)
            next_v = self.cri_target(next_s)
            v_label = reward + mask * next_v
        v_value = self.cri(state)
        obj_critic = self.criterion(v_value, v_label)
        advantage = v_label - v_value.detach()
        return obj_critic, advantage, state, action

    @staticmethod
    def hard_update(target_net, current_net):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data)


class AgentTargetPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.97  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.compute_reward = None

        self.act = None
        self.cri = self.cri_target = None
        self.optimizer = None
        self.criterion = None
        self.device = None

    def init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if if_per_or_gae else self.compute_reward_raw

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        steps = 0
        while steps < target_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                trajectory_list.append((state, other))
                if done:
                    break
                state = next_state
            steps += step + 1
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times=8):
        buffer.update_now_len()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''compute reverse reward'''
        with torch.no_grad():  # Trajectory using reverse reward
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            # buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_value = torch.cat([self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)],
                                  dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_logstd + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(self, buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for update_c in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.update_optimizer(self.optimizer, obj_united)
            self.soft_update(self.cri_target, self.cri, tau=2 ** -8)

        logging_tuple = (obj_critic.item(), obj_actor.item(), self.act.a_logstd.mean().item())
        return logging_tuple

    @staticmethod
    def compute_reward_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def update_optimizer(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentOffPolicyPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.97  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.compute_reward = None

        self.act = None
        self.cri = self.cri_target = None
        self.optimizer = None
        self.criterion = None
        self.device = None

    def init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if if_per_or_gae else self.compute_reward_raw

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()

    def explore_env1(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()
        trajectory_len_list = list()
        steps = 0
        while steps < target_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                trajectory_list.append((state, other))
                if done:
                    break
                state = next_state
            trajectory_len_list.append(step + 1)
            steps += step + 1
        return trajectory_list, trajectory_len_list

    def explore_env2(self, env, reward_scale, gamma):
        trajectory_list = list()
        trajectory_len_list = list()
        steps = 0
        while steps < env.max_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                trajectory_list.append((state, other))
                step += 1
                if done:
                    break
                state = next_state
            trajectory_len_list.append(step + 1)
            steps += step + 1
        return trajectory_list, trajectory_len_list

    def update_net(self, buffer, batch_size, repeat_times=8):
        buffer.update_now_len()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''compute reverse reward'''
        with torch.no_grad():  # Trajectory using reverse reward
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            # buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_value = torch.cat([self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)],
                                  dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_logstd + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(self, buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for update_c in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.update_optimizer(self.optimizer, obj_united)
            self.soft_update(self.cri_target, self.cri, tau=2 ** -8)

        logging_tuple = (obj_critic.item(), obj_actor.item(), self.act.a_logstd.mean().item())
        return logging_tuple

    @staticmethod
    def compute_reward_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def update_optimizer(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentDiscretePPO(AgentPPO):
    def init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if if_per_or_gae else self.compute_reward_raw

        self.act = ActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_action(self, state) -> tuple:
        """select action for PPO

        :array state: state.shape==(state_dim, )

        :return array action: state.shape==(action_dim, )
        :return array noise: noise.shape==(action_dim, ), the noise
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, action_prob = self.act.get_action_prob(states)
        return actions[0].detach().cpu().numpy(), action_prob[0].detach().cpu().numpy()  # plan to remove detach()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        steps = 0
        while steps < target_step:
            state = env.reset()
            step = 0
            for step in range(env.max_step):
                a_int, a_prob = self.select_action(state)

                next_state, reward, done, _ = env.step(a_int)

                other = (reward * reward_scale, 0.0 if done else gamma, a_int, *a_prob)
                trajectory_list.append((state, other))
                if done:
                    break
                state = next_state
            steps += step + 1
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times=8):
        buffer.update_now_len()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''compute reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_a_prob, buf_state = buffer.sample_all()
            buf_a_int = buf_action.long()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)

            # buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)
            buf_logits = torch.log(buf_a_prob.clamp(min=1e-6, max=1 - 1e-6))  # prob_to_logit
            buf_a_val, buf_log_pmf = torch.broadcast_tensors(buf_a_int, buf_logits)
            buf_logprob = buf_log_pmf.gather(-1, buf_a_val[:, :1]).squeeze(-1)

            buf_r_sum, buf_advantage = self.compute_reward(self, buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_a_prob, buf_a_val, buf_log_pmf, buf_logits

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = new_logprob = None
        for update_c in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        logging_tuple = (obj_critic.item(), obj_actor.item(), new_logprob.mean().item())
        return logging_tuple


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = 1 if if_discrete else action_dim  # for self.sample_all(

        other_dim = 1 + 1 + self.action_dim + action_dim
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


class ReplayBuffer2:
    def __init__(self, max_len, state_dim, action_dim, if_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = 1 if if_discrete else action_dim  # for self.sample_all(

        other_dim = 1 + 1 + self.action_dim
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


class ReplayBufferPPO:
    def __init__(self, max_len, state_dim, action_dim, if_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.trajectory_len_list = []
        self.if_full = False
        self.action_dim = 1 if if_discrete else action_dim  # for self.sample_all(
        self.state_dim = state_dim
        self.other_dim = 1 + 1 + self.action_dim + action_dim
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        self.buf_other = np.empty((max_len, self.other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, self.state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other
        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = self.max_len
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def out_trajectory(self, len_trajectory):
        if len(self.trajectory_len_list) == 0:
            return
        outlen = self.trajectory_len_list.pop(0)
        len_trajectory -= outlen
        while ((self.trajectory_len_list[0] < len_trajectory) and \
               (sum(self.trajectory_len_list) > (self.max_len - 2000))) or \
                (sum(self.trajectory_len_list) > (self.max_len - 1000)):
            l = self.trajectory_len_list.pop(0)
            len_trajectory -= l
            outlen += l
        self.next_idx = self.next_idx - outlen
        self.now_len = self.now_len - outlen
        self.buf_state = np.delete(self.buf_state, [range(outlen)], 0)
        self.buf_other = np.delete(self.buf_other, [range(outlen)], 0)
        self.buf_state = np.vstack((self.buf_state, np.empty((outlen, self.state_dim), dtype=np.float32)))
        self.buf_other = np.vstack((self.buf_other, np.empty((outlen, self.other_dim), dtype=np.float32)))
        self.if_full = False


'''Utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # a early time
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")

    def evaluate_save(self, act, steps, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()

            rewards_steps_list = [get_episode_return(self.env, act, self.device) for _ in range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return(self.env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)
                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))  # update recorder

            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"  {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
                      f"  {self.used_time:>8}  ########")

            # plan to
            # if time.time() - self.print_time > self.show_gap:
            print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)}")
        else:
            if_reach_goal = False

        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step + 1


'''env.py'''


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super(PreprocessEnv, self).__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

        self.reset = self.reset_type
        self.step = self.step_type

    def reset_type(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.
    env_name: the environment name, such as XxxXxx-v0
    state_dim: the dimension of state
    action_dim: the dimension of continuous action; Or the number of discrete action
    action_max: the max action of continuous action; action_max == 1 when it is discrete action space
    if_discrete: Is this env a discrete action space?
    target_return: the target episode return, if agent reach this score, then it pass this game (env).
    max_step: the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step

    :env: a standard OpenAI gym environment, it has env.reset() and env.step()
    :bool if_print: print the information of environment. Such as env_name, state_dim ...
    """
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval


'''DEMO'''


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training (off-policy)'''
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 5  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_remove = None  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

    def init_before_training(self, process_id=0):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.gpu_id is None:  # set gpu_id as '0' in default
            self.gpu_id = '0'

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            # import datetime
            # curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # self.cwd = f'./{self.env.env_name}_{agent_name}_' + curr_time
            self.cwd = f'./{self.env.env_name}_{agent_name}'

        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        gpu_id = self.gpu_id[process_id] if isinstance(self.gpu_id, list) else self.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    # max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    learning_rate = args.learning_rate
    if_per_or_gae = args.if_per_or_gae
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)

    buffer_len = target_step + max_step
    buffer = ReplayBuffer(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim,
                          if_discrete=if_discrete)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    total_step = 0

    '''start training'''
    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        steps = len(trajectory_list)
        total_step += steps

        buffer.empty_buffer()
        buffer.extend_buffer_from_list(trajectory_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
            if_train = not ((if_break_early and if_reach_goal)
                            or total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


def train_and_evaluate_op2(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    # max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    learning_rate = args.learning_rate
    if_per_or_gae = args.if_per_or_gae
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)

    buffer_len = target_step + max_step
    buffer = ReplayBuffer2(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim,
                           if_discrete=if_discrete)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    total_step = 0

    '''start training'''
    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        steps = len(trajectory_list)
        total_step += steps

        buffer.extend_buffer_from_list(trajectory_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
            if_train = not ((if_break_early and if_reach_goal)
                            or total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


def train_and_evaluate_offpolicy(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    # max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    learning_rate = args.learning_rate
    if_per_or_gae = args.if_per_or_gae
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)

    buffer_len = target_step + max_step
    buffer = ReplayBufferPPO(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim,
                             if_discrete=if_discrete)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    total_step = 0

    '''start training'''
    if_train = True
    if_first = True
    while if_train:
        with torch.no_grad():
            if if_first:
                trajectory_list, trajectory_len_list = agent.explore_env1(env, target_step, reward_scale, gamma)
                if_first = False
                buffer.extend_buffer_from_list(trajectory_list)
                buffer.trajectory_len_list = trajectory_len_list
            else:
                trajectory_list, trajectory_len_list = agent.explore_env2(env, reward_scale, gamma)
                buffer.out_trajectory(len(trajectory_list))
                buffer.extend_buffer_from_list(trajectory_list)
                buffer.trajectory_len_list += trajectory_len_list
        steps = len(trajectory_list)
        total_step += steps
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)
        i = 0
        if total_step // target_step > i:
            i += 1
            with torch.no_grad():
                if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
                if_train = not ((if_break_early and if_reach_goal)
                                or total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


def demo_continuous_action_offppo():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    # args.agent = AgentPPO()
    # args.agent = AgentTargetPPO()
    args.agent = AgentOffPolicyPPO()

    '''choose environment'''
    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env='Pendulum-v0')
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 1
        args.eval_times2 = 1
    else:
        # args.env = PreprocessEnv(env='Swimmer-v2')
        # args.env = PreprocessEnv(env='HalfCheetah-v2')
        # args.env = PreprocessEnv(env='Walker2d-v2')
        args.env = PreprocessEnv(env='Hopper-v2')
        # args.env = PreprocessEnv(env='Humanoid-v2')
        args.env.target_return = 3000  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** 0  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 2
        args.target_step = args.env.max_step * 8
        args.max_memo = args.target_step + 2 * args.env.max_step
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 2
        args.eval_times2 = 4

    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302

    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334

    '''train and evaluate'''
    train_and_evaluate_offpolicy(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


def demo_continuous_action():
    args = Arguments(if_on_policy=False)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    # args.agent = AgentTargetPPO()

    '''choose environment'''
    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env='Pendulum-v0')
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 1
        args.eval_times2 = 1
    else:
        # args.env = PreprocessEnv(env='Swimmer-v2')
        # args.env = PreprocessEnv(env='HalfCheetah-v2')
        # args.env = PreprocessEnv(env='Walker2d-v2')
        args.env = PreprocessEnv(env='Hopper-v2')
        # args.env = PreprocessEnv(env='Humanoid-v2')
        args.env.target_return = 3000  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** 0  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 8
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 2
        args.eval_times2 = 4

    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302

    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334

    '''train and evaluate'''
    train_and_evaluate(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


def demo_continuous_action():
    args = Arguments(if_on_policy=False)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    # args.agent = AgentTargetPPO()

    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env='Pendulum-v0')
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 1
        args.eval_times2 = 1
    else:
        # args.env = PreprocessEnv(env='Swimmer-v2')
        # args.env = PreprocessEnv(env='HalfCheetah-v2')
        # args.env = PreprocessEnv(env='Walker2d-v2')
        args.env = PreprocessEnv(env='Hopper-v2')
        # args.env = PreprocessEnv(env='Humanoid-v2')
        args.env.target_return = 3000  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** 0  # RewardRange: -1800 < -200 < -50 < 0
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 8
        args.if_allow_break = False
        args.break_step = int(4e5 * 8)
        args.gpu_id = 1
        args.random_seed = 123

        args.eval_gap = 2 ** 7
        args.eval_times1 = 2
        args.eval_times2 = 4

    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302

    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334

    '''train and evaluate'''
    train_and_evaluate(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


def demo_continuous_action_off():
    args = Arguments(if_on_policy=False)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentoffPPO()
    # args.agent = AgentTargetPPO()

    '''choose environment'''

    # args.env = PreprocessEnv(env='Swimmer-v2')
    # args.env = PreprocessEnv(env='HalfCheetah-v2')
    # args.env = PreprocessEnv(env='Walker2d-v2')
    args.env = PreprocessEnv(env='Hopper-v2')
    # args.env = PreprocessEnv(env='Humanoid-v2')
    args.env.target_return = 3000  # set target_reward manually for env 'Pendulum-v0'
    args.reward_scale = 2 ** 0  # RewardRange: -1800 < -200 < -50 < 0
    args.net_dim = 2 ** 8
    args.repeat_times = 2 ** 3
    args.batch_size = args.net_dim * 2
    args.target_step = args.env.max_step
    args.max_memo = 2 ** 18
    args.if_allow_break = False
    args.break_step = int(4e5 * 8)
    args.gpu_id = 1
    args.random_seed = 123

    args.eval_gap = 2 ** 7
    args.eval_times1 = 2
    args.eval_times2 = 4

    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302

    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334

    '''train and evaluate'''
    train_and_evaluate_op2(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()

    '''choose environment'''
    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step * 8
        args.learning_rate = 2 ** -12
        args.if_per_or_gae = True
        args.if_allow_break = True
        args.break_step = int(4e5 * 8)
        args.gpu_id = 2
        args.random_seed = 1237643

        args.eval_gap = 2 ** 6
        args.eval_times1 = 3
        args.eval_times2 = 5

    '''train and evaluate'''
    train_and_evaluate(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


if __name__ == '__main__':
    # demo_discrete_action()
    demo_continuous_action()
    # demo_continuous_action_offppo()
    # demo_continuous_action_off()
