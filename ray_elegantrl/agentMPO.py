from ray_elegantrl.agent import *


class AgentMPO(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        self._num_samples = 64 if args is None else args['num_samples']
        self._per_dim_constraining = True
        self._action_penalization = True

        self.epsilon = 1e-1
        self.epsilon_penalty = 1e-3
        self.epsilon_mean = 1e-2 if args is None else args['epsilon_mean']
        self.epsilon_stddev = 1e-4 if args is None else args['epsilon_stddev']
        self.init_log_temperature = 1. if args is None else args['init_log_temperature']
        self.init_log_alpha_mean = 1. if args is None else args['init_log_alpha_mean']
        self.init_log_alpha_stddev = 10. if args is None else args['init_log_alpha_stddev']
        self.MPO_FLOAT_EPSILON = 1e-8
        self.dual_learning_rate = 1e-2
        self.update_freq = 2
        # self.if_on_policy = True

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = ActorMPO(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        d = self.action_dim if self._per_dim_constraining else 1
        self.log_temperature = self.init_log_temperature * torch.ones(1, dtype=torch.float32, device=self.device)
        self.log_alpha_mean = self.init_log_alpha_mean * torch.ones(d, dtype=torch.float32, device=self.device)
        self.log_alpha_stddev = self.init_log_alpha_stddev * torch.ones(d, dtype=torch.float32, device=self.device)
        self.log_penalty_temperature = self.init_log_temperature * torch.ones(1, dtype=torch.float32,
                                                                              device=self.device)
        self.log_temperature = torch.autograd.Variable(self.log_temperature, requires_grad=True)
        self.log_alpha_mean = torch.autograd.Variable(self.log_alpha_mean, requires_grad=True)
        self.log_alpha_stddev = torch.autograd.Variable(self.log_alpha_stddev, requires_grad=True)
        self.log_penalty_temperature = torch.autograd.Variable(self.log_penalty_temperature, requires_grad=True)
        del self.init_log_temperature, self.init_log_alpha_mean, self.init_log_alpha_stddev
        self.dual_optimizer = torch.optim.Adam([self.log_temperature,
                                                self.log_alpha_mean,
                                                self.log_alpha_stddev,
                                                self.log_penalty_temperature,
                                                ], lr=self.dual_learning_rate)
        # self.act_optimizer = torch.optim.Adam([{'params': self.log_temperature, 'lr': self.dual_learning_rate},
        #                                        {'params': self.log_alpha_mean, 'lr': self.dual_learning_rate},
        #                                        {'params': self.log_alpha_stddev, 'lr': self.dual_learning_rate},
        #                                        {'params': self.act.parameters()},
        #                                        ], lr=self.learning_rate)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.softplus = torch.nn.Softplus(threshold=18)
        # self.criterion = torch.nn.MSELoss()
        self.get_obj_critic = self.get_obj_critic_raw

    @staticmethod
    def select_action(state, policy):
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        action = policy.get_action(states)[0]
        return action.detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len

        # for i in range(int(repeat_times *_target_step)):
        for i in range(int(repeat_times * buf_len / batch_size)):
            # Policy Evaluation
            obj_critic, state, q_label = self.get_obj_critic(buffer, batch_size)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()

            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            # Policy Improvation
            # Sample N additional action for each state
            online_loc, online_cholesky = self.act.get_loc_cholesky(state)  # (B,)
            with torch.no_grad():
                target_loc, target_cholesky = self.act_target.get_loc_cholesky(state)  # (B,)
                target_pi = MultivariateNormal(target_loc, scale_tril=target_cholesky)  # (B,)
                sampled_a = target_pi.sample((self._num_samples,))  # (N, B, dim-a)
                expanded_s = state[None, ...].expand(self._num_samples, -1, -1)  # (N, B, dim-s)
                target_q = self.cri_target.forward(
                    expanded_s.reshape(-1, state.shape[1]),  # (N * B, dim-s)
                    sampled_a.reshape(-1, self.action_dim)  # (N * B, dim-a)
                ).reshape(self._num_samples, batch_size)


            # E-Step
            for _ in range(100):
                # Note: using softplus instead of exponential for numerical stability.
                temperature = self.softplus(input=self.log_temperature) + self.MPO_FLOAT_EPSILON
                alpha_mean = self.softplus(input=self.log_alpha_mean) + self.MPO_FLOAT_EPSILON
                alpha_stddev = self.softplus(input=self.log_alpha_stddev) + self.MPO_FLOAT_EPSILON

                # Computes normalized importance weights for the policy optimization.
                tempered_q_values = target_q / temperature  # no grad
                normalized_weights = torch.softmax(tempered_q_values, dim=0).detach_()  # no grad

                loss_temperature = self.compute_temperature_loss(tempered_q_values, self.epsilon, temperature)
                kl_nonparametric = self.compute_nonparametric_kl_from_normalized_weights(normalized_weights)

                if self._action_penalization:
                    log_penalty_temperature = self.softplus(input=self.log_penalty_temperature) + self.MPO_FLOAT_EPSILON
                    diff_out_of_bound = sampled_a - sampled_a.clip(-1.0, 1.0)
                    cost_out_of_bound = -torch.norm(diff_out_of_bound, dim=-1)
                    tempered_cost_values = cost_out_of_bound / temperature  # no grad
                    normalized_penalty_weights = torch.softmax(tempered_cost_values, dim=0).detach_()  # no grad
                    loss_penalty_temperature = self.compute_temperature_loss(normalized_penalty_weights,
                                                                             self.epsilon_penalty,
                                                                             log_penalty_temperature)
                    loss_temperature += loss_penalty_temperature

                loss_dual = loss_temperature
                self.dual_optimizer.zero_grad()
                loss_dual.backward()
                self.dual_optimizer.step()

            # Decompose the online policy into fixed-mean & fixed-stddev distributions.
            # This has been documented as having better performance in bandit settings,
            # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
            fixed_stddev_dist = MultivariateNormal(online_loc, scale_tril=target_cholesky)
            fixed_mean_dist = MultivariateNormal(target_loc, scale_tril=online_cholesky)

            # Compute the decomposed policy losses.
            loss_policy_mean = self.compute_cross_entropy_loss(
                sampled_a, normalized_weights, fixed_stddev_dist)
            loss_policy_stddev = self.compute_cross_entropy_loss(
                sampled_a, normalized_weights, fixed_mean_dist)

            # Compute the decomposed KL between the target and online policies.
            kl_mean = torch.distributions.kl_divergence(target_pi, fixed_stddev_dist)
            kl_stddev = torch.distributions.kl_divergence(target_pi, fixed_mean_dist)

            # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
            loss_kl_mean, loss_alpha_mean = self.compute_parametric_kl_penalty_and_dual_loss(
                kl_mean, alpha_mean, self.epsilon_mean)
            loss_kl_stddev, loss_alpha_stddev = self.compute_parametric_kl_penalty_and_dual_loss(
                kl_stddev, alpha_stddev, self.epsilon_stddev)

            # Combine losses.
            loss_policy = loss_policy_mean + loss_policy_stddev
            loss_kl_penalty = loss_kl_mean + loss_kl_stddev
            # loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
            loss = loss_policy + loss_kl_penalty
            # loss_dual = loss_alpha_mean + loss_alpha_stddev

            self.act_optimizer.zero_grad()
            loss.backward()
            self.act_optimizer.step()

            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

            self.update_record(d_loc=alpha_mean.mean().item(),
                               d_sca=alpha_stddev.mean().item(),
                               d_eta=temperature.item(),
                               obj_a=loss_policy.item(),
                               obj_c=obj_critic.item(),
                               obj_α=(loss_alpha_mean + loss_alpha_stddev).item(),
                               obj_η=loss_temperature.item(),
                               klq_rel=kl_nonparametric.mean().item() / self.epsilon,
                               max_q=torch.max(target_q, dim=0)[0].mean().item(),
                               min_q=torch.min(target_q, dim=0)[0].mean().item(),
                               )

        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            pi_next_s = self.act_target.get_distribution(next_s)
            sampled_next_a = pi_next_s.sample((self._num_samples,))  # (N, B, dim-action)
            ex_next_s = next_s[None, ...].expand(self._num_samples, -1, -1)  # (N, B, dim-action)
            ex_next_q = self.cri_target(
                ex_next_s.reshape(-1, self.state_dim),
                sampled_next_a.reshape(-1, self.action_dim)
            )
            ex_next_q = ex_next_q.reshape(self._num_samples, batch_size)
            next_q = ex_next_q.mean(dim=0).unsqueeze(dim=1)
            q_label = reward + mask * next_q
        q = self.cri(state, action)
        obj_critic = self.criterion(q, q_label)
        return obj_critic, state, q_label

    def compute_cross_entropy_loss(self,
                                   sampled_actions: torch.Tensor,
                                   normalized_weights: torch.Tensor,
                                   pi: torch.distributions.Distribution) -> torch.Tensor:
        # Compute the M-step loss.
        log_prob = pi.log_prob(sampled_actions)
        # Compute the weighted average log-prob using the normalized weights.
        loss_policy_gradient = -torch.sum(log_prob * normalized_weights, dim=0)
        # Return the mean loss over the batch of states.
        return torch.mean(loss_policy_gradient, dim=0)

    def compute_temperature_loss(self,
                                 tempered_q_values: torch.Tensor,
                                 epsilon: float,
                                 temperature: torch.autograd.Variable) -> [torch.Tensor, torch.Tensor]:
        # q_logsumexp = torch.log(torch.exp(tempered_q_values).mean(dim=0))
        q_logsumexp = torch.logsumexp(tempered_q_values, dim=0)
        log_num_actions = np.log(self._num_samples)
        # loss_temperature = epsilon + torch.mean(q_logsumexp)
        loss_temperature = epsilon + torch.mean(q_logsumexp) - log_num_actions
        loss_temperature = temperature * loss_temperature
        return loss_temperature

    def compute_parametric_kl_penalty_and_dual_loss(self,
                                                    kl: torch.Tensor,
                                                    alpha: torch.autograd.Variable,
                                                    epsilon: float) -> [torch.Tensor, torch.Tensor]:
        # Compute the mean KL over the batch.
        mean_kl = torch.mean(kl, dim=0)
        # Compute the regularization.
        loss_kl = torch.mean(mean_kl * alpha.detach(), dim=0)
        # loss_kl = torch.sum(alpha.detach() * (epsilon - mean_kl), dim=0)
        # Compute the dual loss.
        loss_alpha = torch.sum(alpha * (epsilon - mean_kl.detach()), dim=0)
        return loss_kl, loss_alpha

    def compute_nonparametric_kl_from_normalized_weights(self, normalized_weights: torch.Tensor) -> torch.Tensor:
        """Estimate the actualized KL between the non-parametric and target policies."""
        num_action_samples = normalized_weights.shape[0]
        integrand = torch.log(num_action_samples * normalized_weights + 1e-8)
        return torch.sum(normalized_weights * integrand, dim=0)
