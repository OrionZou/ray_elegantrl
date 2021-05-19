from ray_elegantrl.interaction import beginer
import ray


def mujoco_ppo():
    from ray_elegantrl.configs.configs_ppo import config

    env = {
        'id': 'Hopper-v2',  # LunarLanderContinuous-v2 Hopper-v2
        'state_dim': 11,
        'action_dim': 3,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 3600,
        'max_step': 1024,
        # 'params_name': {'name': 'Safexp-CarGoal0-v0'}
    }
    config['agent']['ratio_clip'] = 0.2
    config['agent']['lambda_gae_adv'] = 0.97
    config['interactor']['rollout_num'] = 4
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def UnSafexp_ppo():
    from ray_elegantrl.configs.configs_ppo import config
    # env = {
    #     'id': 'Safexp-PointGoal0-v0',
    #     'state_dim': 28,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1024,
    #     # 'params_name': {'name': 'Safexp-CarGoal0-v0'}
    # }
    # env = {
    #     'id': 'Safexp-CarGoal1-v0',
    #     'state_dim': 72,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1024,
    # }
    # env = {
    #     'id': 'Safexp-PointGoal1-v0',
    #     'state_dim': 60,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1000,
    # }
    env = {
        'id': 'UnSafety-v0',
        'state_dim': 60,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 30,
        'max_step': 1000,
        'params_name': {'name': 'Safexp-PointGoal1-v0',
                        'is_cost_end': False}
    }
    # env = {
    #     'id': 'SafetyT-v0',
    #     'state_dim': 60,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1000,
    #     'params_name': {'name': 'Safexp-PointGoal1-v0'}
    # }
    # env = {
    #     'id': 'UnSafety-v0',
    #     'state_dim': 72,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1024,
    #     'params_name': {'name': 'Safexp-CarGoal1-v0'}
    # }
    config['agent']['ratio_clip'] = 0.2
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['interactor']['rollout_num'] = 10
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 10
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 1
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['agent']['total_iterations'] = config['evaluator']['break_step'] / config['trainer']['sample_step']
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def Safexp_ppo():
    from ray_elegantrl.configs.configs_ppo import config
    env = {
        'id': 'Safety-v0',
        'state_dim': 60,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 30,
        'max_step': 1000,
        'params_name': {'name': 'Safexp-PointGoal1-v0',
                        'is_cost_end': False}
    }
    # env = {
    #     'id': 'Safety-v0',
    #     'state_dim': 72,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 30,
    #     'max_step': 1024,
    #     'params_name': {'name': 'Safexp-CarGoal1-v0'}
    # }
    config['agent']['ratio_clip'] = 0.2
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['interactor']['rollout_num'] = 10
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 10
    config['interactor']['horizon_step'] = config['trainer']['sample_step']

    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 1
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def Safexp2_ppo():
    from ray_elegantrl.configs.configs_ppo import config
    env = {
        'id': 'SafetyT-v0',
        'state_dim': 60,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 2,
        'target_reward': 30,
        'max_step': 1000,
        'params_name': {'name': 'Safexp-PointGoal1-v0',
                        'is_cost_end': True}
    }
    # env = {
    #     'id': 'SafetyT-v0',
    #     'state_dim': 72,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 2,
    #     'target_reward': 30,
    #     'max_step': 1024,
    #     'params_name': {'name': 'Safexp-CarGoal1-v0'}
    # }
    from ray_elegantrl.agent import AgentPPO_MO2
    config['agent']['class_name'] = AgentPPO_MO2
    config['agent']['ratio_clip'] = 0.2
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['if_auto_weights'] = True
    config['agent']['weights'] = [1., 1.]
    config['agent']['pid_Ki'] = 0.01
    config['agent']['pid_Kp'] = 1
    config['agent']['pid_Kd'] = 4
    config['interactor']['rollout_num'] = 10
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 10
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = [0.99, 0.96]
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 1
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    # import gym
    #
    # env = gym.make('Safexp-PointGoal1-v0')
    # obs = env.reset()
    #
    # done = False
    # while not done:
    #     next_obs, reward, done, info = env.step(env.action_space.sample())
    #     print(obs)
    # from gym import envs
    # print(envs.registry.all())
    # ray.init(address='172.31.233.205:9998', _redis_password='5241590000000000')
    ray.init()
    # ray.init(num_cpus=10)
    # ray.init(local_mode=True)
    # ray.init(num_cpus=12,num_gpus=2)
    # ray.init(num_cpus=12, num_gpus=0)
    # UnSafexp_ppo()
    # Safexp_ppo()
    Safexp2_ppo()
    # mujoco_ppo()