from ray_elegantrl.interaction import beginer
import ray


def test_mpo():
    from ray_elegantrl.configs.configs_mpo import config
    # env = {
    #     'id': 'Hopper-v2',
    #     'state_dim': 11,
    #     'action_dim': 3,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 3600,
    #     'max_step': 1000,
    # }
    env = {
        'id': 'LunarLanderContinuous-v2',
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 3600,
        'max_step': 500,
    }

    config['interactor']['rollout_num'] = 2
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 2
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['buffer']['max_buf'] = 2 ** 17
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def test_mpo2():
    from ray_elegantrl.configs.configs_mpo2 import config
    env = {
        'id': 'Hopper-v2',
        'state_dim': 11,
        'action_dim': 3,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 3600,
        'max_step': 1000,
    }
    # env = {
    #     'id': 'LunarLanderContinuous-v2',
    #     'state_dim': 8,
    #     'action_dim': 2,
    #     'if_discrete_action': False,
    #     'reward_dim': 1,
    #     'target_reward': 3600,
    #     'max_step': 1000,
    # }

    config['interactor']['rollout_num'] = 4
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e7)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 1
    config['buffer']['max_buf'] = config['trainer']['sample_step']
    # config['buffer']['max_buf'] = 2 ** 19
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    # ray.init()
    ray.init(local_mode=True)
    test_mpo2()
