from ray_elegantrl.interaction import beginer
import ray


def demo_d3qn():
    from ray_elegantrl.configs.configs_d3qn import config
    env = {
        'id': 'LunarLander-v2',
        'state_dim': 8,
        'action_dim': 4,
        'if_discrete_action': True,
        'reward_dim': 1,
        'target_reward': 0,
        'max_step': 500,
    }

    config['interactor']['rollout_num'] = 2
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 1
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e6)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['buffer']['max_buf'] = 2 ** 20
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = False
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    ray.init()
    demo_d3qn()
