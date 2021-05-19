from ray_elegantrl.interaction import beginer
import ray


def sumo_trafficlight_ppo(config=None):
    from ray_elegantrl.configs.configs_ppo import config as configs
    from sumo_env.config import params
    params['reward_weights'] = [1., config['reward_weights']]
    env = {
        'id': 'sumoEnv-v0',
        'state_dim': 4,
        'action_dim': 1,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 30,
        'max_step': 150,
        'params_name': {'params': params}
    }
    # from ray_elegantrl.agent import AgentPPO_MO
    # config['agent']['class_name'] = AgentPPO_MO
    # config['agent']['lambda_entropy'] = 0.02
    # config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['if_auto_weights'] = True
    # config['agent']['weights'] = [1., 1.]
    # config['interactor']['gamma'] = [0.99, 0.96]
    from ray_elegantrl.agent import AgentPPO
    configs['agent']['class_name'] = AgentPPO
    configs['interactor']['gamma'] = [config['gamma']]
    configs['agent']['net_dim'] = 2 ** 6
    configs['interactor']['rollout_num'] = 1
    configs['trainer']['sample_step'] = 20 * env['max_step'] * configs['interactor']['rollout_num']
    configs['trainer']['batch_size'] = 2 ** 8
    configs['interactor']['horizon_step'] = configs['trainer']['sample_step']
    configs['trainer']['policy_reuse'] = 2 ** 3
    configs['evaluator']['break_step'] = int(1e6)
    configs['evaluator']['pre_eval_times'] = 20
    configs['evaluator']['eval_times'] = 0
    configs['buffer']['max_buf'] = configs['interactor']['horizon_step']
    configs['env'] = env
    configs['gpu_id'] = 0
    configs['if_cwd_time'] = True
    configs['random_seed'] = 0
    beginer(configs)


def sumo_trafficlight_ppomo(config=None):
    from ray_elegantrl.configs.configs_ppo import config as configs
    from sumo_env.config import params
    params['is_sum_reward'] = False
    params['reward_weights'] = [1., config['reward_weights']]
    env = {
        'id': 'sumoEnv-v0',
        'state_dim': 4,
        'action_dim': 1,
        'if_discrete_action': False,
        'reward_dim': 2,
        'target_reward': 30,
        'max_step': 150,
        'params_name': {'params': params}
    }
    from ray_elegantrl.agent import AgentPPO_MO2
    configs['agent']['class_name'] = AgentPPO_MO2
    configs['agent']['lambda_entropy'] = 0.02
    configs['agent']['lambda_gae_adv'] = 0.97
    configs['agent']['if_auto_weights'] = True
    configs['agent']['weights'] = [1., 1.]
    configs['agent']['pid_Ki'] = 0.01
    configs['agent']['pid_Kp'] = 1
    configs['agent']['pid_Kd'] = 4
    configs['interactor']['gamma'] = [config['gamma_0'], config['gamma_1']]
    # from ray_elegantrl.agent import AgentPPO
    # configs['agent']['class_name'] = AgentPPO
    # configs['interactor']['gamma'] = [config['gamma']]
    configs['agent']['net_dim'] = 2 ** 6
    configs['interactor']['rollout_num'] = 1
    configs['trainer']['sample_step'] = 20 * env['max_step'] * configs['interactor']['rollout_num']
    configs['trainer']['batch_size'] = 2 ** 8
    configs['interactor']['horizon_step'] = configs['trainer']['sample_step']
    configs['trainer']['policy_reuse'] = 2 ** 3
    configs['evaluator']['break_step'] = int(1e6)
    configs['evaluator']['pre_eval_times'] = 20
    configs['evaluator']['eval_times'] = 0
    configs['buffer']['max_buf'] = configs['interactor']['horizon_step']
    configs['env'] = env
    configs['gpu_id'] = 2
    configs['if_cwd_time'] = True
    configs['random_seed'] = 0
    beginer(configs)


if __name__ == '__main__':
    import os
    from ray import tune
    import numpy as np
    import traceback

    # ray.init(address='172.31.233.205:9998', _redis_password='5241590000000000')
    # ray.init(num_cpus=18)
    # ray.init(local_mode=True)
    # config = {'gamma_0': 0.99, 'gamma_1': 0.95, 'reward_weights': 10.}
    # sumo_trafficlight_ppomo(config=config)

    try:
        ray.init(num_cpus=12)
        tune.run(
            tune.with_parameters(sumo_trafficlight_ppo),
            local_dir=os.path.dirname(os.path.dirname(__file__)) + '/ray_results',
            name='sumo_trafficlight_ppo',
            config={'gamma': tune.grid_search([0.95, 0.96, 0.97, 0.98, 0.99, 0.995]),
                    'reward_weights': tune.grid_search([10])},
            resources_per_trial={'cpu': 1, 'extra_cpu': 1},
        )
    except Exception:
        traceback.print_exc()
    finally:
        ray.shutdown()

    try:
        ray.init(num_cpus=12)
        tune.run(
            tune.with_parameters(sumo_trafficlight_ppomo),
            local_dir=os.path.dirname(os.path.dirname(__file__)) + '/ray_results',
            name='sumo_trafficlight_ppomo',
            config={'gamma_0': tune.grid_search([0.95, 0.96, 0.97, 0.98, 0.99, 0.995]),
                    'gamma_1': tune.grid_search([0.95, 0.96, 0.97, 0.98, 0.99, 0.995]),  # , 0.98, 0.99, 0.995
                    'reward_weights': tune.grid_search([10])},
            resources_per_trial={'cpu': 1, 'extra_cpu': 1},
        )
    except Exception:
        traceback.print_exc()
    finally:
        ray.shutdown()
