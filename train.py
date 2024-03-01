import numpy as np
import pickle as pkl
import gymnasium as gym
import d3rlpy
import os, sys, datetime
import h5py
from tqdm import tqdm
# from d4rl import d4rl
from d3rlpy.datasets import get_cartpole # CartPole-v1 dataset
from d3rlpy.dataset import MDPDataset, create_fifo_replay_buffer

from d3rlpy.algos import CQLConfig, DiscreteCQLConfig
from d3rlpy.metrics import *

def cartpole():
    dataset, env = get_cartpole()

    # if you don't use GPU, set device=None instead.
    cql = DiscreteCQLConfig().create(device=None)

    # set environment in scorer function
    env_evaluator = EnvironmentEvaluator(env)
    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

    cql.fit(
        dataset,
        n_steps=10000,
        evaluators={
            'td_error': td_error_evaluator,
            'environment': env_evaluator,
        },
    )

    rewards = env_evaluator(cql, dataset=None)
    observation, _ = env.reset()
    print('='*100)
    print(f'Mean episode return: {evaluate_qlearning_with_environment(cql, env)}')
    print('='*100)
    # return actions based on the greedy-policy
    action = cql.predict(np.expand_dims(observation, axis=0))

    # estimate action-values
    value = cql.predict_value(np.expand_dims(observation, axis=0), action)
    print(f'Observation: {observation}')
    print('='*100)
    print(f'Action: {action}')
    print('='*100)
    print(f'Value: {value}')
    print('='*100)
    print(f'Rewards: {rewards}')
    # save full parameters and configurations in a single file.
    cql.save('cql/cql.d3')
    # load full parameters and build algorithm
    # cql2 = d3rlpy.load_learnable("cql/cql.d3")

    # # save full parameters only
    # cql.save_model('cql/cql.pt')
    # # load full parameters with manual setup
    # cql3 = CQLConfig().create(device="cpu")
    # cql3.build_with_dataset(dataset)
    # cql3.load_model('cql/cql.pt')

    # # save the greedy-policy as TorchScript
    # cql.save_policy('cql/policy.pt')

def d4rl():
    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys
    env = gym.make('Hopper-v4')
    # dataset = d4rl.qlearning_dataset(env)
    h5path = os.path.dirname(os.path.realpath(__file__)) + '/data/hopper_expert-v2.hdf5'
    print(h5path)
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        print(dataset_file.keys())
        print(dataset_file['observations'])
        # for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        #     try:  # first try loading as an array
        #         data_dict[k] = dataset_file[k][:]
        #     except ValueError as e:  # try loading as a scalar
        #         data_dict[k] = dataset_file[k][()]

        # episode_terminals = np.logical_or(dataset_file['terminals'][()], dataset_file['timeouts'][()])
        print(np.array(dataset_file['observations'][()]).shape, np.array(dataset_file['actions'][()]).shape, np.array(dataset_file['rewards'][()]).shape, np.array(dataset_file['terminals'][()]).shape)
        dataset = MDPDataset(dataset_file['observations'][()], dataset_file['actions'][()], dataset_file['rewards'][()], dataset_file['terminals'][()], dataset_file['timeouts'][()])
    
    cql = CQLConfig().create(device=None)
    env_evaluator = EnvironmentEvaluator(env)
    cql.fit(
        dataset,
        n_steps=1000,
        evaluators={
            'td_error': TDErrorEvaluator(),
            'environment': env_evaluator,
        },
    )
    print(f'Mean episode return: {evaluate_qlearning_with_environment(cql, env)}')

def gym_ca(device='cuda', save=False):
    from gym_ca.gym_collision_avoidance.experiments.src.env_utils import create_env
    from gym_ca.gym_collision_avoidance.envs import Config

    env = create_env()

    pkl_path = os.path.dirname(os.path.realpath(__file__)) + '/data/RVO_2_agent_5000/d4rl.p'
    print('Loading data from ' + pkl_path)
    
    with open(pkl_path, 'rb') as f:
        trajectories = pkl.load(f)
        # print(len(trajectories['observations']), trajectories.keys())
    obs, act, rew, ter, tim = [[] for _ in range(5)]
    obs_dim, num_agents = trajectories['observations'][0].shape[2], trajectories['observations'][0].shape[1]
    act_dim = trajectories['actions'][0].shape[2]

    assert Config.MAX_NUM_AGENTS_IN_ENVIRONMENT == num_agents, f'{Config.MAX_NUM_AGENTS_IN_ENVIRONMENT} != {num_agents}'
    
    for i in range(len(trajectories['observations'])):
        obs.extend(trajectories['observations'][i].reshape(-1, obs_dim)[0::num_agents])
        act.extend(trajectories['actions'][i].reshape(-1, act_dim)[0::num_agents])
        rew.extend(np.array(trajectories['rewards'][i].reshape(-1,1)[0::num_agents]).reshape(-1))
        flags = [False]*trajectories['observations'][i].shape[0]
        flags[-1] = trajectories['terminals'][i][0]
        ter.extend(np.array(flags))
        flags[-1] = trajectories['timeouts'][i][0]
        tim.extend(np.array(flags))

    print(np.array(obs).shape, np.array(act).shape, np.array(rew).shape, np.array(ter).shape, np.array(tim).shape)
    assert len(obs) == len(act)
    assert len(act) == len(rew)
    assert len(rew) == len(ter)
    assert len(ter) == len(tim)
    avg_dataset_rew = np.sum(np.array(rew))/len(trajectories['observations'])

    dataset = MDPDataset(np.array(obs), np.array(act), np.array(rew), np.array(ter), np.array(tim))
    
    cql = CQLConfig(
        alpha_threshold=20.0,
        initial_alpha=0.5
    ).create(device=device)
    
    result = cql.fit(
        dataset,
        n_steps=2000000,
        n_steps_per_epoch=10000,
        save_interval=10,
        evaluators={
            'td_error': TDErrorEvaluator(),
            'init_state_val_est': InitialStateValueEstimationEvaluator(),
            'avg_val_est': AverageValueEstimationEvaluator(),
            'cont_action_diff': ContinuousActionDiffEvaluator(),
            'environment': EnvironmentEvaluator(env),
        },
    )

    print(f'Dataset mean episode return: {avg_dataset_rew}')
    print(f'Mean episode return: {evaluate_qlearning_with_environment(cql, env)}')
    if save:
        cql.save_model(f'models/{num_agents}_agent_cql-{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pt')
        print(f'Model saved to models/{num_agents}_agent_cql-{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pt')

def online_gym_ca(device='cuda', save=False):
    from gym_ca.gym_collision_avoidance.experiments.src.env_utils import create_env
    from gym_ca.gym_collision_avoidance.envs import Config
    assert Config.TRAIN_SINGLE_AGENT == True
    env = create_env()
    eval_env = create_env()

    num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT

    buffer = create_fifo_replay_buffer(limit=100000, env=env)
    
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(duration=10000)

    cql = CQLConfig(
        alpha_threshold=50.0
    ).create(device=device)
    
    result = cql.fit_online(
        env=env, 
        buffer=buffer, 
        explorer=explorer, 
        eval_env=eval_env,
        n_steps=3000,
        n_steps_per_epoch=1000,
        update_interval=10
    )
    print(result)
    print(f'Mean episode return: {evaluate_qlearning_with_environment(cql, env)}')
    if save:
        cql.save_model(f'models/{num_agents}_agent_cql-{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pt')
        print(f'Model saved to models/{num_agents}_agent_cql-{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pt')

# cartpole()
# d4rl()
# gym_ca()
online_gym_ca()