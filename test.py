import numpy as np
import pickle as pkl
import gymnasium as gym
import pandas as pd
import argparse
import d3rlpy
from tqdm import tqdm

import os, sys, math
from d3rlpy.algos import CQLConfig, DiscreteCQLConfig
from gym_ca.gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
from gym_ca.gym_collision_avoidance.envs import Config
import gym_ca.gym_collision_avoidance.envs.test_cases as tc

def episode_stats(agents, episode_return, episode_length):
    generic_episode_stats = {
        "total_return": episode_return,
        "episode_length": episode_length,
    }

    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array(
        [a.t - a.straight_line_time_to_reach_goal for a in agents]
    )
    collisions, at_goal, stuck = 0 ,0, 0
    for a in agents:
        if a.in_collision:
            collisions += 1
        elif a.is_at_goal:
            at_goal += 1
        else:
            stuck += 1
    collision = np.array(np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])
    ).tolist()
    outcome = (
        "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
    )
    agent_outcome = 'at_goal' if agents[0].is_at_goal else 'collision' if agents[0].in_collision else 'stuck'
    specific_episode_stats = {
        "num_agents": len(agents),
        "time_to_goal": time_to_goal,
        "total_time_to_goal": np.sum(time_to_goal),
        "extra_time_to_goal": extra_time_to_goal,
        "%_collisions": collisions/len(agents),
        "%_at_goal": at_goal/len(agents),
        "%_stuck": stuck/len(agents),
        "agent_outcome": agent_outcome,
        "ep_outcome": outcome,
        "policies": [agent.policy.str for agent in agents],
    }

    # Merge all stats into a single dict
    return {**generic_episode_stats, **specific_episode_stats}

def reset_test(env, case_num, policies):

    def reset_env(env, agents, case_num, policy,):
        env.unwrapped.plot_policy_name = policy        
        env.set_agents(agents)
        init_obs = env.reset()
        env.unwrapped.test_case_index = case_num
        return init_obs, agents

    agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num], policies=policies)
    
    obs, _ = reset_env(env, agents, case_num+1, policy='CQL')

    return obs[0]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
parser.add_argument('--num_tests', '-N', type=int, choices=range(1,500), default=500)
parser.add_argument('--pols', '-p', type=str, default='RVO')
parser.add_argument('--device', '-d', type=str, default='cuda')
args = parser.parse_args()

model = os.path.dirname(os.path.realpath(__file__)) + '/models/' + args.model

print('Loading model ' + model)
try:
    num_agents = int(args.model[0])
    obs_shape = (num_agents-1)*7 + 6
    cql = CQLConfig().create(device=args.device)
    cql.create_impl(observation_shape=[obs_shape,], action_size=2)
    cql.load_model(model)
except:
    print('Model failed to load')
    sys.exit()


policies = [args.pols] * num_agents
policies[0] = 'external'
num_tests = args.num_tests

assert Config.MAX_NUM_AGENTS_IN_ENVIRONMENT == num_agents

test_cases = pd.read_pickle(
    os.path.dirname(os.path.realpath(__file__)) 
    + f'/gym_ca/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
)

env = create_env()
test_save_dir = os.path.dirname(os.path.realpath(__file__))  + f'/test/{args.model.split(".")[0]}/'
os.makedirs(test_save_dir, exist_ok=False)
env.set_plot_save_dir(test_save_dir)
df = pd.DataFrame()

test_rews, test_steps = [], []

with tqdm(
    total=num_tests
) as pbar:
    for i in range(num_tests):
        obs = reset_test(env, i, policies)

        total_reward = 0
        step = 0
        terminated = False
        timeout = False
        while not terminated:
            act = cql.predict(np.array(obs[0]).reshape(1, obs_shape))
            obs, rew, terminated, truncated, info = env.step(act[0])
            total_reward += rew[0]
            step += 1

        pbar.update(1)
        test_rews.append(total_reward), test_steps.append(step)
        ep_stats = episode_stats(env.agents, total_reward, step)
        df = store_stats(
                            df,
                            {"test_episode": i+1},
                            ep_stats,
                        )
_ = env.reset()    # Needed to generate final figure

print('='*50)
print(f'''Model: {args.model}\n'
    Tests: {num_tests}\n' 
    Steps: avg {sum(test_steps)/num_tests:.0f} | min {min(test_steps)} | max {max(test_steps)}\n'
    Reward: avg {sum(test_rews)/num_tests:.2f} | min {min(test_rews):.2f} | max {max(test_rews):.2f}\n''')
print('='*50)

stats_filename = test_save_dir + f"/stats.p"
df.to_pickle(stats_filename)