"""
Modify source code
    1. to set escnn==False
    2. Using the full dataset to train CQL

wandb config to watch gradients:
    # for name, param in model.named_parameters():
    #     wandb.log({f'gradients/{name}': wandb.Histogram(param.grad)})
"""

import pandas as pd
from d3rlpy.preprocessing import action_scalers
from datetime import datetime
from escnn_model import *
from gym.envs import classic_control
import d3rlpy
import wandb
import gym
import h5py
import numpy as np
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig, DiscreteBCQConfig
from d3rlpy.metrics import EnvironmentEvaluator, DiscreteActionMatchEvaluator
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Offline RL for Cartpole')
parser.add_argument('--seed', type=int, default=168)
parser.add_argument("--algorithm", type=str, choices=
                    ["bc", "td3+bc", "iql", "cql", "awac", "bcq", "bear", "crr", "plas", "plaswp"],
                    default='bc', help="Which algorithm to train ('push' or 'lift').", )
parser.add_argument('--augmentation', '-a', action='store_true')
# parser.add_argument('--escnn', '-e',  action='store_true')
parser.add_argument("--train_ratio", type=float, default=1.0, help="Percentage of data split from full trainset.", )
parser.add_argument("--test_ratio", type=float, default=1.0, help="Percentage of data split from full dataset", )
args = parser.parse_args()

now = datetime.now()
now = now.strftime('%m%d%H%M%S')

WANDB_CONFIG = {
    'algorithm': args.algorithm,
    'seed': args.seed,
    'augmentation': True,
    'escnn': True,
    'train_ratio':  args.train_ratio,
    'test_ratio': args.test_ratio,
}

wandb.init(
    # project='Cartpole_' + 'Offline' + '_EquivariantNN_Prob',
    project='Cartpole_Offline_CQL_Final',
    config=WANDB_CONFIG,
    entity='unicorn_upc_dl',
    # sync_tensorboard=True,
    name=
    # 'train_' + str(WANDB_CONFIG['train_ratio']) + '_' +
    # 'test_' + str(WANDB_CONFIG['test_ratio']) + '_' +
    'augmentation_' + str(WANDB_CONFIG['augmentation']) + '_' +
    'escnn_' + str(WANDB_CONFIG['escnn']) + '_' + now,
)

config = wandb.config

if config.augmentation:
    dataset_h5 = './d3rlpy_data/cartpole_replay_v1.1.0.h5'
    data_h5_file = h5py.File(dataset_h5, 'r')
    obs, actions, rewards, terminals = np.array(data_h5_file['observations']), np.array(data_h5_file['actions']), \
        np.array(data_h5_file['rewards']), np.array(data_h5_file['terminals'])

    obs = np.concatenate((obs, -1*obs), axis=0)
    actions = np.concatenate((actions, 1 - actions), axis=0, dtype=np.int32)
    rewards = np.concatenate((rewards, rewards), axis=0)
    terminals = np.concatenate((terminals, terminals), axis=0)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs, actions=actions, rewards=rewards, terminals=terminals
    )
    env = gym.make("CartPole-v1")
else:
    # NOTE: now all the branch using data here
    dataset, env = d3rlpy.datasets.get_cartpole()

if config.escnn:
    cql = DiscreteCQLConfig(
        # observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        # action_scaler=d3rlpy.preprocessing.MinMaxActionScaler,
        encoder_factory=CartpoleInvEncoderFactory(),  # actor and critic using the same customized encoder
        escnn=config.escnn,

    ).create(device=None)
else:
    cql = DiscreteCQLConfig(
        # observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        # action_scaler=d3rlpy.preprocessing.MinMaxActionScaler,
        # escnn=config.escnn,
        ).create(device=None)

cql.build_with_dataset(dataset)   # check whether we put the paramters into optimizer
# wandb.watch(cql.impl.modules.q_funcs[0], log='all')

results = cql.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    # n_epochs=10
    evaluators={
        'reward': EnvironmentEvaluator(env),
        # 'action match': action_match_evaluator,
        # 'td_error': TDErrorEvaluator(episodes=dataset.episodes)
    },
)

results = [result[1] for result in results]

# num_epochs = len(results)
# col_name = list(results[0].keys())
# losses = [result['loss'] for result in results]
# rewards = [result['reward'] for result in results]
# action_matches = [result['action match'] for result in results]
# result_dict = {
#     'loss': losses,
#     'reward': rewards,
#     'action_match': action_matches,
# }
#
# csv_file = 'augmentation_' + str(config.augmentation) + '_' + \
#            'escnn_' + str(config.escnn) + '_' + now + \
#            '.csv'
# result_df = pd.DataFrame(result_dict)
# import os
# result_folder = './results/BCQ'
# csv_file = os.path.join(result_folder, csv_file)
# result_df.to_csv(csv_file)

# for result in results:
#     wandb.log({**result})


print('results:  ', results)
print('finishing fitting!!}')
