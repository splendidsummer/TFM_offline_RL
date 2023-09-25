import pandas as pd
from escnn_model import *
import d3rlpy
import wandb
import gym
import h5py
import numpy as np
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from d3rlpy.algos import DiscreteBCQConfig
from d3rlpy.metrics import EnvironmentEvaluator, DiscreteActionMatchEvaluator
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Offline RL for Cartpole')
parser.add_argument('--seed', type=int, default=168)
parser.add_argument("--algorithm", type=str, choices=
                    ["bc", "td3+bc", "iql", "cql", "awac", "bcq", "bear", "crr", "plas", "plaswp"],
                    default='bc', help="Which algorithm to train ('push' or 'lift').", )
parser.add_argument('--augmentation', '-a', action='store_true')
parser.add_argument('--escnn', '-e',  action='store_true')
parser.add_argument("--train_ratio", type=float, default=0.01, help="Percentage of data split from full trainset.", )
parser.add_argument("--test_ratio", type=float, default=0.2, help="Percentage of data split from full dataset", )
args = parser.parse_args()

now = datetime.now()
now = now.strftime('%m%d%H%M%S')

WANDB_CONFIG = {
    'algorithm': args.algorithm,
    'seed': args.seed,
    'augmentation': args.augmentation,
    'escnn': args.escnn,
    'train_ratio':  args.train_ratio,
    'test_ratio': args.test_ratio,
}

wandb.init(
    project='Cartpole_Offline_BCQ',
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
    bcq = DiscreteBCQConfig(
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        # action_scaler=d3rlpy.preprocessing.MinMaxActionScaler,
        encoder_factory=C2EncoderFactory(),
    ).create(device=None)
else:
    bcq = DiscreteBCQConfig(
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        # action_scaler=d3rlpy.preprocessing.MinMaxActionScaler,
        ).create(device=None)

full_trainset, testset = train_test_split(dataset.episodes, test_size=config.test_ratio, shuffle=False)
# trainset, _ = train_test_split(full_trainset, train_size=config.train_ratio, shuffle=False)

observations, actions, rewards, terminals = [], [], [], []
for episode in full_trainset:
    observations += episode.observations.tolist()
    actions += episode.actions.tolist()
    rewards += episode.rewards.tolist()
    terminals += [0 for _ in range(len(episode.observations.tolist())-1)] + [1]

observations, actions, rewards, terminals = np.array(observations), \
    np.array(actions, dtype=np.int32), np.array(rewards), np.array(terminals)

full_trainset = d3rlpy.dataset.MDPDataset(
    observations=observations, actions=actions, rewards=rewards, terminals=terminals
)

action_match_evaluator = DiscreteActionMatchEvaluator(episodes=testset)


bcq.build_with_dataset(full_trainset)
results = bcq.fit(
    dataset,
    n_steps=2000,
    n_steps_per_epoch=1000,
    # n_epochs=10,
    evaluators={
        'reward': EnvironmentEvaluator(env),
        'action match': action_match_evaluator,
        # 'td_error': TDErrorEvaluator(episodes=dataset.episodes)
    },
)

results = [result[1] for result in results]
num_epochs = len(results)
col_name = list(results[0].keys())
losses = [result['loss'] for result in results]
rewards = [result['reward'] for result in results]
action_matches = [result['action match'] for result in results]

result_dict = {
    'loss': losses,
    'reward': rewards,
    'action_match': action_matches,
}

csv_file = 'augmentation_' + str(config.augmentation) + '_' + \
           'escnn_' + str(config.escnn) + '_' + now + \
           '.csv'

result_df = pd.DataFrame(result_dict)

import os
result_folder = './results/BCQ'
csv_file = os.path.join(result_folder, csv_file)
result_df.to_csv(csv_file)


for result in results:
    wandb.log({**result})


print('results:  ', results)
print('finishing fitting!!}')

