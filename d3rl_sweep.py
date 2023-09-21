import d3rlpy
import gym
import gc
import seaborn as sns
import wandb
import os
import datetime
import pathlib
import logging, argparse
import numpy as np
from d3rlpy.dataset import MDPDataset
import rrc_2022_datasets
from rrc_2022_datasets import TriFingerDatasetEnv
from sklearn.model_selection import train_test_split
from d3rlpy.algos import BC, TD3PlusBC, IQL, CQL, AWAC, \
    BCQ, BEAR, CRR, PLAS, PLASWithPerturbation
import pickle
from d3rlpy.metrics.scorer import evaluate_on_environment, continuous_action_diff_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.preprocessing import action_scalers
import utils
from config import *


def main(config=None):
    wandb.init(config=None)
    config = wandb.config
    print(config)
    env_name = "trifinger-cube-lift-sim-expert-v0"
    env = gym.make(
        env_name,
        disable_env_checker=True,
        flatten_obs=True,
    )
    # dataset = env.get_dataset()
    file_name = 'dataset/lift_30_raw.npy'
    dataset = np.load(file_name, allow_pickle=True).item()
    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    timeouts = dataset['timeouts']
    dataset = MDPDataset(obs, actions, rewards, timeouts)
    train_episodes, test_episodes = dataset[:-3], dataset[-3:]
    # train_episodes, test_episodes = train_test_split(dataset, test_size=0.001)

    model = IQL(
        # seed=config.seed,
        learning_rate=config.lr,
        # batch_size=config.batch_size,
        use_gpu=False,
        scaler=config.scaler,
        action_scaler=config.action_scaler,
    )

    model.build_with_dataset(dataset)
    evaluate_scorer = evaluate_on_environment(env)

    results = model.fit(
        train_episodes,
        n_epochs=10,
        eval_episodes=test_episodes,
        scorers={
            'return': evaluate_scorer,
            'val_loss': continuous_action_diff_scorer,
        })

    results = [result[1] for result in results]
    for result in results:
        wandb.log({**result})
    print('results:  ', results)
    print('finishing fitting!!')
    # model_path = 'lift_bc_model.pth'
    # policy_path = 'lift_bc_policy.pth'
    # model.save_model(os.path.join(wandb.run.dir, model_path))
    # model.save_policy(os.path.join(wandb.run.dir, policy_path))
    del dataset, model
    gc.collect()
    wandb.finish()


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project=SWEEP_PROJECT_NAME+'FINAL')
    wandb.agent(sweep_id, main)
