import d3rlpy
import gym
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
from d3rlpy.algos import IQLConfig, CQLConfig
import pickle
from d3rlpy.metrics import EnvironmentEvaluator, DiscreteActionMatchEvaluator, \
    TDErrorEvaluator, AverageValueEstimationEvaluator
from d3rlpy.preprocessing import action_scalers
import utils
from config import *
import stable_baselines3 as sb3
from escnn_model import *


def main(args):
    WANDB_CONFIG = {
        "escnn": False,
        "augmentation": False,
        "task": args.task,
        "algorithm": args.algorithm,
        "n_epochs": args.n_epochs,
        "probs": args.probs,
        "dataset_type": args.dataset_type,
        'seed': args.seed,
        'actor_learning_rate': args.actor_learning_rate,
        'critic_learning_rate': args.critic_learning_rate,
        # 'alpha': args.alpha,
        'expectile': args.expectile,
        'n_critic': args.n_critics
    }
    # WANDB_CONFIG.update({'model_config': model_config})
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    # wandb.init(
    #     job_type='Data augmentation',
    #     project='debugging',
    #     # project='Trifinger_' + args.task +'_Dataaugmentation',
    #     config=WANDB_CONFIG,
    #     sync_tensorboard=True,
    #     # entity='Symmetry_RL',
    #     name='train_' + args.task + '_' + args.algorithm + '_' + args.dataset_type + '_'
    #          + str(int(100*args.probs)) + '_' + str(args.n_epochs) + 'epochs' + now,
    #     # notes = 'some notes related',
    #     ####
    # )

    if args.task == "push":
        env_name = "trifinger-cube-push-sim-expert-v0"
    elif args.task == "lift":
        env_name = "trifinger-cube-lift-sim-expert-v0"
    else:
        print("Invalid task %s" % args.task)

    env = gym.make(
        env_name,
        disable_env_checker=True,
        # flatten_obs=False,
        flatten_obs=True,
        # obs_to_keep=obs_to_keep,
        # visualization=True,  # enable visualization
    )
    # Type here should be encoder factory !!!
    actor_encoder_factory = MainEncoderFactory(trifinger_actor_emlp_args)
    critic_encoder_factory = MainEncoderFactory(trifinger_critic_emlp_args)
    value_encoder_factory = MainEncoderFactory(trifinger_value_emlp_args)

    dataset = env.get_dataset()
    dataset = MDPDataset(dataset['observations'], dataset['actions'], dataset['rewards'], dataset['timeouts'])

    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    iql = IQLConfig(
        actor_encoder_factory=actor_encoder_factory,
        critic_encoder_factory=critic_encoder_factory,
        value_encoder_factory=value_encoder_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
        action_scaler=action_scaler,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        n_critics=args.n_critics,
        expectile=args.expectile,
    ).create(device=None)

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.

    iql.build_with_dataset(dataset)

    results = iql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        # n_epochs=10,
        evaluators={
            'reward': EnvironmentEvaluator(env),
            # 'td_error': TDErrorEvaluator(episodes=dataset.episodes)
        },
    )

    # results = [result[1] for result in results]
    # for result in results:
    #     wandb.log({**result})

    print('results: ', results)
    print('finishing fitting!!')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default='lift', choices=["push", "lift"],
                        help="Which task to evaluate ('push' or 'lift').", )
    parser.add_argument("--algorithm", type=str, default='iql', choices=
    ["bc", "td3+bc", "iql", "cql", "awac", "bcq", "bear", "crr", "plas", "plaswp"],
                        help="Which algorithm to train ('push' or 'lift').", )
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of episodes to run. Default: %(default)s", )
    parser.add_argument("--seed", type=int, default=168,
                        help="Random seed number!", )
    parser.add_argument("--probs",  type=float, default=1.0,
                        help="Percentage of truncated data.",)
    parser.add_argument("--dataset_type",  type=str, default='raw', choices=["raw", "aug"],
                        help="Whether using raw dataset or using augmented dataset",)
    parser.add_argument("--actor_learning_rate",  type=float, default=0.0003,
                        help="Actor learning rate.",)
    parser.add_argument("--critic_learning_rate",  type=float, default=0.0003,
                        help="Critic learning rate.",)
    parser.add_argument("--n_critics", type=int, default=2,
                        help='the number of Q functions for ensemble')
    # parser.add_argument("--alpha",  type=float, default=2.5, help="Alpha value for calculating actor loss!",)
    parser.add_argument("--expectile",  type=float, default=0.7, help="the expectile value for value function training",)
    parser.add_argument('--escnn', '-e', action='store_true')
    parser.add_argument('--augmentation', '-a', action='store_true')

    # parser.add_argument("--policy_path", type=str, help="The path of trained model",)
    # parser.add_argument("--visualization", "-v", action="store_true",
    #     help="Enable visualization of environment.",)
    # parser.add_argument("--n_episodes", type=int, default=64,
    #     help="Number of episodes to run. Default: %(default)s",)
    # parser.add_argument("--output", type=pathlib.Path, metavar="FILENAME",
    #     help="Save results to a JSON file.",)

    args = parser.parse_args()
    main(args)

