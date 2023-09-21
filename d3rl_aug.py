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
from d3rlpy.algos import IQLConfig, CQLConfig, BCConfig, AWACConfig, \
    BCQConfig, TD3PlusBCConfig, BEARConfig
import pickle
from d3rlpy.metrics import EnvironmentEvaluator, DiscreteActionMatchEvaluator, \
    TDErrorEvaluator, AverageValueEstimationEvaluator

from d3rlpy.preprocessing import action_scalers
import utils
from config import *
from escnn_model import *

def main(args):
    WANDB_CONFIG = {
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

    wandb.init(
        job_type='Data augmentation',
        project='debugging_trifinger_offline',
        # project='Trifinger_' + args.task +'_Dataaugmentation',
        config=WANDB_CONFIG,
        sync_tensorboard=True,
        # entity='Symmetry_RL',
        name='train_' + args.task + '_' + args.algorithm + '_' + args.dataset_type + '_'
             + str(int(100*args.probs)) + '_' + str(args.n_epochs) + 'epochs' + now,
        # notes = 'some notes related',
        ####
    )

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

    # dataset = env.get_dataset()

    if args.dataset_type == 'raw':
        file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.npy'
        dataset = np.load(file_name, allow_pickle=True).item()
    elif args.dataset_type == 'aug':
        if args.probs <= 0.30:
            file_name = 'dataset/' + args.task + '_' + str(int(args.probs * 100)) + '_' + args.dataset_type + '.npy'
            dataset = np.load(file_name, allow_pickle=True).item()
        else:
            file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.pck'
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)

    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    timeouts = dataset['timeouts']

    dataset = MDPDataset(obs, actions, rewards, timeouts)
    valset = np.load('dataset/lift_valset.npy', allow_pickle=True).item()
    valset = MDPDataset(valset['observations'], valset['actions'], valset['rewards'], valset['timeouts'])

    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    model = IQLConfig(
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
        action_scaler=action_scaler,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        n_critics=args.n_critics,
        expectile=args.expectile,
    ).create(device=None)

    model.build_with_dataset(dataset)
    wandb.watch(model)

    env_evaluator = EnvironmentEvaluator(env)

    results = model.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        # n_epochs=args.n_epochs,
        evaluators={
            'return': env_evaluator,
            # 'val_actor_loss': continuous_action_diff_scorer,
            # 'val_value_estimations': average_value_estimation_scorer,
        }
    )
    results = [result[1] for result in results]
    for result in results:
        wandb.log({**result})

    print('results:  ', results)
    print('finishing fitting!!')

    # model_path = args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' + \
    #             str(args.n_epochs) + 'epochs' + str(int(args.seed)) + '.pt'

    # policy_path = args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' + \
    #              str(args.n_epochs) + 'epochs' + str(int(args.seed)) + '_policy.pt'

    # model.save_model(os.path.join(wandb.run.dir, model_path))
    # model.save_policy(os.path.join(wandb.run.dir, policy_path))

    # wandb.save(args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' +
    #            str(args.n_epochs) + 'epochs' + '.pt')
    # wandb.save(args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' +
    #            str(args.n_epochs) + 'epochs' + '_policy.pt')
    #


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, choices=["push", "lift"],
                        help="Which task to evaluate ('push' or 'lift').", )
    parser.add_argument("--algorithm", type=str, choices=
    ["bc", "td3+bc", "iql", "cql", "awac", "bcq", "bear", "crr", "plas", "plaswp"],
                        help="Which algorithm to train ('push' or 'lift').", )
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of episodes to run. Default: %(default)s", )
    parser.add_argument("--seed", type=int, default=168,
                        help="Random seed number!", )
    parser.add_argument("--probs",  type=float, default=1.0,
                        help="Percentage of truncated data.",)
    parser.add_argument("--dataset_type",  type=str, choices=["raw", "aug"],
                        help="Whether using raw dataset or using augmented dataset",)
    parser.add_argument("--actor_learning_rate",  type=float, default=0.0003,
                        help="Actor learning rate.",)
    parser.add_argument("--critic_learning_rate",  type=float, default=0.0003,
                        help="Critic learning rate.",)
    parser.add_argument("--n_critics", type=int, default=2,
                        help='the number of Q functions for ensemble')
    # parser.add_argument("--alpha",  type=float, default=2.5, help="Alpha value for calculating actor loss!",)
    parser.add_argument("--expectile",  type=float, default=0.7, help="the expectile value for value function training",)

    # parser.add_argument("--policy_path", type=str, help="The path of trained model",)
    # parser.add_argument("--visualization", "-v", action="store_true",
    #     help="Enable visualization of environment.",)
    # parser.add_argument("--n_episodes", type=int, default=64,
    #     help="Number of episodes to run. Default: %(default)s",)
    # parser.add_argument("--output", type=pathlib.Path, metavar="FILENAME",
    #     help="Save results to a JSON file.",)

    args = parser.parse_args()
    main(args)

