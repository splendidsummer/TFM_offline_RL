from abc import ABC

import d3rlpy
import gym
from d3rlpy.dataset import MDPDataset
import rrc_2022_datasets
from sklearn.model_selection import train_test_split
from d3rlpy.algos import SAC, BC
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.preprocessing import action_scalers

env = gym.make(
    # "trifinger-cube-push-sim-expert-v0",
    "trifinger-cube-lift-sim-expert-v0",
    disable_env_checker=True,
    # flatten_obs=False,
    flatten_obs=True,
    # obs_to_keep=obs_to_keep,
    # visualization=True,  # enable visualization
)

dataset = env.get_dataset()
obs = dataset['observations']
actions = dataset['actions']
rewards = dataset['rewards']
timeouts = dataset['timeouts']

##################################################################
# Preprocess data
# 1. Delete features not relevant:
#               1.1 robot_id
#               1.2 ???
# 2.Normalize is not appropriate for this case!!
# 3. Transform key points to position&orientation
##################################################################
dataset = MDPDataset(obs, actions, rewards, timeouts)
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# if you don't use GPU, set use_gpu=False instead.
bc = BC(
    use_gpu=False,
    scaler='standard',
    action_scaler='min_max',
)

# initialize neural networks with the given observation shape and action size.
# this is not necessary when you directly call fit or fit_online method.
bc.build_with_dataset(dataset)
# set environment in scorer function
evaluate_scorer = evaluate_on_environment(env)
# evaluate algorithm on the environment
# rewards = evaluate_scorer(sac)


if __name__ == '__main__':

    bc.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_epochs=10,
        scorers={
            'enviroment': evaluate_scorer
        })

    print('finishing fitting!!')

    bc.save_model('bc.pt')
    bc.save_policy('bc_policy.pt')

