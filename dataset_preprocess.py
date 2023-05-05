import gym
import argparse, os
import gc
import numpy as np
import h5py
from d3rlpy.dataset import MDPDataset
import torch
import models
from utils import Normlizor
import rrc_2022_datasets


def load_h5_file(args):
    if args.task == 'sim_lift_exp':
        task_name = "trifinger-cube-lift-sim-expert-v0"
        print("Training the simulation lifting task with expert dataset")
    elif args.task == 'sim_push_exp':
        task_name = "trifinger-cube-push-sim-expert-v0"
        print("Training the simulation pushing task with expert dataset")
    else:
        raise RuntimeError(
            'The task name you input is invalid, only push and lift are avaliable')

    env = gym.make(
        task_name,
        # "trifinger-cube-lift-sim-expert-v0",
        disable_env_checker=True,
    )
    gym_dataset = env.get_dataset()

    raw_dataset = {}
    raw_dataset['observations'] = gym_dataset['observations']
    raw_dataset['rewards'] = gym_dataset['rewards']
    raw_dataset['timeouts'] = gym_dataset['timeouts']
    raw_dataset['actions'] = gym_dataset['actions']
    save_path = './save/' + task_name + '.npy'
    np.save(save_path, raw_dataset)
    del gym_dataset
    gc.collect()
    norm = Normlizor()

    dataset = np.load(save_path,allow_pickle=True).item()
    task_name = os.path.split(save_path)[1].split('.')[0]
    norm.init_norm_with_dataset(dataset, 'std', name='train_norm_params', save_path='./save'+'/'+task_name)
    dataset = MDPDataset(dataset["observations"],
                         dataset["actions"],
                         dataset["rewards"],
                         dataset["timeouts"].astype(np.int32)
                         )
    dataset.dump(os.path.join(os.path.split(save_path)[0], task_name, 'train_dataset.h5'))
    del dataset
    gc.collect()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sim_push_exp', help='The name of the task')
    # parser.add_argument('--raw-dataset-path', default=None, help='If load the exist dataset')
    # parser.add_argument('--save-path', default=None, help='Where to save the models and data')
    args = parser.parse_args()
    load_h5_file(args)




