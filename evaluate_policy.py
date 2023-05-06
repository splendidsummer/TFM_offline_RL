import numpy as np
import torch
import policies
import argparse
import json
import logging
import pathlib
import typing
import gym
from rrc_2022_datasets import Evaluation, PolicyBase, TriFingerDatasetEnv


class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        torch_model_path,
    ):
        self.device = "cpu"

        # load torch script
        self.policy = torch.jit.load(
            torch_model_path, map_location=torch.device(self.device)
        )

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = self.policy(observation.unsqueeze(0))
        action = action.detach().numpy()[0]
        action = np.clip(action, -0.397, 0.397)
        return action


class TorchPushPolicy(TorchBasePolicy):
    """
    Policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, push_path):
        model_path = policies.get_model_path(push_path)
        super().__init__(model_path)


class TorchLiftPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.

    Expects flattened observations.
    """

    def __init__(self, lift_path):
        model_path = policies.get_model_path(lift_path)
        super().__init__(model_path)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, choices=["push", "lift"],
        help="Which task to evaluate ('push' or 'lift').", )
    parser.add_argument("--algorithm", type=str, choices=["bc", "td3+bc", "iql", "cql"],
        help="Which algorithm to evaluate ('push' or 'lift').", )
    # parser.add_argument("--policy_path", type=str, help="The path of trained model",)
    parser.add_argument("--visualization", "-v", action="store_true",
        help="Enable visualization of environment.",)
    parser.add_argument("--n_episodes", type=int, default=64,
        help="Number of episodes to run. Default: %(default)s",)
    parser.add_argument("--output", type=pathlib.Path, metavar="FILENAME",
        help="Save results to a JSON file.",)
    args = parser.parse_args()
    policy_path = args.task + '_' + args.algorithm + '_policy.pt'

    if args.task == "push":
        env_name = "trifinger-cube-push-sim-expert-v0"
        policy = TorchPushPolicy(policy_path)

    elif args.task == "lift":
        env_name = "trifinger-cube-lift-sim-expert-v0"
        policy = TorchPushPolicy(policy_path)
    else:
        print("Invalid task %s" % args.task)

    flatten_observations = policy.is_using_flattened_observations()
    if flatten_observations:
        print("Using flattened observations")
    else:
        print("Using structured observations")

    env = typing.cast(
        TriFingerDatasetEnv,
        gym.make(
            env_name,
            disable_env_checker=True,
            visualization=args.visualization,
            flatten_obs=flatten_observations,
        ),
    )

    evaluation = Evaluation(env)
    eval_res = evaluation.evaluate(policy=policy, n_episodes=args.n_episodes,)
    json_result = json.dumps(eval_res, indent=4)

    print("Evaluation result: ")
    print(json_result)

    if args.output:
        args.output.write_text(json_result)

