import dataclasses
from typing import Any, Dict, Sequence

import numpy as np
from typing_extensions import Protocol

from ..constants import ActionSpace
from .types import Observation, ObservationSequence
from .utils import (
    detect_action_space,
    get_dtype_from_observation,
    get_dtype_from_observation_sequence,
    get_shape_from_observation,
    get_shape_from_observation_sequence,
)

__all__ = [
    "Signature",
    "Transition",
    "PartialTrajectory",
    "EpisodeBase",
    "Episode",
    "DatasetInfo",
]


@dataclasses.dataclass(frozen=True)
class Signature:
    r"""Signature of arrays.

    Args:
        dtype: List of numpy data types.
        shape: List of array shapes.
    """
    dtype: Sequence[np.dtype]
    shape: Sequence[Sequence[int]]

    def sample(self) -> Sequence[np.ndarray]:
        r"""Returns sampled arrays.

        Returns:
            List of arrays based on dtypes and shapes.
        """
        return [
            np.random.random(shape).astype(dtype)
            for shape, dtype in zip(self.shape, self.dtype)
        ]


@dataclasses.dataclass(frozen=True)
class Transition:
    r"""Transition tuple.

    Args:
        observation: Observation.
        action: Action
        reward: Reward. This could be a multi-step discounted return.
        next_observation: Observation at next timestep. This could be
            observation at multi-step ahead.
        terminal: Flag of environment termination.
        interval: Timesteps between ``observation`` and ``next_observation``.
    """
    observation: Observation  # (...)
    action: np.ndarray  # (...)
    reward: np.ndarray  # (1,)
    next_observation: Observation  # (...)
    terminal: float
    interval: int

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation(self.observation)
        dtype = get_dtype_from_observation(self.observation)
        if isinstance(self.observation, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.action.dtype],
            shape=[self.action.shape],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.reward.dtype],
            shape=[self.reward.shape],
        )


@dataclasses.dataclass(frozen=True)
class PartialTrajectory:
    r"""Partial trajectory.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        returns_to_go: Sequence of remaining returns.
        terminals: Sequence of terminal flags.
        timesteps: Sequence of timesteps.
        masks: Sequence of masks that represent padding.
        length: Sequence length.
    """
    observations: ObservationSequence  # (L, ...)
    actions: np.ndarray  # (L, ...)
    rewards: np.ndarray  # (L, 1)
    returns_to_go: np.ndarray  # (L, 1)
    terminals: np.ndarray  # (L, 1)
    timesteps: np.ndarray  # (L,)
    masks: np.ndarray  # (L,)
    length: int

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation sigunature.

        Returns:
            Observation signature.
        """
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def __len__(self) -> int:
        return self.length


class EpisodeBase(Protocol):
    r"""Episode interface.

    ``Episode`` represens an entire episode.
    """

    @property
    def observations(self) -> ObservationSequence:
        r"""Returns sequence of observations.

        Returns:
            Sequence of observations.
        """
        raise NotImplementedError

    @property
    def actions(self) -> np.ndarray:
        r"""Returns sequence of actions.

        Returns:
            Sequence of actions.
        """
        raise NotImplementedError

    @property
    def rewards(self) -> np.ndarray:
        r"""Returns sequence of rewards.

        Returns:
            Sequence of rewards.
        """
        raise NotImplementedError

    @property
    def terminated(self) -> bool:
        r"""Returns environment terminal flag.

        This flag becomes true when this episode is terminated. For timeout,
        this flag stays false.

        Returns:
            Terminal flag.
        """
        raise NotImplementedError

    @property
    def observation_signature(self) -> Signature:
        r"""Returns observation signature.

        Returns:
            Observation signature.
        """
        raise NotImplementedError

    @property
    def action_signature(self) -> Signature:
        r"""Returns action signature.

        Returns:
            Action signature.
        """
        raise NotImplementedError

    @property
    def reward_signature(self) -> Signature:
        r"""Returns reward signature.

        Returns:
            Reward signature.
        """
        raise NotImplementedError

    def size(self) -> int:
        r"""Returns length of an episode.

        Returns:
            Episode length.
        """
        raise NotImplementedError

    def compute_return(self) -> float:
        r"""Computes total episode return.

        Returns:
            Total episode return.
        """
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        r"""Returns serized episode data.

        Returns:
            Serialized episode data.
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "EpisodeBase":
        r"""Constructs episode from serialized data.

        This is an inverse operation of ``serialize`` method.

        Args:
            serializedData: Serialized episode data.

        Returns:
            Episode object.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transition_count(self) -> int:
        r"""Returns the number of transitions.

        Returns:
            Number of transitions.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Episode:
    r"""Standard episode implementation.

    Args:
        observations: Sequence of observations.
        actions: Sequence of actions.
        rewards: Sequence of rewards.
        terminated: Flag of environment termination.
    """
    observations: ObservationSequence
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool

    @property
    def observation_signature(self) -> Signature:
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def size(self) -> int:
        return int(self.actions.shape[0])

    def compute_return(self) -> float:
        return float(np.sum(self.rewards))

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "Episode":
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
        )

    def __len__(self) -> int:
        return self.actions.shape[0]  # type: ignore

    @property
    def transition_count(self) -> int:
        return self.size() if self.terminated else self.size() - 1


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    r"""Dataset information.

    Args:
        observation_signature: Observation signature.
        action_signature: Action signature.
        reward_signature: Reward signature.
        action_space: Action space type.
        action_size: Size of action-space. For continuous action-space,
            this represents dimension of action vectors. For discrete
            action-space, this represents the number of discrete actions.
    """
    observation_signature: Signature
    action_signature: Signature
    reward_signature: Signature
    action_space: ActionSpace
    action_size: int

    @classmethod
    def from_episodes(cls, episodes: Sequence[EpisodeBase]) -> "DatasetInfo":
        r"""Constructs from sequence of episodes.

        Args:
            episodes: Sequence of episodes.

        Returns:
            DatasetInfo object.
        """
        action_space = detect_action_space(episodes[0].actions)
        if action_space == ActionSpace.CONTINUOUS:
            action_size = episodes[0].action_signature.shape[0][0]
        else:
            max_action = 0
            for episode in episodes:
                max_action = max(int(np.max(episode.actions)), max_action)
            action_size = max_action + 1  # index should start from 0
        return DatasetInfo(
            observation_signature=episodes[0].observation_signature,
            action_signature=episodes[0].action_signature,
            reward_signature=episodes[0].reward_signature,
            action_space=action_space,
            action_size=action_size,
        )
