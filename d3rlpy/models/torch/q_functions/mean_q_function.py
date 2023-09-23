from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import escnn
from escnn.nn import EquivariantModule, FieldType
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..encoders import Encoder, EncoderWithAction
from .base import (
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    QFunctionOutput,
)
from .utility import compute_huber_loss, compute_reduce, pick_value_by_action

__all__ = [
    "DiscreteMeanQFunction",
    "EquivariantDiscreteMeanQFunction",
    "ContinuousMeanQFunction",
    "EquivariantContinuousMeanQFunction",
    "DiscreteMeanQFunctionForwarder",
    "ContinuousMeanQFunctionForwarder",
]


def quaternion2rot(quaternion: torch.Tensor) -> torch.Tensor:
    r = R.from_quat(quaternion)
    rot = torch.tensor(r.as_matrix())
    flatten_rot = torch.flatten(rot, start_dim=1, end_dim=-1)
    return flatten_rot


# Here we should design the process function in model forward call.
def process_trifinger_obs(batch_obs):
    transformed_ob_dim = 148  # to be confirmed
    batch_size = batch_obs.shape[0]
    transformed_obs = torch.zeros((batch_size, transformed_ob_dim), dtype=torch.float32)
    pos_ones = torch.ones((batch_size, 1))
    transformed_obs[:, :24] = batch_obs[:, :24]
    transformed_obs[:, 24:33] = batch_obs[:, 24:33]
    transformed_obs[:, 33: 57] = batch_obs[:, 33: 57]
    transformed_obs[:, 57: 58] = batch_obs[:, 57: 58]
    transformed_obs[:, 58: 59] = batch_obs[:, 58: 59]
    transformed_obs[:, 59: 83] = batch_obs[:, 59: 83]
    transformed_obs[:, 83: 92] = quaternion2rot(batch_obs[:, 83: 87])
    transformed_obs[:, 92: 96] = torch.cat([batch_obs[:, 87: 90], pos_ones], axis=-1)
    transformed_obs[:, 96: 99] = batch_obs[:, 90: 93]
    transformed_obs[:, 99: 103] = torch.cat([batch_obs[:, 93: 96], pos_ones], axis=-1)
    transformed_obs[:, 103: 107] = torch.cat([batch_obs[:, 96: 99], pos_ones], axis=-1)
    transformed_obs[:, 107: 111] = torch.cat([batch_obs[:, 99: 102], pos_ones], axis=-1)
    transformed_obs[:, 111: 120] = batch_obs[:, 102: 111]
    transformed_obs[:, 120: 129] = batch_obs[:, 111: 120]
    transformed_obs[:, 129:130] = batch_obs[:, 120: 121]
    transformed_obs[:, 130: 139] = batch_obs[:, 121: 130]
    transformed_obs[:, 139: 148] = batch_obs[:, 130: 139]

    return transformed_obs


# This func is used to extract the invariant features of the outputs of equivariant encoder
def compute_invariant_features(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
    n_inv_features = len(field_type.irreps)
    # TODO: Ensure isotypic basis i.e irreps of the same type are consecutive to each other.
    inv_features = []
    for field_start, field_end, rep in zip(field_type.fields_start, field_type.fields_end,
                                           field_type.representations):
        # Each field here represents a representation of an Isotypic Subspace. This rep is only composed of a single
        # irrep type.
        x_field = x[..., field_start:field_end]  # whether x is Tensor
        num_G_stable_spaces = len(rep.irreps)  # Number of G-invariant features = multiplicity of irrep
        # Again this assumes we are already in an Isotypic basis
        assert len(np.unique(rep.irreps, axis=0)) == 1, "This only works for now on the Isotypic Basis"
        # This basis is useful because we can apply the norm in a vectorized way
        # Reshape features to [batch, num_G_stable_spaces, num_features_per_G_stable_space]
        x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
        # Compute G-invariant measures as the norm of the features in each G-stable space
        inv_field_features = torch.norm(x_field_p, dim=-1)
        # Append to the list of inv features
        inv_features.append(inv_field_features)
    # Concatenate all the invariant features
    inv_features = torch.cat(inv_features, dim=-1)
    assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
    return inv_features


class DiscreteMeanQFunction(DiscreteQFunction):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder


############################################################################
## Equivariant Encoder + Invariant Extractor + Normal NN head
############################################################################
# class EquivariantDiscreteMeanQFunction(DiscreteQFunction):
#     # _encoder: Encoder
#     # _fc: nn.Linear
#
#     # def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
#     def __init__(self, encoder, hidden_size: int, action_size: int,
#                  # head_hidden_size=128,
#                  num_hidden_layers=2,
#                  ):
#         super().__init__()
#         self._encoder = encoder
#         self._head = nn.Sequential()
#         for i in range(num_hidden_layers):
#             self._head.add_module(f'normal_hidden{i}', nn.Linear(hidden_size, hidden_size))
#             self._head.add_module(f'normal_activation{i}', nn.ReLU())
#         self._out_fc = nn.Linear(hidden_size, action_size)
#
#     def forward(self, x: torch.Tensor) -> QFunctionOutput:
#         # cart position, cart velocity, pole angle, pole angle velocity
#         out = self._encoder(x)
#         # out_type = self._encoder.out_type
#         out_type = self._encoder.network.out_type  # out group repretation type of IIDbatch
#         out = compute_invariant_features(out, out_type)   # invariant features here
#         out = self._head(out)
#
#         return QFunctionOutput(
#             q_value=self._out_fc(out),
#             quantiles=None,
#             taus=None,
#         )
#
#     @property
#     # def encoder(self) -> Encoder:
#     def encoder(self):
#         return self._encoder


############################################################################
## Equivariant Encoder for Discrete Action Value
############################################################################
class EquivariantDiscreteMeanQFunction(DiscreteQFunction):
    # _encoder: Encoder
    # _fc: nn.Linear
    """
    Remark:
        1. Here we are plan to the newest equivariant version in models.py.
        2. To build the equivariant encoder:
            2.1 Create the CartpoleEnvEncoder in escnn_model.py with
                a) the in_type corresponding to the state space
                b) the out_type corresponding to flip the outputs of the 2 neurons in the last layer
            2.2 Call the CartpoleEnvEncoder in CartpoleEnvEncoderFactory
            2.3 Set the customized encoder factory = CartpoleEnvEncoderFactory()
        3. Simplify the forward pass of Q function by only using out = self._encoder(inputs)

    Notice: Here the output of encoder are not being extracted as invariant features later!!!
            But we may try to extract the invariant features later.
    """

    # def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
    def __init__(self, encoder, hidden_size: int, action_size: int,
                 # head_hidden_size=128,
                 num_hidden_layers=2,
                 ):
        super().__init__()
        self._encoder = encoder

    def forward(self, x: torch.Tensor) -> QFunctionOutput:
        # cart position, cart velocity, pole angle, pole angle velocity
        out = self._encoder(x)
        return QFunctionOutput(
            q_value=out,
            quantiles=None,
            taus=None,
        )

    @property
    # def encoder(self) -> Encoder:
    def encoder(self):
        return self._encoder


class DiscreteMeanQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DiscreteMeanQFunction
    _action_size: int

    def __init__(self, q_func: DiscreteMeanQFunction, action_size: int):
        self._q_func = q_func
        self._action_size = action_size

    def compute_expected_q(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_func(x).q_value

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        one_hot = F.one_hot(actions.view(-1), num_classes=self._action_size)
        value = (self._q_func(observations).q_value * one_hot.float()).sum(
            dim=1, keepdim=True
        )
        y = rewards + gamma * target * (1 - terminals)
        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is None:
            return self._q_func(x).q_value
        return pick_value_by_action(
            self._q_func(x).q_value, action, keepdim=True
        )


class ContinuousMeanQFunction(ContinuousQFunction):
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, hidden_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x, action)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class ContinuousMeanQFunctionForwarder(ContinuousQFunctionForwarder):
    # _q_func: ContinuousMeanQFunction

    # def __init__(self, q_func: ContinuousMeanQFunction):
    def __init__(self, q_func):
        self._q_func = q_func

    def compute_expected_q(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func(x, action).q_value

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        value = self._q_func(observations, actions).q_value
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func(x, action).q_value


class EquivariantContinuousMeanQFunction(ContinuousQFunction):
    # _encoder: EncoderWithAction
    _out_fc: nn.Linear

    def __init__(self,
                 # encoder: EncoderWithAction,
                 encoder,
                 hidden_size: int,
                 num_hidden_size=2,
                 ):

        super().__init__()
        self._encoder = encoder
        self._head = nn.Sequential()
        self.activation = nn.ReLU()
        for i in range(num_hidden_size):
            self._head.add_module(f'head_hidden_layer_{i}', nn.Linear(hidden_size, hidden_size))
            self._head.add_module(f'head_hidden_activation_{i}', self.activation)

        self._out_fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> QFunctionOutput:
        """

        Args:
            x:  applying preprocess on states
            action:

        Returns:

        """
        x = process_trifinger_obs(x)
        out = torch.cat([x, action], dim=-1)
        out = self._encoder(out)
        # out_type = self._encoder.out_type
        out_type = self._encoder[-1].out_type
        inv_features = compute_invariant_features(out, out_type)
        out = self._head(inv_features)

        return QFunctionOutput(
            q_value=self._out_fc(out),
            quantiles=None,
            taus=None,
        )

    @property
    # def encoder(self) -> EncoderWithAction:
    def encoder(self):
        return self._encoder

