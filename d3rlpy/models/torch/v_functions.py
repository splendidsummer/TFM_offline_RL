from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import Encoder
from scipy.spatial.transform import Rotation as R


__all__ = ["ValueFunction", "compute_v_function_error"]


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


class ValueFunction(nn.Module):  # type: ignore
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return cast(torch.Tensor, self._fc(h))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


class EquivariantValueFunction(nn.Module):  # type: ignore
    # _encoder: Encoder
    _fc: nn.Linear

    # def __init__(self, encoder: Encoder, hidden_size: int):
    def __init__(self,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = process_trifinger_obs(x)
        out = self._encoder(x)
        # out_type = self._encoder.out_type
        out_type = self._encoder[-1].out_type
        inv_features = compute_invariant_features(out, out_type)
        out = self._head(inv_features)
        return cast(torch.Tensor, self._out_fc(out))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


def compute_v_function_error(
    v_function: ValueFunction, observations: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    v_t = v_function(observations)
    loss = F.mse_loss(v_t, target)
    return loss
