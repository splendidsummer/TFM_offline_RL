from typing import cast
import torch
import torch.nn.functional as F
from torch import nn
from .q_functions.mean_q_function import compute_invariant_features,\
    process_trifinger_obs
from .encoders import Encoder
from scipy.spatial.transform import Rotation as R


__all__ = [
    "ValueFunction",
    "EquivariantValueFunction",
    "compute_v_function_error",
]


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