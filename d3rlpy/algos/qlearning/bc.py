import dataclasses
from typing import Dict, Generic, TypeVar

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_categorical_policy,
    create_equivariant_categorical_policy,
    create_deterministic_policy,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.bc_impl import (
    BCBaseImpl,
    BCImpl,
    BCModules,
    DiscreteBCImpl,
    DiscreteBCModules,
)

__all__ = ["BCConfig", "BC", "DiscreteBCConfig", "DiscreteBC"]


TBCConfig = TypeVar("TBCConfig", bound="LearnableConfig")


class _BCBase(Generic[TBCConfig], QLearningAlgoBase[BCBaseImpl, TBCConfig]):
    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update_imitator(batch)
        return {"loss": loss}


@dataclasses.dataclass()
class BCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        policy_type (str): the policy type. Available options are
            ``['deterministic', 'stochastic']``.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
    """
    batch_size: int = 100
    learning_rate: float = 1e-3
    policy_type: str = "deterministic"
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()

    def create(self, device: DeviceArg = False) -> "BC":
        return BC(self, device)

    @staticmethod
    def get_type() -> str:
        return "bc"


class BC(_BCBase[BCConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        if self._config.policy_type == "deterministic":
            imitator = create_deterministic_policy(
                observation_shape,
                action_size,
                self._config.encoder_factory,
                device=self._device,
            )
        elif self._config.policy_type == "stochastic":
            imitator = create_normal_policy(
                observation_shape,
                action_size,
                self._config.encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
                device=self._device,
            )
        else:
            raise ValueError(f"invalid policy_type: {self._config.policy_type}")

        optim = self._config.optim_factory.create(
            imitator.parameters(), lr=self._config.learning_rate
        )

        modules = BCModules(optim=optim, imitator=imitator)

        self._impl = BCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteBCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log \pi_\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        beta (float): Reguralization factor.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
    """
    batch_size: int = 100
    learning_rate: float = 1e-3
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    beta: float = 0.5

    def create(self, device: DeviceArg = False) -> "DiscreteBC":
        return DiscreteBC(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_bc"


class DiscreteBC(_BCBase[DiscreteBCConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        # imitator = create_categorical_policy(
        #     observation_shape,
        #     action_size,
        #     self._config.encoder_factory,
        #     device=self._device,
        # )

        imitator = create_equivariant_categorical_policy(
            observation_shape,
            # action_size,
            self._config.encoder_factory,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            imitator.parameters(), lr=self._config.learning_rate
        )

        modules = DiscreteBCModules(optim=optim, imitator=imitator)

        self._impl = DiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            beta=self._config.beta,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(BCConfig)
register_learnable(DiscreteBCConfig)