from typing import Sequence, Tuple, cast
import escnn
import torch
from torch import nn
from ..dataset import Shape
from .encoders import EncoderFactory
from .q_functions import QFunctionFactory
from .torch import (
    CategoricalPolicy,
    EquivariantCategoricalPolicy,
    ConditionalVAE,
    ContinuousDecisionTransformer,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    DiscreteDecisionTransformer,
    DiscreteEnsembleQFunctionForwarder,
    GlobalPositionEncoding,
    NormalPolicy,
    EquivariantNormalPolicy,
    Parameter,
    SimplePositionEncoding,
    VAEDecoder,
    VAEEncoder,
    ValueFunction,
    EquivariantValueFunction,
    compute_output_size,
)
from .utility import create_activation

__all__ = [
    "create_discrete_q_function",
    "create_continuous_q_function",
    "create_equivariant_continuous_q_function",
    "create_deterministic_policy",
    "create_deterministic_residual_policy",
    "create_categorical_policy",
    "create_equivariant_categorical_policy",
    "create_normal_policy",
    "create_equivariant_normal_policy",
    "create_conditional_vae",
    "create_value_function",
    "create_equivariant_value_function",
    "create_parameter",
    "create_continuous_decision_transformer",
    "create_discrete_decision_transformer",
]


def create_discrete_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    n_ensembles: int = 1,
) -> Tuple[nn.ModuleList, DiscreteEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        hidden_size = compute_output_size([observation_shape], encoder)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
            hidden_size = compute_output_size([observation_shape], encoder)
        q_func, forwarder = q_func_factory.create_discrete(
            encoder, hidden_size, action_size
        )
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    q_func_modules.to(device)
    ensemble_forwarder = DiscreteEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


def create_equivariant_discrete_q_function(
    observation_shape,
    action_size,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    n_ensembles: int = 1,
) :  # Original Output type: -> Tuple[nn.ModuleList, DiscreteEnsembleQFunctionForwarder]
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():  # Why using cast here??
            p.register_hook(lambda grad: grad / n_ensembles)
        # Is this compatible with  EquivariantModule?
        for p in cast(escnn.nn.EquivariantModule, encoder).parameters():  # Why using cast here??
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
            # hidden_size = compute_output_size(
            #     [observation_shape], encoder, device
            # )
            hidden_size = 256

        q_func, forwarder = q_func_factory.create_equivaraint_discrete(
            encoder, hidden_size, action_size
        )
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    q_func_modules.to(device)
    ensemble_forwarder = DiscreteEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


def create_continuous_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    n_ensembles: int = 1,
) -> Tuple[nn.ModuleList, ContinuousEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        hidden_size = compute_output_size(
            [observation_shape, (action_size,)], encoder
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
            hidden_size = compute_output_size(
                [observation_shape, (action_size,)], encoder
            )
        q_func, forwarder = q_func_factory.create_continuous(
            encoder, hidden_size
        )
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    q_func_modules.to(device)
    ensemble_forwarder = ContinuousEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


# TODO: modify
def create_equivariant_continuous_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    n_ensembles: int = 1,
):   # -> Tuple[nn.ModuleList, ContinuousEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        # I assume here it does not need any modification
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        hidden_size = compute_output_size(
            [observation_shape, (action_size,)], encoder
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
            hidden_size = compute_output_size(
                [observation_shape, (action_size,)], encoder
            )
        q_func, forwarder = q_func_factory.create_equivariant_continuous(
            encoder, hidden_size
        )
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    q_func_modules.to(device)
    ensemble_forwarder = ContinuousEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


def create_deterministic_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
) -> DeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = DeterministicPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
    )
    policy.to(device)
    return policy


def create_deterministic_residual_policy(
    observation_shape: Shape,
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
    device: str,
) -> DeterministicResidualPolicy:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder
    )
    policy = DeterministicResidualPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
        scale=scale,
    )
    policy.to(device)
    return policy


def create_normal_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> NormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = NormalPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )
    policy.to(device)
    return policy


# TODO:  modify
def create_equivariant_normal_policy(
    observation_shape: Shape,
    action_size: int,
    # encoder_factory: EncoderFactory,
    encoder_factory,
    device: str,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> NormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)  # without using group BN layer
    policy = EquivariantNormalPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )
    policy.to(device)
    return policy


def create_categorical_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = CategoricalPolicy(
        encoder=encoder, hidden_size=hidden_size, action_size=action_size
    )
    policy.to(device)
    return policy


def create_equivariant_categorical_policy(
    observation_shape: Shape,
    # action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
) -> EquivariantCategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    # hidden_size = compute_output_size([observation_shape], encoder, device)
    policy = EquivariantCategoricalPolicy(encoder=encoder)
    policy.to(device)
    return policy


def create_conditional_vae(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> ConditionalVAE:
    encoder_encoder = encoder_factory.create_with_action(
        observation_shape, action_size
    )
    decoder_encoder = encoder_factory.create_with_action(
        observation_shape, latent_size
    )
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder_encoder
    )
    encoder = VAEEncoder(
        encoder=encoder_encoder,
        hidden_size=hidden_size,
        latent_size=latent_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
    )
    decoder = VAEDecoder(
        encoder=decoder_encoder,
        hidden_size=hidden_size,
        action_size=action_size,
    )
    policy = ConditionalVAE(encoder=encoder, decoder=decoder)
    policy.to(device)
    return policy


def create_value_function(
    observation_shape: Shape, encoder_factory: EncoderFactory, device: str
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    value_func = ValueFunction(encoder, hidden_size)
    value_func.to(device)
    return value_func


# TODO: modify
def create_equivariant_value_function(
    observation_shape: Shape, encoder_factory: EncoderFactory, device: str
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    value_func = EquivariantValueFunction(encoder, hidden_size)
    value_func.to(device)
    return value_func


def create_parameter(
    shape: Sequence[int], initial_value: float, device: str
) -> Parameter:
    data = torch.full(shape, initial_value, dtype=torch.float32)
    parameter = Parameter(data)
    parameter.to(device)
    return parameter


def create_continuous_decision_transformer(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    attn_dropout: float,
    resid_dropout: float,
    embed_dropout: float,
    activation_type: str,
    position_encoding_type: str,
    device: str,
) -> ContinuousDecisionTransformer:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)

    if position_encoding_type == "simple":
        position_encoding = SimplePositionEncoding(hidden_size, max_timestep)
    elif position_encoding_type == "global":
        position_encoding = GlobalPositionEncoding(
            hidden_size, max_timestep, context_size
        )
    else:
        raise ValueError(
            f"invalid position_encoding_type: {position_encoding_type}"
        )

    transformer = ContinuousDecisionTransformer(
        encoder=encoder,
        feature_size=hidden_size,
        position_encoding=position_encoding,
        action_size=action_size,
        num_heads=num_heads,
        context_size=context_size,
        num_layers=num_layers,
        attn_dropout=attn_dropout,
        resid_dropout=resid_dropout,
        embed_dropout=embed_dropout,
        activation=create_activation(activation_type),
    )
    transformer.to(device)
    return transformer


def create_discrete_decision_transformer(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    attn_dropout: float,
    resid_dropout: float,
    embed_dropout: float,
    activation_type: str,
    position_encoding_type: str,
    device: str,
) -> DiscreteDecisionTransformer:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)

    if position_encoding_type == "simple":
        position_encoding = SimplePositionEncoding(hidden_size, max_timestep)
    elif position_encoding_type == "global":
        position_encoding = GlobalPositionEncoding(
            hidden_size, max_timestep, context_size
        )
    else:
        raise ValueError(
            f"invalid position_encoding_type: {position_encoding_type}"
        )

    transformer = DiscreteDecisionTransformer(
        encoder=encoder,
        feature_size=hidden_size,
        position_encoding=position_encoding,
        action_size=action_size,
        num_heads=num_heads,
        context_size=context_size,
        num_layers=num_layers,
        attn_dropout=attn_dropout,
        resid_dropout=resid_dropout,
        embed_dropout=embed_dropout,
        activation=create_activation(activation_type),
    )
    transformer.to(device)
    return transformer