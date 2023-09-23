from typing import List, Tuple, Union
import d3rlpy.models.encoders
import torch
from escnn.nn import EquivariantModule, FieldType
import torch.cuda
import logging
import escnn.group
from models import EMLP
from build_reps import *

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################
# Setting the structure params of Trifinger for EMLP model
########################################################################
units = [256, 256]
trifinger_gspace = escnn.gspaces.no_base_space(Trifinger_G)
trifinger_activation = escnn.nn.ReLU
trifinger_n_hidden_neurons = 256
trifinger_num_regular_field = int(np.ceil(trifinger_n_hidden_neurons / Trifinger_G.order()))
# Compute the observation space Isotypic Rep from the regular representation
# Define the observation space in the ISOTYPIC BASIS!
trifinger_rep_features_iso_basis = group_utils.isotypic_basis(Trifinger_G, trifinger_num_regular_field, prefix='ObsSpace')
trifinger_inv_encoder_out_type = FieldType(trifinger_gspace, [rep_iso for rep_iso in trifinger_rep_features_iso_basis.values()])

##################################################################
# Argments updated for the newest EMLP impl
#################################################################

emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': policy_in_type,
    'out_type': value_out_type,
}

categorical_emlp_args = {
    'in_type': policy_in_type,
    'out_type': categorical_prob_type,
    'activation': "ELU",
}

cartpole_critic_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': value_in_type,
    'out_type': value_out_type,
}

inv_encoder_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': policy_in_type,
    'out_type': inv_encoder_out_type,
}

trifinger_actor_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': Trifinger_policy_in_type,
    'out_type': Trifinger_policy_out_type,
}

trifinger_trivial_actor_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': Trifinger_policy_in_type,
    'out_type': Trifinger_trivial_policy_out_type,
}


# Set the out_type with representations of isotypic basis
trifinger_critic_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': Trifinger_value_in_type,
    'out_type': trifinger_inv_encoder_out_type,
}

# Set the out_type with representations of isotypic basis
trifinger_value_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': Trifinger_value_in_type,
    'out_type': trifinger_inv_encoder_out_type,
}


class MainEncoder(torch.nn.Module):
    def __init__(self, feature_size, emlp_args):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**emlp_args)

    def forward(self, inputs):
        outs = self.network(inputs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class MainEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'RandomGroup'

    def __init__(self, emlp_args, feature_size=256):
        self.feature_size = feature_size
        self.emlp_args = emlp_args

    def create(self, observation_shape):
        return MainEncoder(self.feature_size, self.emlp_args)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'C2Group'


class C2Encoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLPNoLast(**emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class CartpoleInvEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**inv_encoder_emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs).tensor
        return outs

    def get_feature_size(self):
        return self.feature_size


#############################################################################
### TODO: Using the equivariant Q function approximator to test the CQL
#############################################################################
class CartpoleEnvEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**categorical_emlp_args)

    def forward(self, inputs):
        outs = self.network(inputs).tensor
        return outs

    def get_feature_size(self):
        return self.feature_size


class C2CategoricalEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLPCategorical(**categorical_emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class C2ActorEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**categorical_emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class C2CriticEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**categorical_emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class CartpoleInvEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleInv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return CartpoleInvEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleInv'


class CartpoleEnvEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleEnv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return CartpoleEnvEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleEnv'


class C2EncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'C2Group'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return C2Encoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'C2Group'


class C2CategoricalEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'C2GroupCategoricalProb'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return C2CategoricalEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'C2Group'


class C2ActorEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'C2ActorEncoder'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return C2ActorEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'C2Group'


class C2CriticEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'C2CriticEncoder'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return C2CriticEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'C2Group'


class EMLPNoLast(EquivariantModule):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""

    def __init__(self,
                 units: int,
                 activation: Union[EquivariantModule, List[EquivariantModule]],
                 in_type: FieldType,
                 out_type: FieldType,
                 with_bias=True,
                 ):

        super(EMLPNoLast, self).__init__()

        logging.info("Initing EMLP (PyTorch)")
        self.num_layers = len(units) + 1
        self.in_type, self.out_type = in_type, out_type
        self.gspace = self.in_type.gspace
        self.activations = activation if isinstance(activation, list) else [activation] * (self.num_layers - 1)

        n_hidden_layers = self.num_layers - 2
        if n_hidden_layers == 0:
            log.warning(f"{self} model initialized with 0 hidden layers")

        regular_rep = self.gspace.fibergroup.regular_representation
        layer_in_type = in_type

        self.net = escnn.nn.SequentialModule()

        for n in range(self.num_layers - 1):
            layer_out_type = FieldType(self.gspace, [regular_rep] * \
                                       int(np.ceil(units[n] / self.gspace.fibergroup.order())))
            activation = self.activations[n](layer_out_type)

            self.net.add_module(f"linear_{n}", escnn.nn.Linear(layer_in_type, layer_out_type, bias=with_bias))
            self.net.add_module(f"act_{n}", activation)
            self.net.add_module(f"batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type)),
            layer_in_type = layer_out_type

        # Add final layer
        # head_layer = escnn.nn.Linear(layer_in_type, out_type, bias=with_bias)
        # self.net.add_module("head", head_layer)
        # Test the entire model is equivariant.
        self.net.check_equivariance()

        for m in self.net.modules():
            if isinstance(m, escnn.nn.Linear):
                if getattr(m, "bias", None) is not None:
                    with torch.no_grad():
                        m.bias.zero_()
                        print(m.bias)

    def forward(self, x):
        """Forward pass of the EMLP model."""
        x = self.net.in_type(x)
        return self.net(x).tensor

    def get_hparams(self):
        return {'num_layers':    len(self.net),
                'hidden_ch':     self.num_hidden_regular_fields,
                'activation':    str(self.activations.__class__.__name__),
                'Repin':         str(self.rep_in),
                'Repout':        str(self.rep_in),
                'init_mode':     str(self.init_mode),
                'inv_dim_scale': self.inv_dims_scale,
                }

    def reset_parameters(self, init_mode=None):
        """Initialize weights and biases of E-MLP model."""
        raise NotImplementedError()

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return (batch_size, self.out_type.size)


class EMLPCategorical(EquivariantModule):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""
    def __init__(self,
                 units: int,
                 activation: Union[EquivariantModule, List[EquivariantModule]],
                 in_type: FieldType,
                 out_type: FieldType,
                 with_bias=True,
                 ):

        super(EMLPCategorical, self).__init__()

        logging.info("Initing EMLP (PyTorch)")
        self.num_layers = len(units) + 1
        self.in_type, self.out_type = in_type, out_type
        self.gspace = self.in_type.gspace
        self.activations = activation if isinstance(activation, list) else [activation] * (self.num_layers - 1)

        n_hidden_layers = self.num_layers - 2
        if n_hidden_layers == 0:
            log.warning(f"{self} model initialized with 0 hidden layers")

        regular_rep = self.gspace.fibergroup.regular_representation
        layer_in_type = in_type

        self.net = escnn.nn.SequentialModule()

        for n in range(self.num_layers - 1):
            layer_out_type = FieldType(self.gspace, [regular_rep] * \
                                       int(np.ceil(units[n] / self.gspace.fibergroup.order())))
            activation = self.activations[n](layer_out_type)

            self.net.add_module(f"linear_{n}", escnn.nn.Linear(layer_in_type, layer_out_type, bias=with_bias))
            self.net.add_module(f"act_{n}", activation)
            # self.net.add_module(f"batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type)),
            layer_in_type = layer_out_type

        # Add final layer
        head_layer = escnn.nn.Linear(layer_in_type, out_type, bias=with_bias)
        self.net.add_module("head", head_layer)
        # Test the entire model is equivariant.
        self.net.check_equivariance()

        for m in self.net.modules():
            if isinstance(m, escnn.nn.Linear):
                if getattr(m, "bias", None) is not None:
                    with torch.no_grad():
                        m.bias.zero_()
                        print(m.bias)

    def forward(self, x):
        """Forward pass of the EMLP model."""
        x = self.net.in_type(x)
        return self.net(x).tensor

    def get_hparams(self):
        return {'num_layers':    len(self.net),
                'hidden_ch':     self.num_hidden_regular_fields,
                'activation':    str(self.activations.__class__.__name__),
                'Repin':         str(self.rep_in),
                'Repout':        str(self.rep_in),
                'init_mode':     str(self.init_mode),
                'inv_dim_scale': self.inv_dims_scale,
                }

    def reset_parameters(self, init_mode=None):
        """Initialize weights and biases of E-MLP model."""
        raise NotImplementedError()

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return (batch_size, self.out_type.size)


if __name__ == '__main__':
    cartpole_inv_encoder = EMLP(**inv_encoder_emlp_args)
    print(111)

