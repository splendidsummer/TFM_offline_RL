"""
Notice:  1. Header should include 2-3 layers to get better performance
"""

from typing import List, Tuple, Union
import d3rlpy.models.encoders
import torch
from escnn.nn import EquivariantModule, FieldType
import torch.cuda
import logging
import escnn.group
from escnn.group import CyclicGroup
from util_funcs import group_utils
from escnn.nn import init
from build_reps import *

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = {
    'perm': [[0, 1]],
    'reflex': [[-1, -1]],
    'perm_TqQ': [[0, 1]],
    'reflex_TqQ': [[-1, -1]],
    'value_perm': [[0, 1, 2, 3, 4]],
    'value_reflex': [[-1, -1, -1, -1, -1]],  # 0,1 process of the action to 2*action - 1
    'policy_perm': [[0, 1]],
    'policy_reflex': [[-1, 1]],
    'categorical_policy_perm': [[1, 0]],
    'categorical_policy_reflex': [[1, 1]],
}

G = CyclicGroup._generator(2)  # Cyclicgroup2
rep_field = float
dimQ_js, dimTqQ_js = 2, 2
rep_Q_js = {G.identity: np.eye(dimQ_js, dtype=rep_field)}
rep_po = {G.identity: np.eye(dimQ_js, dtype=rep_field)}
rep_va = {G.identity: np.eye(len(cfg['value_perm'][0]), dtype=rep_field)}
rep_out_prob = {G.identity: np.eye(2, dtype=rep_field)}
rep_categorical = {G.identity: np.eye(2, dtype=rep_field)}

# Generate ESCNN representation of generators
for g_gen, perm, refx in zip(G.generators, cfg['perm'], cfg['reflex']):
    assert len(perm) == dimQ_js == len(refx), \
        f"Dimension of joint-space position coordinates dim(Q_js)={dimQ_js} != dim(rep_Q_JS): {len(refx)}"
    refx = np.array(refx, dtype=rep_field)
    rep_Q_js[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

# Generate ESCNN representation of generators
for g_gen, perm, refx in zip(G.generators, cfg['policy_perm'], cfg['policy_reflex']):
    refx = np.array(refx, dtype=rep_field)
    rep_po[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

for g_gen, perm, refx in zip(G.generators, cfg['value_perm'], cfg['value_reflex']):
    refx = np.array(refx, dtype=rep_field)
    rep_va[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

for g_gen, perm, refx in zip(G.generators, cfg['categorical_policy_perm'], cfg['categorical_policy_reflex']):
    refx = np.array(refx, dtype=rep_field)
    rep_categorical[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)


# Generate the entire group
rep_Q_js = group_rep_from_gens(G, rep_Q_js)
rep_po = group_rep_from_gens(G, rep_po)
rep_va = group_rep_from_gens(G, rep_va)
rep_categorical = group_rep_from_gens(G, rep_categorical)
rep_Q_js.name = 'Q_js'
rep_TqQ_js = rep_Q_js
rep_TqQ_js.name = 'TqQ_js'
rep_po.name = 'Policy'
rep_va.name = 'Value'
rep_categorical.name = 'Categorical'


# Add representations to the group.
G.representations.update(Q_js=rep_Q_js,
                         TqQ_js=rep_TqQ_js,
                         Policy=rep_po,
                         Value=rep_va,
                         Categorical=rep_categorical,
                         )

rep_QJ = G.representations['Q_js']
rep_TqJ = G.representations['TqQ_js']
rep_po = G.representations['Policy']
rep_va = G.representations['Value']
rep_categorical = G.representations['Categorical']

gspace = escnn.gspaces.no_base_space(G)
value_in_type = FieldType(gspace, [rep_va])
policy_in_type = FieldType(gspace, [rep_QJ, rep_TqJ])
value_out_type = FieldType(gspace, [gspace.trivial_repr])
policy_out_type = FieldType(gspace, [rep_po])
categorical_prob_type = FieldType(gspace, [rep_categorical])

########################################################################
# Setting the structure params of Cartpole for EMLP model
########################################################################
units = [256, 256]
activation = escnn.nn.ReLU
n_hidden_neurons = 256
num_regular_field = int(n_hidden_neurons / G.order())
# Compute the observation space Isotypic Rep from the regular representation
# Define the observation space in the ISOTYPIC BASIS!

rep_features_iso_basis = group_utils.isotypic_basis(G, num_regular_field, prefix='ObsSpace')
inv_encoder_out_type = FieldType(gspace, [rep_iso for rep_iso in rep_features_iso_basis.values()])

########################################################################
# Setting the structure params of Trifinger for EMLP model
########################################################################
trifinger_units = [256, 256]
trifinger_gspace = escnn.gspaces.no_base_space(Trifinger_G)
trifinger_activation = escnn.nn.ReLU
trifinger_n_hidden_neurons = 256
trifinger_num_regular_field = int(np.ceil(trifinger_n_hidden_neurons / Trifinger_G.order()))
# Compute the observation space Isotypic Rep from the regular representation
# Define the observation space in the ISOTYPIC BASIS!
trifinger_rep_features_iso_basis = group_utils.isotypic_basis(Trifinger_G, trifinger_num_regular_field, prefix='ObsSpace')
trifinger_inv_encoder_out_type = FieldType(trifinger_gspace, [rep_iso for rep_iso in trifinger_rep_features_iso_basis.values()])

emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': policy_in_type,
    'out_type': value_out_type,
}

categorical_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': policy_in_type,
    'out_type': categorical_prob_type,
}

cartpole_actor_emlp_args = {
    'units': units,
    'activation': escnn.nn.ReLU,
    'in_type': policy_in_type,
    'out_type': categorical_prob_type,
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


class EMLP(torch.nn.Module):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""

    def __init__(self,
                 units: List[int],
                 activation: Union[EquivariantModule, List[EquivariantModule]],
                 in_type: FieldType,
                 out_type: FieldType,
                 with_bias=True,
                 ):

        super(EMLP, self).__init__()

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

        self.net = torch.nn.SequentialModule()

        for n in range(self.num_layers - 1):
            layer_out_type = FieldType(self.gspace, [regular_rep] * \
                                       int(np.ceil(units[n] / self.gspace.fibergroup.order())))
            activation = self.activations[n](layer_out_type)

            self.net.add_module(f"equivariant_linear_{n}", escnn.nn.Linear(layer_in_type, layer_out_type, bias=with_bias))
            self.net.add_module(f"equivariant_activation_{n}", activation)
            self.net.add_module(f"equivariant_batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type)),
            layer_in_type = layer_out_type

        # Add final layer
        head_layer = escnn.nn.Linear(layer_in_type, out_type, bias=with_bias)
        self.net.add_module("equivariant_out_linear", head_layer)
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
        return self.net(x)

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


# TODO: MODIFY HERE
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


# TODO: MODIFY HERE
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


# TODO: MODIFY HERE
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


class EMLP(EquivariantModule):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""

    def __init__(self,
                 units: List[int],
                 activation: Union[EquivariantModule, List[EquivariantModule]],
                 in_type: FieldType,
                 out_type: FieldType,
                 with_bias=True,
                 ):

        super(EMLP, self).__init__()

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
        head_layer = escnn.nn.Linear(layer_in_type, out_type, bias=with_bias)
        self.net.add_module("head", head_layer)
        # self.reset_parameters()
        # Test the entire model is equivariant
        self.net.check_equivariance()

    def forward(self, x):
        """Forward pass of the EMLP model."""
        x = self.net.in_type(x)
        return self.net(x)

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
        for m in self.net.modules():
            if isinstance(m, escnn.nn.Linear):
                init.generalized_he_init(m.weights)
                with torch.no_grad():
                    m.bias.zero_()

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return (batch_size, self.out_type.size)


class MLP(torch.nn.Module):
    """Standard baseline MLP. Representations and group are used for shapes only."""

    def __init__(self, d_in, d_out, num_hidden_units=128, num_layers=3,
                 activation: Union[torch.nn.Module, List[torch.nn.Module]] = torch.nn.ReLU,
                 with_bias=True, init_mode="fan_in"):
        """Constructor of a Multi-Layer Perceptron (MLP) model.

        This utility class allows to easily instanciate a G-equivariant MLP architecture.

        Args:
            d_in: Dimension of the input space.
            d_out: Dimension of the output space.
            num_hidden_units: Number of hidden units in the intermediate layers.
            num_layers: Number of layers in the MLP including input and output/head layers. That is, the number of
            activation (escnn.nn.EquivariantModule, list(escnn.nn.EquivariantModule)): If a single activation module is
            provided it will be used for all layers except the output layer. If a list of activation modules is provided
            then `num_layers` activation equivariant modules should be provided.
            with_bias: Whether to include a bias term in the linear layers.
            init_mode: Not used until now. Will be used to initialize the weights of the MLP
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.init_mode = init_mode
        self.hidden_channels = num_hidden_units
        self.activation = activation

        logging.info("Initializing MLP")

        dim_in = self.d_in
        dim_out = num_hidden_units
        self.net = torch.nn.Sequential()
        for n in range(num_layers - 1):
            dim_out = num_hidden_units
            block = torch.nn.Sequential()
            block.add_module(f"linear_{n}", torch.nn.Linear(dim_in, dim_out, bias=with_bias))
            block.add_module(f"batchnorm_{n}", torch.nn.BatchNorm1d(dim_out))
            block.add_module(f"act_{n}", activation())

            self.net.add_module(f"block_{n}", block)
            dim_in = dim_out
        # Add last layer
        linear_out = torch.nn.Linear(in_features=dim_out, out_features=self.d_out, bias=with_bias)
        self.net.add_module("head", linear_out)

        self.reset_parameters(init_mode=self.init_mode)

    def forward(self, input):
        output = self.net(input)
        return output

    def get_hparams(self):
        return {'num_layers': len(self.net),
                'hidden_ch':  self.hidden_channels,
                'init_mode':  self.init_mode}

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, torch.nn.Sequential):
                tensor = module[0].weight
                activation = module[-1].__class__.__name__
            elif isinstance(module, torch.nn.Linear):
                tensor = module.weight
                activation = "Linear"
            else:
                raise NotImplementedError(module.__class__.__name__)

            if "fan_in" == self.init_mode or "fan_out" == self.init_mode:
                torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
            elif 'normal' in self.init_mode.lower():
                split = self.init_mode.split('l')
                std = 0.1 if len(split) == 1 else float(split[1])
                torch.nn.init.normal_(tensor, 0, std)
            else:
                raise NotImplementedError(self.init_mode)

        log.info(f"MLP initialized with mode: {self.init_mode}")


if __name__ == '__main__':
    cartpole_inv_encoder = EMLP(**inv_encoder_emlp_args)
    print(111)
