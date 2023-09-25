import hydra.core.global_hydra
import escnn
import numpy as np
import escnn.group
from escnn.nn import FieldType
from hydra import compose, initialize
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.algebra_utils import gen_permutation_matrix
from morpho_symm.utils.group_utils import group_rep_from_gens


def load_trifinger_G(robot_cfg):
    """"
    Args:
        robot_cfg : group configuration load by hydra
        Remarks: The Trifinger state information:
                        1. Generalized Coordinates of Joints --> rep_QJ
                        2. Generalized Velocity of Joints --> rep_QJ
                        3. Cube Center Position --> rep_Ed
                        4. Cube Orientation (quaternion) --> Flatten to rotation matrix
                                                         --> rep_Od + rep_Od + rep_Od
                        5. Target Cube Center Position --> rep_Ed
                        6. Target Cube Orientation (quaternion) --> --> Flatten to rotation matrix
                                                         --> rep_Od + rep_Od + rep_Od
                        7. Previous/Current(????) Action --> rrep_TqJ

    Return:

    todo: modify more perm&reflex matrix in 3 dimensions, and 12 dimensions
          1. fingertip force
          2. fingertip position
          3. fingertip
    """
    _, G = load_symmetric_system(robot_cfg)
    rep_QJ = G.representations['Q_js']
    rep_TqJ = G.representations['TqQ_js']
    rep_Ed = G.representations['Ed']
    rep_Od = G.representations['Od']

    rep_field = float if robot_cfg.rep_fields.lower() != 'complex' else complex
    rep_Tips = {G.identity: np.eye(len(robot_cfg.permutation_Tips[0]), dtype=rep_field)}
    # Check a representation for each generator is provided
    assert len(robot_cfg.permutation_Tips) == len(robot_cfg.reflection_Tips) >= len(G.generators), \
        f"Not enough representation provided for the joint-space `Q_js`. " \
        f"Found {len(robot_cfg.permutation_Tips)} but symmetry group {G} has {len(G.generators)} generators."

    # Generate ESCNN representation of generators
    for g_gen, perm, refx in zip(G.generators, robot_cfg.permutation_Tips, robot_cfg.reflection_Tips):
        refx = np.array(refx, dtype=rep_field)
        rep_Tips[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)
    # Generate the entire group
    rep_Tips = group_rep_from_gens(G, rep_Tips)
    rep_Tips.name = 'Tips'
    G.representations.update(Tips=rep_Tips)
    rep_Tips = G.representations['Tips']

    #########################################################################
    # Here to build trifinger representation
    #########################################################################
    gspace = escnn.gspaces.no_base_space(G)
    trivial_repr = gspace.trivial_repr

    action_rep = rep_QJ
    cube_keypoints_rep = rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od  # 24
    prev_action_rep = action_rep  # 9
    cube_desired_keypoints_rep = rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od  # 24
    confidence_rep = trivial_repr  # 1
    delay_rep = trivial_repr  # 1
    dupli_cube_keypoints_rep = rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od + rep_Od  # 24
    # First transform quarterions into rotation matrix, then flatten the matrix
    cube_quarterion_rep = rep_Od + rep_Od + rep_Od  # 9
    # Cube position value [x, y, z] --> [x, y, z, 1], then use rep_Ed here
    cube_pos_rep = rep_Ed  # 4
    finger_tip_force_rep = rep_Tips  # 3
    # finger1 tip position [x1, y1, z1] --> [x1, y1, z1, 1], then use rep_Ed here
    # The same ops for other tips
    finger_tip_pos_rep = rep_Ed + rep_Ed + rep_Ed  # 12
    finger_tip_vel_rep = rep_Od + rep_Od + rep_Od  # 9
    finger_general_coors_reps = rep_QJ  # 9
    robot_ids_rep = trivial_repr  # 1
    finger_torques = rep_TqJ  # 9
    finger_general_vel_rep = rep_TqJ  # 9

    # Total dim of full state == 148
    full_state_rep = [
        cube_keypoints_rep,
        prev_action_rep,
        cube_desired_keypoints_rep,
        confidence_rep,
        delay_rep,
        dupli_cube_keypoints_rep,
        cube_quarterion_rep,
        cube_pos_rep,
        finger_tip_force_rep,
        finger_tip_pos_rep,
        finger_tip_vel_rep,
        finger_general_coors_reps,
        robot_ids_rep,
        finger_torques,
        finger_general_vel_rep,
    ]

    action_type = FieldType(gspace, [action_rep])
    full_state_type = FieldType(gspace, full_state_rep)   #
    full_state_action_type = FieldType(gspace, [*full_state_rep, action_rep])

    value_in_type = full_state_type
    value_out_type = FieldType(gspace, [trivial_repr])
    policy_in_type = full_state_action_type
    policy_out_type = action_type
    trivial_action_out_type = FieldType(gspace, [action_rep] * 9)

    return G, value_in_type, policy_in_type, value_out_type, policy_out_type, trivial_action_out_type

robot_name = 'Trifinger'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
initialize(config_path="cfg/robot", version_base='1.3')
trifinger_group_cfg = compose(config_name=f"{robot_name.lower()}.yaml")

Trifinger_G, Trifinger_value_in_type, Trifinger_policy_in_type, \
    Trifinger_value_out_type, Trifinger_policy_out_type, \
    Trifinger_trivial_action_out_type = load_trifinger_G(trifinger_group_cfg)
