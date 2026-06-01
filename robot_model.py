import numpy as np

from config import LEG_PREFIXES, NOMINAL_HIP_ANGLE, NOMINAL_KNEE_ANGLE
from math_utils import _leg_extension_from_angles


def discover_robot_joints(robot):
    leg_dofs = []
    wheel_dofs = []
    wheel_names = []
    standing_dof_pos = np.zeros(robot.n_dofs)
    leg_joint_dofs = {prefix: {"hip": None, "knee": None} for prefix in LEG_PREFIXES}

    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        if len(dofs_idx) == 1:
            idx = int(dofs_idx[0])
            joint_name = joint.name.lower()

            if "wheel" in joint_name:
                wheel_dofs.append(idx)
                wheel_names.append(joint.name)
            else:
                joint_prefix = joint_name.split("_", 1)[0]
                if joint_name.endswith("_hip") and joint_prefix in leg_joint_dofs:
                    leg_dofs.append(idx)
                    leg_joint_dofs[joint_prefix]["hip"] = idx
                    standing_dof_pos[idx] = NOMINAL_HIP_ANGLE
                elif joint_name.endswith("_knee") and joint_prefix in leg_joint_dofs:
                    leg_dofs.append(idx)
                    leg_joint_dofs[joint_prefix]["knee"] = idx
                    standing_dof_pos[idx] = NOMINAL_KNEE_ANGLE

    missing_leg_joints = []
    for prefix in LEG_PREFIXES:
        for joint_type in ("hip", "knee"):
            if leg_joint_dofs[prefix][joint_type] is None:
                missing_leg_joints.append(f"{prefix}_{joint_type}")
    if missing_leg_joints:
        raise RuntimeError(f"Missing expected leg joints: {', '.join(missing_leg_joints)}")

    leg_dofs = np.array(sorted(set(leg_dofs)), dtype=np.int32)
    wheel_dofs = np.array(wheel_dofs, dtype=np.int32)
    standing_leg_pos = standing_dof_pos[leg_dofs].copy()
    leg_dof_to_local_idx = {int(dof_idx): i for i, dof_idx in enumerate(leg_dofs)}

    return {
        "leg_dofs": leg_dofs,
        "wheel_dofs": wheel_dofs,
        "wheel_names": wheel_names,
        "standing_dof_pos": standing_dof_pos,
        "standing_leg_pos": standing_leg_pos,
        "leg_joint_dofs": leg_joint_dofs,
        "leg_dof_to_local_idx": leg_dof_to_local_idx,
    }


def compute_nominal_leg_z(thigh_length, calf_length):
    return _leg_extension_from_angles(
        NOMINAL_HIP_ANGLE,
        NOMINAL_KNEE_ANGLE,
        thigh_length,
        calf_length,
    )


def build_spawn_qpos(robot, standing_dof_pos):
    qs = np.zeros(robot.n_qs)
    qs[0:3] = [0, 0, 0.5]
    qs[3:7] = [1, 0, 0, 0]

    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        qs_idx = np.atleast_1d(joint.qs_idx_local)
        if len(dofs_idx) == 1 and len(qs_idx) == 1:
            qs[qs_idx[0]] = standing_dof_pos[dofs_idx[0]]

    return qs
