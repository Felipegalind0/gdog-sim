import os
import tempfile

import numpy as np

from config import BONE_RESPAWN_COOLDOWN_STEPS


def sample_random_terrain_xy(
    rng,
    terrain_info,
    min_center_distance=0.0,
    avoid_xy=None,
    min_avoid_distance=0.0,
):
    n_subterrains = terrain_info.get("n_subterrains") or (1, 1)
    subterrain_size = terrain_info.get("subterrain_size") or (4.0, 4.0)

    terrain_width = max(float(n_subterrains[0]) * float(subterrain_size[0]), 1e-3)
    terrain_depth = max(float(n_subterrains[1]) * float(subterrain_size[1]), 1e-3)

    margin = min(1.0, 0.1 * min(terrain_width, terrain_depth))
    x_min = -0.5 * terrain_width + margin
    x_max = 0.5 * terrain_width - margin
    y_min = -0.5 * terrain_depth + margin
    y_max = 0.5 * terrain_depth - margin

    if x_min >= x_max:
        x_min, x_max = -0.5 * terrain_width, 0.5 * terrain_width
    if y_min >= y_max:
        y_min, y_max = -0.5 * terrain_depth, 0.5 * terrain_depth

    min_center_distance = max(float(min_center_distance), 0.0)
    avoid_xy_vec = None
    if avoid_xy is not None:
        avoid_xy_vec = np.asarray(avoid_xy, dtype=float).reshape(-1)
        if avoid_xy_vec.size < 2:
            avoid_xy_vec = None
    min_avoid_distance = max(float(min_avoid_distance), 0.0)

    for _ in range(48):
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        if np.hypot(x, y) < min_center_distance:
            continue
        if avoid_xy_vec is not None and np.hypot(x - avoid_xy_vec[0], y - avoid_xy_vec[1]) < min_avoid_distance:
            continue
        if np.hypot(x, y) >= min_center_distance:
            return x, y

    return float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))


def sample_bone_spawn_pose(rng, terrain_info, avoid_xy=None):
    bone_x, bone_y = sample_random_terrain_xy(
        rng,
        terrain_info,
        min_center_distance=2.0,
        avoid_xy=avoid_xy,
        min_avoid_distance=2.5,
    )
    bone_spawn_z = float(rng.uniform(0.70, 1.10))
    bone_yaw_deg = float(rng.uniform(-180.0, 180.0))
    return bone_x, bone_y, bone_spawn_z, bone_yaw_deg


def create_temp_bone_urdf(shaft_length, shaft_radius, end_height, end_radius):
    shaft_length = max(float(shaft_length), 1e-3)
    shaft_radius = max(float(shaft_radius), 1e-3)
    end_height = max(float(end_height), 1e-3)
    end_radius = max(float(end_radius), 1e-3)

    density = 450.0
    shaft_mass = float(density * np.pi * shaft_radius * shaft_radius * shaft_length)
    end_mass = float(density * np.pi * end_radius * end_radius * end_height)
    total_mass = max(shaft_mass + 2.0 * end_mass, 1e-6)

    shaft_ixx = 0.5 * shaft_mass * shaft_radius * shaft_radius
    shaft_iyy = (shaft_mass / 12.0) * (3.0 * shaft_radius * shaft_radius + shaft_length * shaft_length)
    shaft_izz = shaft_iyy

    end_ixx_local = (end_mass / 12.0) * (3.0 * end_radius * end_radius + end_height * end_height)
    end_iyy_local = end_ixx_local
    end_izz_local = 0.5 * end_mass * end_radius * end_radius
    end_offset = 0.5 * shaft_length

    end_ixx = end_ixx_local
    end_iyy = end_iyy_local + end_mass * end_offset * end_offset
    end_izz = end_izz_local + end_mass * end_offset * end_offset

    ixx = max(shaft_ixx + 2.0 * end_ixx, 1e-9)
    iyy = max(shaft_iyy + 2.0 * end_iyy, 1e-9)
    izz = max(shaft_izz + 2.0 * end_izz, 1e-9)

    urdf_text = f"""<?xml version=\"1.0\"?>
<robot name=\"spawned_bone\">
    <link name=\"bone_link\">
        <inertial>
            <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
            <mass value=\"{total_mass:.8f}\"/>
            <inertia ixx=\"{ixx:.8f}\" ixy=\"0\" ixz=\"0\" iyy=\"{iyy:.8f}\" iyz=\"0\" izz=\"{izz:.8f}\"/>
        </inertial>

        <visual>
            <origin xyz=\"0 0 0\" rpy=\"0 1.57079632679 0\"/>
            <geometry>
                <cylinder radius=\"{shaft_radius:.8f}\" length=\"{shaft_length:.8f}\"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz=\"0 0 0\" rpy=\"0 1.57079632679 0\"/>
            <geometry>
                <cylinder radius=\"{shaft_radius:.8f}\" length=\"{shaft_length:.8f}\"/>
            </geometry>
        </collision>

        <visual>
            <origin xyz=\"{end_offset:.8f} 0 0\" rpy=\"0 0 0\"/>
            <geometry>
                <cylinder radius=\"{end_radius:.8f}\" length=\"{end_height:.8f}\"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz=\"{end_offset:.8f} 0 0\" rpy=\"0 0 0\"/>
            <geometry>
                <cylinder radius=\"{end_radius:.8f}\" length=\"{end_height:.8f}\"/>
            </geometry>
        </collision>

        <visual>
            <origin xyz=\"{-end_offset:.8f} 0 0\" rpy=\"0 0 0\"/>
            <geometry>
                <cylinder radius=\"{end_radius:.8f}\" length=\"{end_height:.8f}\"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz=\"{-end_offset:.8f} 0 0\" rpy=\"0 0 0\"/>
            <geometry>
                <cylinder radius=\"{end_radius:.8f}\" length=\"{end_height:.8f}\"/>
            </geometry>
        </collision>
    </link>
</robot>
"""

    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode="w")
    temp_urdf.write(urdf_text)
    temp_urdf.close()
    return temp_urdf.name


def spawn_bone_entity(scene, rng, terrain_info, gs):
    bone_x, bone_y, bone_spawn_z, bone_yaw_deg = sample_bone_spawn_pose(rng, terrain_info)
    bone_length = float(rng.uniform(0.22, 0.38))
    bone_radius = float(rng.uniform(0.022, 0.036))
    bone_cap_height = float(rng.uniform(0.10, 0.16))
    bone_cap_radius = float(bone_radius * rng.uniform(1.25, 1.70))
    bone_urdf_path = create_temp_bone_urdf(
        shaft_length=bone_length,
        shaft_radius=bone_radius,
        end_height=bone_cap_height,
        end_radius=bone_cap_radius,
    )

    bone_entity = scene.add_entity(
        gs.morphs.URDF(
            file=bone_urdf_path,
            pos=(bone_x, bone_y, bone_spawn_z),
            euler=(0.0, 0.0, bone_yaw_deg),
        ),
        name="spawned_bone",
    )
    print("Spawned random bone prop:")
    print(f"  pos: ({bone_x:.3f}, {bone_y:.3f}, {bone_spawn_z:.3f})")
    print(
        "  shape: "
        f"glued shaft length={bone_length:.3f} m, radius={bone_radius:.3f} m, yaw={bone_yaw_deg:+.1f} deg"
    )
    print(
        "  ends: "
        f"2 glued perpendicular cylinders height={bone_cap_height:.3f} m, radius={bone_cap_radius:.3f} m"
    )

    return bone_entity, bone_urdf_path


def check_bone_respawn(bone_entity, robot, rng, terrain_info, cooldown_steps_remaining):
    if cooldown_steps_remaining > 0:
        return cooldown_steps_remaining - 1

    robot_bone_contacts = robot.get_contacts(with_entity=bone_entity)
    robot_bone_geom_a = robot_bone_contacts.get("geom_a")
    has_robot_bone_contact = False
    if robot_bone_geom_a is not None:
        if hasattr(robot_bone_geom_a, "numel"):
            has_robot_bone_contact = bool(int(robot_bone_geom_a.numel()) > 0)
        else:
            has_robot_bone_contact = bool(np.asarray(robot_bone_geom_a).size > 0)

    if not has_robot_bone_contact:
        return 0

    robot_pos_after_step = robot.get_pos()
    if hasattr(robot_pos_after_step, "cpu"):
        robot_pos_after_step = robot_pos_after_step.cpu()
    robot_pos_after_step = np.asarray(robot_pos_after_step, dtype=float).reshape(-1)

    new_x, new_y, new_z, new_yaw_deg = sample_bone_spawn_pose(
        rng,
        terrain_info,
        avoid_xy=robot_pos_after_step[:2],
    )
    new_yaw_rad = float(np.deg2rad(new_yaw_deg))
    half_angle = 0.5 * new_yaw_rad
    bone_entity.set_pos((new_x, new_y, new_z), zero_velocity=True)
    bone_entity.set_quat(
        (float(np.cos(half_angle)), 0.0, 0.0, float(np.sin(half_angle))),
        zero_velocity=True,
        relative=False,
    )
    print(
        "Bone collision detected. Respawned bone at "
        f"({new_x:.3f}, {new_y:.3f}, {new_z:.3f}), yaw={new_yaw_deg:+.1f} deg"
    )

    return BONE_RESPAWN_COOLDOWN_STEPS
