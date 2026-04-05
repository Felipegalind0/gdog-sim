import numpy as np


def _to_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _yaw_from_quat_wxyz(quat):
    q = _to_numpy_1d(quat)[:4]
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return 0.0
    w, x, y, z = q / norm
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_from_quat_wxyz(quat):
    q = _to_numpy_1d(quat)[:4]
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return 0.0, 0.0
    w, x, y, z = q / norm

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return float(roll), float(pitch)


def _leg_extension_from_angles(hip_angle, knee_angle, thigh_length, calf_length):
    return float(thigh_length * np.cos(hip_angle) + calf_length * np.cos(hip_angle + knee_angle))


def _ik_two_link_for_vertical_position(
    desired_leg_z,
    thigh_length,
    calf_length,
    hip_limits=(-1.5, 1.5),
    knee_limits=(-2.5, 0.0),
):
    """Approximate sagittal-plane IK with target kept under the hip (x ~= 0)."""
    l1 = max(float(thigh_length), 1e-6)
    l2 = max(float(calf_length), 1e-6)

    knee_min, knee_max = knee_limits
    hip_min, hip_max = hip_limits

    reach_max = max(l1 + l2 - 1e-5, 1e-5)
    reach_min = np.sqrt(max(l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * np.cos(knee_min), 1e-10))
    z_target = float(np.clip(desired_leg_z, reach_min, reach_max))

    cos_knee = (z_target * z_target - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
    knee_angle = -np.arccos(cos_knee)
    knee_angle = float(np.clip(knee_angle, knee_min, knee_max))

    # x target is set to zero, so alpha is zero in this simplified leg geometry.
    hip_angle = -np.arctan2(l2 * np.sin(knee_angle), l1 + l2 * np.cos(knee_angle))
    hip_angle = float(np.clip(hip_angle, hip_min, hip_max))

    return hip_angle, knee_angle


def _forward_xy_from_quat_wxyz(quat):
    q = _to_numpy_1d(quat)[:4]
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    w, x, y, z = q / norm
    # World-frame projection of local +X axis onto XY plane.
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fxy = np.array([fx, fy], dtype=np.float64)
    fxy_norm = np.linalg.norm(fxy)
    if fxy_norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return fxy / fxy_norm


def _yaw_rotmat(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _spherical_from_local_offset(local_offset):
    local = _to_numpy_1d(local_offset)[:3]
    radius = max(np.linalg.norm(local), 1e-9)
    lat = np.arcsin(np.clip(local[2] / radius, -1.0, 1.0))
    lon = np.arctan2(local[1], -local[0])
    return lon, lat, radius


def _local_offset_from_spherical(lon, lat, radius):
    cos_lat = np.cos(lat)
    return np.array(
        [
            -radius * cos_lat * np.cos(lon),
            radius * cos_lat * np.sin(lon),
            radius * np.sin(lat),
        ],
        dtype=np.float64,
    )


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _pose_from_pos_lookat_up(pos, lookat, up):
    pos = _to_numpy_1d(pos)[:3]
    lookat = _to_numpy_1d(lookat)[:3]
    up = _to_numpy_1d(up)[:3]

    z_axis = pos - lookat
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        z_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm

    up_proj = up - np.dot(up, z_axis) * z_axis
    up_norm = np.linalg.norm(up_proj)
    if up_norm < 1e-9:
        up_proj = np.array([0.0, 0.0, 1.0], dtype=np.float64) - z_axis[2] * z_axis
        up_norm = np.linalg.norm(up_proj)
    if up_norm < 1e-9:
        up_proj = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = np.linalg.norm(up_proj)

    y_axis = up_proj / up_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = pos
    return pose
