import genesis as gs
import argparse
import numpy as np
from jinja2 import Template
import tempfile
import os
import threading
import json
import importlib
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    print("aiortc not installed. WebRTC disabled. Using WebSockets as primary.")

SIM_DT = 0.01

# PID tuning parameters (meters of suspension height offset output).
ROLL_SETPOINT = 0.0
PITCH_SETPOINT = 0.0
ROLL_KP = 0.16
ROLL_KI = 0.00
ROLL_KD = 0.03
PITCH_KP = 0.16
PITCH_KI = 0.00
PITCH_KD = 0.03

MAX_PID_Z_ADJUST = 0.05
MAX_LEG_DELTA_Z = 0.06

HIP_ANGLE_MIN = -1.5
HIP_ANGLE_MAX = 1.5
KNEE_ANGLE_MIN = -2.5
KNEE_ANGLE_MAX = 0.0

NOMINAL_HIP_ANGLE = 0.785
NOMINAL_KNEE_ANGLE = -1.57
LEG_PREFIXES = ("fl", "fr", "rl", "rr")

PITCH_MIX_SIGN_DEFAULT = 1.0
ROLL_MIX_SIGN_DEFAULT = -1.0
KP_TUNE_STEP = 0.01
KI_TUNE_STEP = 0.001
KD_TUNE_STEP = 0.002
PID_GAIN_MAX = 2.0
RESPAWN_SETTLE_STEPS = 20


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limit=None, integral_limit=None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, measurement, dt):
        measurement = float(measurement)
        dt = max(float(dt), 1e-6)
        error = self.setpoint - measurement

        self.integral += error * dt
        if self.integral_limit is not None:
            integral_abs_limit = abs(float(self.integral_limit))
            self.integral = float(np.clip(self.integral, -integral_abs_limit, integral_abs_limit))

        if not self.initialized:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limit is not None:
            output_abs_limit = abs(float(self.output_limit))
            output = float(np.clip(output, -output_abs_limit, output_abs_limit))
        return float(output)


def _clip_pid_gain(value):
    return float(np.clip(float(value), 0.0, PID_GAIN_MAX))

class CommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self.vx = 0.0
        self.omega = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_zoom = 0.0

    def update(self, vx, omega, cam_dx=0.0, cam_dy=0.0, cam_zoom=0.0):
        with self._lock:
            self.vx = vx
            self.omega = omega
            # Camera inputs are interpreted as per-message deltas.
            self.cam_dx += cam_dx
            self.cam_dy += cam_dy
            self.cam_zoom += cam_zoom

    def get(self):
        with self._lock:
            out = (self.vx, self.omega, self.cam_dx, self.cam_dy, self.cam_zoom)
            self.cam_dx = 0.0
            self.cam_dy = 0.0
            self.cam_zoom = 0.0
            return out

state = CommandState()
app = FastAPI()


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_command_payload(payload):
    vx = _as_float(payload.get("vx", 0.0))
    omega = _as_float(payload.get("omega", 0.0))
    cam_dx = _as_float(payload.get("cam_dx", payload.get("dx", 0.0)))
    cam_dy = _as_float(payload.get("cam_dy", payload.get("dy", 0.0)))
    cam_zoom = _as_float(payload.get("cam_zoom", payload.get("zoom", 0.0)))
    return vx, omega, cam_dx, cam_dy, cam_zoom

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/offer")
async def offer(params: dict):
    if not HAS_WEBRTC:
        return {"error": "WebRTC not installed"}
        
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    
    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                state.update(*_parse_command_payload(data))
            except Exception:
                pass

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                state.update(*_parse_command_payload(parsed))
            except Exception:
                pass
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

def generate_random_robot_urdf(rng):
    """Reads the Jinja template, randomizes parameters, and returns a temporary URDF path and params."""
    with open('gdog.urdf.jinja', 'r') as f:
        template = Template(f.read())

    # Generate Random Parameters (Domain Randomization)
    robot_params = {
        'body_length': rng.uniform(0.4, 0.6),
        'body_width': rng.uniform(0.1, 0.2),
        'body_height': rng.uniform(0.08, 0.15),
        'body_mass': rng.uniform(3.0, 5.0),
        'leg_thickness': rng.uniform(0.04, 0.06),
        'thigh_length': rng.uniform(0.18, 0.25),
        'calf_thickness': rng.uniform(0.03, 0.05),
        'calf_length': rng.uniform(0.18, 0.25),
        'wheel_radius': rng.uniform(0.04, 0.08),
        'wheel_width': rng.uniform(0.03, 0.06)
    }

    # Render URDF XML string
    urdf_string = template.render(**robot_params)

    # Write to a temporary file
    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode='w')
    temp_urdf.write(urdf_string)
    temp_urdf.close()

    print("Generated random robot with parameters:")
    for k, v in robot_params.items():
        print(f"  {k}: {v:.3f}")

    return temp_urdf.name, robot_params


def generate_random_terrain_morph(rng):
    n_subterrains = (3, 3)
    subterrain_size = (8.0, 8.0)

    terrain_pool = [
        "flat_terrain",
        "random_uniform_terrain",
        "wave_terrain",
        "sloped_terrain",
        "pyramid_sloped_terrain",
    ]
    terrain_weights = np.array([0.35, 0.25, 0.15, 0.15, 0.10], dtype=np.float64)
    terrain_weights /= terrain_weights.sum()

    center_i = n_subterrains[0] // 2
    center_j = n_subterrains[1] // 2
    subterrain_types = []
    for i in range(n_subterrains[0]):
        row = []
        for j in range(n_subterrains[1]):
            # Keep the center patch flat to make spawn behavior stable.
            if i == center_i and j == center_j:
                row.append("flat_terrain")
            else:
                row.append(str(rng.choice(terrain_pool, p=terrain_weights)))
        subterrain_types.append(row)

    vertical_scale = float(rng.uniform(0.004, 0.01))
    horizontal_scale = 0.25

    subterrain_parameters = {
        "random_uniform_terrain": {
            "min_height": float(-rng.uniform(0.02, 0.08)),
            "max_height": float(rng.uniform(0.02, 0.08)),
            "step": float(rng.choice(np.array([0.02, 0.03, 0.04, 0.05]))),
            "downsampled_scale": float(rng.uniform(0.2, 0.6)),
        },
        "wave_terrain": {
            "num_waves": float(rng.integers(1, 4)),
            "amplitude": float(rng.uniform(0.03, 0.12)),
        },
        "sloped_terrain": {
            "slope": float(rng.uniform(-0.25, 0.25)),
        },
        "pyramid_sloped_terrain": {
            "slope": float(rng.uniform(-0.2, 0.2)),
        },
    }

    terrain_pos = (
        -0.5 * n_subterrains[0] * subterrain_size[0],
        -0.5 * n_subterrains[1] * subterrain_size[1],
        0.0,
    )

    terrain_morph = gs.morphs.Terrain(
        pos=terrain_pos,
        randomize=False,
        n_subterrains=n_subterrains,
        subterrain_size=subterrain_size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        subterrain_types=subterrain_types,
        subterrain_parameters=subterrain_parameters,
    )

    return terrain_morph, {
        "n_subterrains": n_subterrains,
        "subterrain_size": subterrain_size,
        "horizontal_scale": horizontal_scale,
        "vertical_scale": vertical_scale,
        "subterrain_types": subterrain_types,
    }


def generate_space_sky_texture(rng, height=512, width=1024, brightness=0.9):
    # Dark blue-black base keeps the sky space-like instead of flat gray.
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    gradient = 0.25 + 0.75 * (1.0 - y)

    sky = np.zeros((height, width, 3), dtype=np.float32)
    sky[..., 0] = 0.010 + 0.008 * gradient
    sky[..., 1] = 0.014 + 0.010 * gradient
    sky[..., 2] = 0.030 + 0.020 * gradient

    # Add a faint galactic band so the panorama reads as space, not monochrome.
    band_center = rng.uniform(0.35, 0.65)
    band_width = rng.uniform(0.06, 0.11)
    band = np.exp(-0.5 * ((y - band_center) / max(band_width, 1e-6)) ** 2)
    waves = 0.8 + 0.2 * np.sin(2.0 * np.pi * (3.0 * x + float(rng.uniform(0.0, 1.0))))
    band_strength = band * waves
    sky[..., 0] += 0.020 * band_strength
    sky[..., 1] += 0.028 * band_strength
    sky[..., 2] += 0.045 * band_strength

    # Dense star field with cool/warm white color variation.
    n_stars = int(height * width * 0.005)
    ys = rng.integers(0, height, size=n_stars)
    xs = rng.integers(0, width, size=n_stars)
    star_colors = np.clip(
        rng.normal(loc=np.array([0.94, 0.95, 1.00]), scale=np.array([0.08, 0.08, 0.06]), size=(n_stars, 3)),
        0.75,
        1.0,
    )
    star_luma = rng.uniform(0.55, 1.0, size=(n_stars, 1))
    star_rgb = star_colors * star_luma
    sky[ys, xs] = np.maximum(sky[ys, xs], star_rgb)

    # A handful of brighter stars with tiny cross bloom to improve readability.
    n_bright = max(36, n_stars // 50)
    by = rng.integers(1, height - 1, size=n_bright)
    bx = rng.integers(1, width - 1, size=n_bright)
    for y_idx, x_idx in zip(by, bx):
        core = float(rng.uniform(0.92, 1.0))
        halo = max(0.45, core - 0.40)
        core_rgb = np.array([core, core, core], dtype=np.float32)
        halo_rgb = np.array([halo, halo, halo], dtype=np.float32)
        sky[y_idx, x_idx] = np.maximum(sky[y_idx, x_idx], core_rgb)
        sky[y_idx - 1, x_idx] = np.maximum(sky[y_idx - 1, x_idx], halo_rgb)
        sky[y_idx + 1, x_idx] = np.maximum(sky[y_idx + 1, x_idx], halo_rgb)
        sky[y_idx, x_idx - 1] = np.maximum(sky[y_idx, x_idx - 1], halo_rgb)
        sky[y_idx, x_idx + 1] = np.maximum(sky[y_idx, x_idx + 1], halo_rgb)

    sky = np.clip(sky * float(brightness), 0.0, 1.0)
    return (255.0 * sky).astype(np.uint8)


def generate_moon_albedo_texture(rng, size=512):
    # Multi-scale noise creates low-cost moon-like albedo variation.
    coarse_1 = rng.normal(0.0, 1.0, size=(64, 64))
    coarse_2 = rng.normal(0.0, 1.0, size=(128, 128))
    noise_1 = np.repeat(np.repeat(coarse_1, 8, axis=0), 8, axis=1)[:size, :size]
    noise_2 = np.repeat(np.repeat(coarse_2, 4, axis=0), 4, axis=1)[:size, :size]
    h = 0.7 * noise_1 + 0.3 * noise_2

    # Add crater-like bowl/rim marks in the albedo to improve readability.
    crater_field = np.zeros((size, size), dtype=np.float32)
    n_craters = 70
    for _ in range(n_craters):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        radius = float(rng.uniform(8.0, 36.0))

        x0 = max(0, int(cx - 1.6 * radius))
        x1 = min(size, int(cx + 1.6 * radius) + 1)
        y0 = max(0, int(cy - 1.6 * radius))
        y1 = min(size, int(cy + 1.6 * radius) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d = dist / max(radius, 1e-6)

        bowl = -0.30 * np.exp(-2.0 * d * d)
        rim = 0.18 * np.exp(-((d - 1.0) / 0.22) ** 2)
        crater_field[y0:y1, x0:x1] += (bowl + rim).astype(np.float32)

    h = h + crater_field
    h = (h - h.min()) / max(h.max() - h.min(), 1e-8)

    base = 92.0 + 110.0 * h
    bands = 0.94 + 0.06 * np.sin(2.0 * np.pi * (8.0 * h + float(rng.uniform(0.0, 1.0))))
    albedo = np.clip(base * bands, 0.0, 255.0).astype(np.uint8)
    return np.stack([albedo, albedo, albedo], axis=-1)


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
    return float(
        thigh_length * np.cos(hip_angle) + calf_length * np.cos(hip_angle + knee_angle)
    )


def _ik_two_link_for_vertical_position(
    desired_leg_z,
    thigh_length,
    calf_length,
    hip_limits=(HIP_ANGLE_MIN, HIP_ANGLE_MAX),
    knee_limits=(KNEE_ANGLE_MIN, KNEE_ANGLE_MAX),
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

def main():
    parser = argparse.ArgumentParser(description="MVP Wheeled Robot Dog Simulator")
    parser.add_argument("--render", action="store_true", help="Enable interactive 3D viewer")
    parser.add_argument("--video", action="store_true", help="Record and save a video (mp4)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible robot and terrain randomization")
    args = parser.parse_args()

    # Genesis seed binding expects a signed 32-bit integer.
    seed_modulus = np.iinfo(np.int32).max + 1
    if args.seed is None:
        runtime_seed = int(np.random.SeedSequence().entropy) % seed_modulus
    else:
        runtime_seed = int(args.seed) % seed_modulus
    print(f"Using simulation seed: {runtime_seed}")

    rng = np.random.default_rng(runtime_seed)

    gs.init(backend=gs.gpu, logging_level="warning", seed=runtime_seed)
    sim_dt = SIM_DT

    scene = gs.Scene(
        show_viewer=args.render,
        sim_options=gs.options.SimOptions(dt=sim_dt),
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(1.5, -1.5, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=False, 
            background_color=(0.0, 0.0, 0.01),
            ambient_light=(0.9, 0.9, 0.9),
        )
    )

    sky_texture = generate_space_sky_texture(rng, brightness=0.9)
    sky_surface = gs.surfaces.Emission(
        emissive_texture=gs.textures.ImageTexture(image_array=sky_texture, encoding="srgb"),
        smooth=True,
        double_sided=True,
    )
    sky = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=260.0,
            fixed=True,
            collision=False,
            visualization=True,
        ),
        surface=sky_surface,
        name="space_sky",
    )

    terrain_morph, terrain_info = generate_random_terrain_morph(rng)
    moon_albedo = generate_moon_albedo_texture(rng)
    moon_surface = gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(image_array=moon_albedo, encoding="srgb"),
        roughness=0.92,
        metallic=0.02,
    )
    terrain = scene.add_entity(terrain_morph, surface=moon_surface, name="moon_terrain")

    print("Configured lunar visual theme: space sky sphere + moon terrain surface")

    print("Generated random terrain configuration:")
    print(f"  n_subterrains: {terrain_info['n_subterrains']}")
    print(f"  subterrain_size: {terrain_info['subterrain_size']}")
    print(f"  horizontal_scale: {terrain_info['horizontal_scale']:.3f}")
    print(f"  vertical_scale: {terrain_info['vertical_scale']:.4f}")
    print("  subterrain_types:")
    for row in terrain_info["subterrain_types"]:
        print(f"    {row}")

    # --- Randomization Step ---
    urdf_path, robot_params = generate_random_robot_urdf(rng)
    
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0, 0, 0.5) # Raised higher to ensure wheels don't clip on larger random spawns
        )
    )

    cam = None
    if args.video:
        cam = scene.add_camera(res=(1920, 1080), pos=(1.5, -1.5, 1.0), lookat=(0, 0, 0.3), fov=40, GUI=False)

    scene.build()

    # Clean up the temporary file now that Genesis has parsed it
    os.remove(urdf_path)

    # Separate Leg DoFs from Wheel DoFs
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

    thigh_length = float(robot_params["thigh_length"])
    calf_length = float(robot_params["calf_length"])
    nominal_leg_z = _leg_extension_from_angles(
        NOMINAL_HIP_ANGLE,
        NOMINAL_KNEE_ANGLE,
        thigh_length,
        calf_length,
    )

    roll_pid = PIDController(
        kp=ROLL_KP,
        ki=ROLL_KI,
        kd=ROLL_KD,
        setpoint=ROLL_SETPOINT,
        output_limit=MAX_PID_Z_ADJUST,
    )
    pitch_pid = PIDController(
        kp=PITCH_KP,
        ki=PITCH_KI,
        kd=PITCH_KD,
        setpoint=PITCH_SETPOINT,
        output_limit=MAX_PID_Z_ADJUST,
    )

    print(
        "Suspension IK parameters: "
        f"thigh_length={thigh_length:.3f} m, "
        f"calf_length={calf_length:.3f} m"
    )

    suspension_tune_lock = threading.Lock()
    suspension_tune = {
        "enabled": True,
        "pitch_mix_sign": float(PITCH_MIX_SIGN_DEFAULT),
        "roll_mix_sign": float(ROLL_MIX_SIGN_DEFAULT),
    }
    pending_actions = {"respawn": False}
    suspension_debug_text = "susp:init"

    def _suspension_tune_summary():
        with suspension_tune_lock:
            return (
                f"enabled={int(suspension_tune['enabled'])} "
                f"pitch(kp={pitch_pid.kp:.3f},ki={pitch_pid.ki:.3f},kd={pitch_pid.kd:.3f},"
                f"sign={suspension_tune['pitch_mix_sign']:+.0f}) "
                f"roll(kp={roll_pid.kp:.3f},ki={roll_pid.ki:.3f},kd={roll_pid.kd:.3f},"
                f"sign={suspension_tune['roll_mix_sign']:+.0f})"
            )

    def _reset_pid_tuning_to_defaults():
        with suspension_tune_lock:
            pitch_pid.kp = float(PITCH_KP)
            pitch_pid.ki = float(PITCH_KI)
            pitch_pid.kd = float(PITCH_KD)
            roll_pid.kp = float(ROLL_KP)
            roll_pid.ki = float(ROLL_KI)
            roll_pid.kd = float(ROLL_KD)
            suspension_tune["pitch_mix_sign"] = float(PITCH_MIX_SIGN_DEFAULT)
            suspension_tune["roll_mix_sign"] = float(ROLL_MIX_SIGN_DEFAULT)
            suspension_tune["enabled"] = True
        pitch_pid.reset()
        roll_pid.reset()

    if args.render:
        print("Live suspension tuning keys:")
        print("  Q/A: pitch Kp +/-    W/S: pitch Kd +/-    T/G: pitch Ki +/-")
        print("  E/D: roll  Kp +/-    R/F: roll  Kd +/-    Y/H: roll  Ki +/-")
        print("  P: toggle pitch sign | O: toggle roll sign | L: enable/disable suspension")
        print("  SPACE: respawn pose + reset PID integrators | BACKSPACE: reset defaults")

    # Initialize quaternion states for QPos
    qs = np.zeros(robot.n_qs)
    qs[0:3] = [0, 0, 0.5] 
    qs[3:7] = [1, 0, 0, 0] 
    
    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        qs_idx = np.atleast_1d(joint.qs_idx_local)
        if len(dofs_idx) == 1 and len(qs_idx) == 1:
            qs[qs_idx[0]] = standing_dof_pos[dofs_idx[0]]
            
    robot.set_qpos(qs)
    spawn_qs = qs.copy()
    respawn_settle_steps = 0

    if args.video and cam:
        cam.start_recording()

    camera_center_local = np.array([0.0, 0.0, 0.3], dtype=np.float64)
    camera_default_offset_local = np.array([-1.5, 0.0, 0.8], dtype=np.float64)
    camera_lat_sensitivity = 0.015
    camera_scroll_lon_sensitivity = 0.04
    camera_scroll_lat_sensitivity = 0.04
    camera_zoom_sensitivity = 0.12
    camera_yaw_follow_gain = -1.0
    camera_min_lat = np.deg2rad(-80.0)
    camera_max_lat = np.deg2rad(80.0)
    camera_min_zoom = 0.6
    camera_max_zoom = 8.0
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    cam_lon0, cam_lat0, cam_radius = _spherical_from_local_offset(camera_default_offset_local)
    cam_state = {
        "x": 0.0,
        "y": 0.0,
        "lon": float(cam_lon0),
        "lat": float(cam_lat0),
        "zoom": float(cam_radius),
    }
    prev_forward_xy = None
    sim_yaw_total = 0.0
    pyrender_viewer = scene.viewer._pyrender_viewer if (args.render and scene.viewer is not None) else None
    shift_held = False
    pending_shift_scroll = 0.0
    pending_scroll_lon = 0.0
    pending_scroll_lat = 0.0
    zoom_input_lock = threading.Lock()
    fps_value = 0.0
    fps_last_t = time.perf_counter()

    if pyrender_viewer is not None:
        viewer_module = importlib.import_module(type(pyrender_viewer).__module__)
        pyglet_key = viewer_module.pyglet.window.key
        original_on_key_press = pyrender_viewer.on_key_press
        original_on_key_release = pyrender_viewer.on_key_release

        def _on_key_press_with_shift(symbol, modifiers):
            nonlocal shift_held
            if symbol in (pyglet_key.LSHIFT, pyglet_key.RSHIFT) or (modifiers & pyglet_key.MOD_SHIFT):
                shift_held = True
            consumed = False

            with suspension_tune_lock:
                if symbol == pyglet_key.Q:
                    pitch_pid.kp = _clip_pid_gain(pitch_pid.kp + KP_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.A:
                    pitch_pid.kp = _clip_pid_gain(pitch_pid.kp - KP_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.W:
                    pitch_pid.kd = _clip_pid_gain(pitch_pid.kd + KD_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.S:
                    pitch_pid.kd = _clip_pid_gain(pitch_pid.kd - KD_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.T:
                    pitch_pid.ki = _clip_pid_gain(pitch_pid.ki + KI_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.G:
                    pitch_pid.ki = _clip_pid_gain(pitch_pid.ki - KI_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.E:
                    roll_pid.kp = _clip_pid_gain(roll_pid.kp + KP_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.D:
                    roll_pid.kp = _clip_pid_gain(roll_pid.kp - KP_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.R:
                    roll_pid.kd = _clip_pid_gain(roll_pid.kd + KD_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.F:
                    roll_pid.kd = _clip_pid_gain(roll_pid.kd - KD_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.Y:
                    roll_pid.ki = _clip_pid_gain(roll_pid.ki + KI_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.H:
                    roll_pid.ki = _clip_pid_gain(roll_pid.ki - KI_TUNE_STEP)
                    consumed = True
                elif symbol == pyglet_key.P:
                    suspension_tune["pitch_mix_sign"] *= -1.0
                    consumed = True
                elif symbol == pyglet_key.O:
                    suspension_tune["roll_mix_sign"] *= -1.0
                    consumed = True
                elif symbol == pyglet_key.L:
                    suspension_tune["enabled"] = not suspension_tune["enabled"]
                    if not suspension_tune["enabled"]:
                        pitch_pid.reset()
                        roll_pid.reset()
                    consumed = True
                elif symbol == pyglet_key.SPACE:
                    pending_actions["respawn"] = True
                    consumed = True
                elif symbol == pyglet_key.BACKSPACE:
                    consumed = True

            if symbol == pyglet_key.BACKSPACE:
                _reset_pid_tuning_to_defaults()

            if consumed:
                print(f"[suspension-tune] {_suspension_tune_summary()}")
                return True
            return original_on_key_press(symbol, modifiers)

        def _on_key_release_with_shift(symbol, modifiers):
            nonlocal shift_held
            if symbol in (pyglet_key.LSHIFT, pyglet_key.RSHIFT):
                shift_held = False
            elif not (modifiers & pyglet_key.MOD_SHIFT):
                shift_held = False
            return original_on_key_release(symbol, modifiers)

        def _on_mouse_scroll_camera_control(x, y, dx, dy):
            nonlocal pending_shift_scroll, pending_scroll_lon, pending_scroll_lat
            # Map scroll to deterministic camera state updates.
            with zoom_input_lock:
                if shift_held:
                    pending_shift_scroll += float(dy)
                else:
                    pending_scroll_lon += float(dx)
                    pending_scroll_lat -= float(dy)
            return True

        pyrender_viewer.on_key_press = _on_key_press_with_shift
        pyrender_viewer.on_key_release = _on_key_release_with_shift
        pyrender_viewer.on_mouse_scroll = _on_mouse_scroll_camera_control

    def update_follow_camera(cam_dx=0.0, cam_dy=0.0, cam_zoom=0.0):
        nonlocal prev_forward_xy, sim_yaw_total
        nonlocal cam_radius, pending_shift_scroll, pending_scroll_lon, pending_scroll_lat
        nonlocal fps_value, fps_last_t

        if not (args.render or cam):
            return

        base_pos = _to_numpy_1d(robot.get_pos())[:3]
        forward_xy = _forward_xy_from_quat_wxyz(robot.get_quat())
        heading = np.arctan2(forward_xy[1], forward_xy[0])

        cam_state["x"] = float(base_pos[0])
        cam_state["y"] = float(base_pos[1])

        yaw_delta_measured = 0.0
        if prev_forward_xy is not None:
            dot = float(np.clip(np.dot(prev_forward_xy, forward_xy), -1.0, 1.0))
            cross_z = float(prev_forward_xy[0] * forward_xy[1] - prev_forward_xy[1] * forward_xy[0])
            yaw_delta_measured = np.arctan2(cross_z, dot)
        prev_forward_xy = forward_xy

        sim_yaw_total += yaw_delta_measured
        cam_state["lon"] = _wrap_to_pi(cam_state["lon"] + camera_yaw_follow_gain * yaw_delta_measured)

        viewer_zoom_delta = 0.0
        viewer_scroll_lon_delta = 0.0
        viewer_scroll_lat_delta = 0.0
        with zoom_input_lock:
            if pending_shift_scroll != 0.0:
                viewer_zoom_delta = pending_shift_scroll
                pending_shift_scroll = 0.0
            if pending_scroll_lon != 0.0:
                viewer_scroll_lon_delta = pending_scroll_lon
                pending_scroll_lon = 0.0
            if pending_scroll_lat != 0.0:
                viewer_scroll_lat_delta = pending_scroll_lat
                pending_scroll_lat = 0.0

        cam_state["lon"] = _wrap_to_pi(cam_state["lon"] + camera_scroll_lon_sensitivity * viewer_scroll_lon_delta)

        camera_center = np.array(
            [
                cam_state["x"],
                cam_state["y"],
                float(base_pos[2] + camera_center_local[2]),
            ],
            dtype=np.float64,
        )

        if args.render and scene.viewer is not None:
            # Only vertical orbit (lat) is user-controlled via mouse drag.
            viewer_impl = scene.viewer._pyrender_viewer
            viewer_mouse_pressed = bool(getattr(viewer_impl, "viewer_flags", {}).get("mouse_pressed", False))
            trackball = getattr(viewer_impl, "_trackball", None)
            viewer_rotate_drag = False
            if trackball is not None:
                trackball_state = getattr(trackball, "_state", None)
                rotate_state = getattr(trackball, "STATE_ROTATE", 0)
                viewer_rotate_drag = viewer_mouse_pressed and trackball_state == rotate_state

            if viewer_rotate_drag:
                viewer_pos = _to_numpy_1d(scene.viewer.camera_pos)[:3]
                viewer_offset = viewer_pos - camera_center
                radius_obs = np.linalg.norm(viewer_offset)
                if radius_obs > 1e-9:
                    obs_lat = np.arcsin(np.clip(viewer_offset[2] / radius_obs, -1.0, 1.0))
                    cam_state["lat"] = float(np.clip(obs_lat, camera_min_lat, camera_max_lat))

        # Remote mouse/touch input and no-shift vertical scroll both change lat.
        cam_state["lat"] = float(
            np.clip(
                cam_state["lat"]
                + camera_lat_sensitivity * cam_dy
                + camera_scroll_lat_sensitivity * viewer_scroll_lat_delta,
                camera_min_lat,
                camera_max_lat,
            )
        )

        total_zoom_delta = float(cam_zoom + viewer_zoom_delta)
        if total_zoom_delta != 0.0:
            cam_radius *= np.exp(-camera_zoom_sensitivity * total_zoom_delta)
            cam_radius = float(np.clip(cam_radius, camera_min_zoom, camera_max_zoom))
        cam_state["zoom"] = float(cam_radius)

        now_t = time.perf_counter()
        dt = now_t - fps_last_t
        fps_last_t = now_t
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            fps_value = inst_fps if fps_value == 0.0 else (0.9 * fps_value + 0.1 * inst_fps)

        camera_local_offset = _local_offset_from_spherical(cam_state["lon"], cam_state["lat"], cam_radius)
        camera_pos = camera_center + camera_local_offset
        camera_pose = _pose_from_pos_lookat_up(camera_pos, camera_center, world_up)

        if args.render and scene.viewer is not None:
            scene.viewer.set_camera_pose(pose=camera_pose)

        if cam:
            cam.set_pose(pos=camera_pos, lookat=camera_center, up=world_up)

        if pyrender_viewer is not None:
            pyrender_viewer.set_message_text(
                f"x:{cam_state['x']:.3f} "
                f"y:{cam_state['y']:.3f} "
                f"lon:{np.degrees(cam_state['lon']):.1f}deg "
                f"lat:{np.degrees(cam_state['lat']):.1f}deg "
                f"zoom:{cam_state['zoom']:.3f} "
                f"fps:{fps_value:.1f} "
                f"yawTotal:{np.degrees(sim_yaw_total):.1f}deg "
                f"heading:{np.degrees(heading):.1f}deg "
                f"{suspension_debug_text}"
            )

    update_follow_camera()

    print("Starting WebRTC FastAPI thread...")
    flask_thread = threading.Thread(target=run_server, daemon=True)
    flask_thread.start()

    print("Starting simulation loop...")
    steps = 500 if args.video else 10000

    for i in range(steps):
        do_respawn = False
        with suspension_tune_lock:
            if pending_actions["respawn"]:
                pending_actions["respawn"] = False
                do_respawn = True
            suspension_enabled = bool(suspension_tune["enabled"])
            pitch_mix_sign = float(suspension_tune["pitch_mix_sign"])
            roll_mix_sign = float(suspension_tune["roll_mix_sign"])

        if do_respawn:
            robot.set_qpos(spawn_qs.copy())
            pitch_pid.reset()
            roll_pid.reset()
            respawn_settle_steps = RESPAWN_SETTLE_STEPS
            state.update(0.0, 0.0)

        # 1. Simulated IMU state extraction from base orientation.
        roll, pitch = _roll_pitch_from_quat_wxyz(robot.get_quat())

        # 2. PID roll/pitch stabilization outputs desired per-side Z adjustments.
        pitch_delta_z = 0.0
        roll_delta_z = 0.0
        target_leg_pos = standing_leg_pos.copy()

        if respawn_settle_steps > 0:
            respawn_settle_steps -= 1
        elif suspension_enabled:
            pitch_pid_output = float(np.clip(pitch_pid.update(pitch, sim_dt), -MAX_PID_Z_ADJUST, MAX_PID_Z_ADJUST))
            roll_pid_output = float(np.clip(roll_pid.update(roll, sim_dt), -MAX_PID_Z_ADJUST, MAX_PID_Z_ADJUST))

            pitch_delta_z = pitch_mix_sign * pitch_pid_output
            roll_delta_z = roll_mix_sign * roll_pid_output

            # 3. Mix pitch and roll corrections into each leg target.
            leg_delta_z = {
                "fl": pitch_delta_z - roll_delta_z,
                "fr": pitch_delta_z + roll_delta_z,
                "rl": -pitch_delta_z - roll_delta_z,
                "rr": -pitch_delta_z + roll_delta_z,
            }

            for prefix, delta_z in leg_delta_z.items():
                desired_leg_z = nominal_leg_z + float(np.clip(delta_z, -MAX_LEG_DELTA_Z, MAX_LEG_DELTA_Z))
                hip_angle, knee_angle = _ik_two_link_for_vertical_position(
                    desired_leg_z=desired_leg_z,
                    thigh_length=thigh_length,
                    calf_length=calf_length,
                )

                hip_dof = leg_joint_dofs[prefix]["hip"]
                knee_dof = leg_joint_dofs[prefix]["knee"]
                target_leg_pos[leg_dof_to_local_idx[hip_dof]] = hip_angle
                target_leg_pos[leg_dof_to_local_idx[knee_dof]] = knee_angle

        # 4. Apply leg position control from the suspension IK.
        robot.control_dofs_position(target_leg_pos, dofs_idx_local=leg_dofs)

        # 5. Safely grab remote inputs from our WebRTC state
        vx, omega, cam_dx, cam_dy, cam_zoom = state.get()
        
        # Calculate skid-steer kinematics:
        left_vel = vx - omega
        right_vel = vx + omega

        target_wheel_vel = np.zeros(len(wheel_dofs))
        for j, joint_name in enumerate(wheel_names):
            # Infer side from wheel prefix: fl/rl are left, fr/rr are right.
            wheel_prefix = joint_name.split("_", 1)[0].lower()
            if wheel_prefix in ("fl", "rl"):
                target_wheel_vel[j] = left_vel
            elif wheel_prefix in ("fr", "rr"):
                target_wheel_vel[j] = right_vel
            else:
                # Fallback for unexpected naming: keep translational command.
                target_wheel_vel[j] = vx

        # 6. Drive the wheels with velocity control
        robot.control_dofs_velocity(target_wheel_vel, dofs_idx_local=wheel_dofs)

        suspension_debug_text = (
            f"susp:{'on' if suspension_enabled else 'off'} "
            f"roll:{np.degrees(roll):+.1f}deg "
            f"pitch:{np.degrees(pitch):+.1f}deg "
            f"dzR:{roll_delta_z:+.3f} dzP:{pitch_delta_z:+.3f} "
            f"sgnR:{roll_mix_sign:+.0f} sgnP:{pitch_mix_sign:+.0f} "
            f"kpr:{roll_pid.kp:.2f} kdr:{roll_pid.kd:.2f} "
            f"kpp:{pitch_pid.kp:.2f} kdp:{pitch_pid.kd:.2f}"
        )
        
        scene.step()
        update_follow_camera(cam_dx=cam_dx, cam_dy=cam_dy, cam_zoom=cam_zoom)
        
        if cam and i % 2 == 0: 
            cam.render()

    if args.video and cam:
        cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
        print("Video saved to wheeled_go2.mp4")

if __name__ == "__main__":
    main()