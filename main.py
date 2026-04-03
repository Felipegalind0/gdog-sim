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
    """Reads the Jinja template, randomizes parameters, and returns a path to a temporary URDF."""
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

    return temp_urdf.name


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

    scene = gs.Scene(
        show_viewer=args.render,
        sim_options=gs.options.SimOptions(dt=0.01),
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
    urdf_path = generate_random_robot_urdf(rng)
    
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
    
    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        if len(dofs_idx) == 1:
            idx = dofs_idx[0]
            
            if "wheel" in joint.name:
                wheel_dofs.append(idx)
                wheel_names.append(joint.name)
            else:
                leg_dofs.append(idx)
                # Apply standing pose
                if "hip" in joint.name:
                    standing_dof_pos[idx] = 0.785
                elif "knee" in joint.name:
                    standing_dof_pos[idx] = -1.57

    leg_dofs = np.array(leg_dofs)
    wheel_dofs = np.array(wheel_dofs)
    target_leg_pos = standing_dof_pos[leg_dofs]

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
                f"heading:{np.degrees(heading):.1f}deg"
            )

    update_follow_camera()

    print("Starting WebRTC FastAPI thread...")
    flask_thread = threading.Thread(target=run_server, daemon=True)
    flask_thread.start()

    print("Starting simulation loop...")
    steps = 500 if args.video else 10000

    for i in range(steps):
        # 1. Maintain the standing position with the legs
        robot.control_dofs_position(target_leg_pos, dofs_idx_local=leg_dofs)
        
        # 2. Safely grab remote inputs from our WebRTC state
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

        # 3. Drive the wheels with velocity control
        robot.control_dofs_velocity(target_wheel_vel, dofs_idx_local=wheel_dofs)
        
        scene.step()
        update_follow_camera(cam_dx=cam_dx, cam_dy=cam_dy, cam_zoom=cam_zoom)
        
        if cam and i % 2 == 0: 
            cam.render()

    if args.video and cam:
        cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
        print("Video saved to wheeled_go2.mp4")

if __name__ == "__main__":
    main()