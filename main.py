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
import difflib
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
ROLL_KP = 0.05
ROLL_KI = 0.05
ROLL_KD = 0.05
PITCH_KP = 0.01
PITCH_KI = 0.00
PITCH_KD = 0.00

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
PID_GAIN_MAX = 2.0
RESPAWN_SETTLE_STEPS = 20
COMMAND_BUFFER_MAX = 96
COMMAND_HISTORY_MAX = 100
KEYBOARD_VX_CMD = 6.0
KEYBOARD_YAW_CMD = 1.5
MAX_REMOTE_PITCH_SETPOINT_RAD = float(np.deg2rad(12.0))
MAX_REMOTE_ROLL_SETPOINT_RAD = float(np.deg2rad(12.0))
CANONICAL_CMD_KEYS = (
    "kp",
    "ki",
    "kd",
    "rp",
    "ri",
    "rd",
    "p_sign",
    "r_sign",
    "susp",
    "debug_pitch",
    "debug_roll",
    "debug_yaw",
    "debug_speed",
)


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

        # Live telemetry for debug visualization.
        self.last_measurement = 0.0
        self.last_dt = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output_unclipped = 0.0
        self.last_output = 0.0
        self.last_was_clipped = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

        self.last_measurement = 0.0
        self.last_dt = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output_unclipped = 0.0
        self.last_output = 0.0
        self.last_was_clipped = False

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
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        output_unclipped = float(output)
        was_clipped = False

        if self.output_limit is not None:
            output_abs_limit = abs(float(self.output_limit))
            output = float(np.clip(output, -output_abs_limit, output_abs_limit))

            # Tiny epsilon avoids false positives from fp rounding noise.
            was_clipped = abs(output - output_unclipped) > 1e-10

        self.last_measurement = measurement
        self.last_dt = dt
        self.last_error = float(error)
        self.last_derivative = float(derivative)
        self.last_p_term = float(p_term)
        self.last_i_term = float(i_term)
        self.last_d_term = float(d_term)
        self.last_output_unclipped = output_unclipped
        self.last_output = float(output)
        self.last_was_clipped = bool(was_clipped)

        return float(output)


def _clip_pid_gain(value):
    return float(np.clip(float(value), 0.0, PID_GAIN_MAX))


def _parse_bool_token(value):
    token = str(value).strip().lower()
    if token in ("1", "true", "on", "yes", "y"):
        return True
    if token in ("0", "false", "off", "no", "n"):
        return False
    return None


def _normalize_command_key(key):
    return str(key).strip().lower().replace("-", "_").replace(".", "_").replace(" ", "")


def _suggest_command_keys(key):
    norm_key = _normalize_command_key(key)
    suggestions = difflib.get_close_matches(norm_key, CANONICAL_CMD_KEYS, n=3, cutoff=0.5)
    if suggestions:
        return f"did you mean: {', '.join(suggestions)}?"
    return ""


def _unknown_key_message(key):
    key_raw = str(key).strip()
    suggestion_text = _suggest_command_keys(key_raw)
    base = (
        f"unknown key '{key_raw}'. valid keys: {', '.join(CANONICAL_CMD_KEYS)}"
    )
    return f"{base}; {suggestion_text}" if suggestion_text else base


def _value_hint_for_target(target):
    scope, field, _ = target
    if scope in ("pitch", "roll"):
        return f"float in [0, {PID_GAIN_MAX}]"
    if scope == "tune" and field == "enabled":
        return "on/off (or true/false/1/0)"
    if scope == "tune":
        return "-1 or 1 (or flip/toggle/invert)"
    if scope == "debug":
        return "on/off (or true/false/1/0)"
    return "valid value"


def _example_for_target(target):
    scope, field, _ = target
    if scope == "pitch":
        examples = {
            "kp": "kp=0.20",
            "ki": "ki=0.00",
            "kd": "kd=0.03",
        }
        return examples.get(field, "kp=0.20")
    if scope == "roll":
        examples = {
            "kp": "rp=0.20",
            "ki": "ri=0.00",
            "kd": "rd=0.03",
        }
        return examples.get(field, "rp=0.20")
    if scope == "tune" and field == "enabled":
        return "susp=on"
    if scope == "tune":
        return "p_sign=-1"
    if scope == "debug":
        return f"{field}=on"
    return "status"


def _resolve_command_target(key):
    norm_key = _normalize_command_key(key)
    key_map = {
        "p": ("pitch", "kp", "p"),
        "kp": ("pitch", "kp", "p"),
        "p_kp": ("pitch", "kp", "p"),
        "pkp": ("pitch", "kp", "p"),
        "pitch_kp": ("pitch", "kp", "p"),
        "pitchkp": ("pitch", "kp", "p"),
        "i": ("pitch", "ki", "i"),
        "ki": ("pitch", "ki", "i"),
        "p_ki": ("pitch", "ki", "i"),
        "pki": ("pitch", "ki", "i"),
        "pitch_ki": ("pitch", "ki", "i"),
        "pitchki": ("pitch", "ki", "i"),
        "d": ("pitch", "kd", "d"),
        "kd": ("pitch", "kd", "d"),
        "p_kd": ("pitch", "kd", "d"),
        "pkd": ("pitch", "kd", "d"),
        "pitch_kd": ("pitch", "kd", "d"),
        "pitchkd": ("pitch", "kd", "d"),
        "rp": ("roll", "kp", "rp"),
        "r_kp": ("roll", "kp", "rp"),
        "rkp": ("roll", "kp", "rp"),
        "roll_kp": ("roll", "kp", "rp"),
        "rollkp": ("roll", "kp", "rp"),
        "ri": ("roll", "ki", "ri"),
        "r_ki": ("roll", "ki", "ri"),
        "rki": ("roll", "ki", "ri"),
        "roll_ki": ("roll", "ki", "ri"),
        "rollki": ("roll", "ki", "ri"),
        "rd": ("roll", "kd", "rd"),
        "r_kd": ("roll", "kd", "rd"),
        "rkd": ("roll", "kd", "rd"),
        "roll_kd": ("roll", "kd", "rd"),
        "rollkd": ("roll", "kd", "rd"),
        "p_sign": ("tune", "pitch_mix_sign", "p_sign"),
        "psign": ("tune", "pitch_mix_sign", "p_sign"),
        "pitch_sign": ("tune", "pitch_mix_sign", "p_sign"),
        "r_sign": ("tune", "roll_mix_sign", "r_sign"),
        "rsign": ("tune", "roll_mix_sign", "r_sign"),
        "roll_sign": ("tune", "roll_mix_sign", "r_sign"),
        "susp": ("tune", "enabled", "susp"),
        "suspension": ("tune", "enabled", "susp"),
        "enabled": ("tune", "enabled", "susp"),
        "debug_pitch": ("debug", "debug_pitch", "debug_pitch"),
        "dbg_pitch": ("debug", "debug_pitch", "debug_pitch"),
        "debug_roll": ("debug", "debug_roll", "debug_roll"),
        "dbg_roll": ("debug", "debug_roll", "debug_roll"),
        "debug_yaw": ("debug", "debug_yaw", "debug_yaw"),
        "dbg_yaw": ("debug", "debug_yaw", "debug_yaw"),
        "debug_speed": ("debug", "debug_speed", "debug_speed"),
        "dbg_speed": ("debug", "debug_speed", "debug_speed"),
    }
    return key_map.get(norm_key)


def _read_command_target(target, pitch_pid, roll_pid, suspension_tune, debug_flags):
    scope, field, label = target
    if scope == "pitch":
        value = float(getattr(pitch_pid, field))
        return f"{label}={value:.6f}"
    if scope == "roll":
        value = float(getattr(roll_pid, field))
        return f"{label}={value:.6f}"
    if scope == "tune":
        if field == "enabled":
            return f"{label}={'on' if suspension_tune['enabled'] else 'off'}"
        return f"{label}={float(suspension_tune[field]):+.0f}"
    if scope == "debug":
        return f"{label}={'on' if debug_flags[field] else 'off'}"
    return f"unknown target: {label}"


def _set_command_target(target, value_token, pitch_pid, roll_pid, suspension_tune, debug_flags):
    scope, field, label = target
    if scope == "pitch":
        value = _clip_pid_gain(float(value_token))
        setattr(pitch_pid, field, value)
        return f"{label}={value:.6f}"
    if scope == "roll":
        value = _clip_pid_gain(float(value_token))
        setattr(roll_pid, field, value)
        return f"{label}={value:.6f}"
    if scope == "tune":
        if field == "enabled":
            bool_token = _parse_bool_token(value_token)
            if bool_token is None:
                bool_token = bool(float(value_token))
            suspension_tune["enabled"] = bool(bool_token)
            return f"{label}={'on' if suspension_tune['enabled'] else 'off'}"

        value_norm = str(value_token).strip().lower()
        if value_norm in ("flip", "toggle", "invert"):
            suspension_tune[field] *= -1.0
            return f"{label}={float(suspension_tune[field]):+.0f}"

        sign_value = 1.0 if float(value_token) >= 0.0 else -1.0
        suspension_tune[field] = sign_value
        return f"{label}={float(suspension_tune[field]):+.0f}"

    if scope == "debug":
        bool_token = _parse_bool_token(value_token)
        if bool_token is None:
            bool_token = bool(float(value_token))
        debug_flags[field] = bool(bool_token)
        return f"{label}={'on' if debug_flags[field] else 'off'}"

    raise ValueError(f"unknown target: {label}")


def _suspension_status_string(pitch_pid, roll_pid, suspension_tune, debug_flags):
    return (
        f"susp={'on' if suspension_tune['enabled'] else 'off'} "
        f"p={pitch_pid.kp:.4f} i={pitch_pid.ki:.4f} d={pitch_pid.kd:.4f} "
        f"rp={roll_pid.kp:.4f} ri={roll_pid.ki:.4f} rd={roll_pid.kd:.4f} "
        f"p_sign={float(suspension_tune['pitch_mix_sign']):+.0f} "
        f"r_sign={float(suspension_tune['roll_mix_sign']):+.0f} "
        f"debug_pitch={'on' if debug_flags['debug_pitch'] else 'off'} "
        f"debug_roll={'on' if debug_flags['debug_roll'] else 'off'} "
        f"debug_yaw={'on' if debug_flags['debug_yaw'] else 'off'} "
        f"debug_speed={'on' if debug_flags['debug_speed'] else 'off'}"
    )


def _execute_suspension_command(
    raw_cmd,
    pitch_pid,
    roll_pid,
    suspension_tune,
    debug_flags,
    pending_actions,
    reset_defaults_fn,
):
    cmd = str(raw_cmd).strip()
    if not cmd:
        return "parse error: empty command. expected '<key>=<value>' or '<key>?'. try 'help'"

    if cmd.startswith("/"):
        cmd = cmd[1:].strip()
    if not cmd:
        return "parse error: empty command after '/'. example: kp=0.2"

    lowered = cmd.lower()

    if lowered in ("help", "h", "?"):
        return (
            "syntax: <var_name>? reads value, <var_name>=<value> sets value. "
            "commands: status, p?/kp?, p=0.1, i?/ki?, d?/kd?, rp?, rp=0.1, ri=0.0, rd=0.02, "
            "p_sign=1|-1, r_sign=1|-1, susp=on|off, "
            "debug_pitch=on|off, debug_roll=on|off, debug_yaw=on|off, debug_speed=on|off, "
            "respawn, reset"
        )
    if lowered in ("status", "show", "st"):
        return _suspension_status_string(pitch_pid, roll_pid, suspension_tune, debug_flags)
    if lowered in ("respawn", "spawn", "rs"):
        pending_actions["respawn"] = True
        return "respawn requested"
    if lowered in ("reset", "defaults"):
        reset_defaults_fn()
        return "defaults restored"
    if lowered in ("pid_reset", "clear_i", "clear"):
        pitch_pid.reset()
        roll_pid.reset()
        return "pid state reset"

    if lowered.startswith("set "):
        cmd = cmd[4:].strip()
        lowered = cmd.lower()
    if lowered.startswith("get "):
        cmd = cmd[4:].strip() + "?"
        lowered = cmd.lower()

    if lowered.endswith("?"):
        key = lowered[:-1].strip()
        if not key:
            return "parse error: missing key before '?'. example: kp?"
        target = _resolve_command_target(key)
        if target is None:
            return _unknown_key_message(key)
        return _read_command_target(target, pitch_pid, roll_pid, suspension_tune, debug_flags)

    if "=" in cmd:
        key, value = cmd.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return "parse error: missing key before '='. example: kp=0.2"
        if not value:
            return f"parse error: missing value for '{key}'. example: {key}=0.2"

        target = _resolve_command_target(key)
        if target is None:
            return _unknown_key_message(key)
        try:
            return _set_command_target(target, value, pitch_pid, roll_pid, suspension_tune, debug_flags)
        except Exception as exc:
            hint = _value_hint_for_target(target)
            example = _example_for_target(target)
            return (
                f"invalid value for '{key}': {exc}. expected {hint}. "
                f"example: {example}"
            )

    return (
        f"parse error: '{cmd}'. expected '<key>=<value>' or '<key>?', "
        "or one of: status, help, respawn, reset, pid_reset"
    )

class CommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self.vx = 0.0
        self.omega = 0.0
        self.pitch_cmd = 0.0
        self.roll_cmd = 0.0
        self.cam_dx = 0.0
        self.cam_dy = 0.0
        self.cam_zoom = 0.0
        self.text_cmds = []

    def update(self, vx, omega, pitch_cmd=0.0, roll_cmd=0.0, cam_dx=0.0, cam_dy=0.0, cam_zoom=0.0, txt_cmd=None):
        with self._lock:
            self.vx = vx
            self.omega = omega
            self.pitch_cmd = pitch_cmd
            self.roll_cmd = roll_cmd
            # Camera inputs are interpreted as per-message deltas.
            self.cam_dx += cam_dx
            self.cam_dy += cam_dy
            self.cam_zoom += cam_zoom
            if txt_cmd:
                self.text_cmds.append(str(txt_cmd))

    def get(self):
        with self._lock:
            out = (
                self.vx,
                self.omega,
                self.pitch_cmd,
                self.roll_cmd,
                self.cam_dx,
                self.cam_dy,
                self.cam_zoom,
                list(self.text_cmds),
            )
            self.cam_dx = 0.0
            self.cam_dy = 0.0
            self.cam_zoom = 0.0
            self.text_cmds.clear()
            return out

    def push_command(self, txt_cmd):
        if txt_cmd is None:
            return
        cmd = str(txt_cmd).strip()
        if not cmd:
            return
        with self._lock:
            self.text_cmds.append(cmd)

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
    pitch_cmd = _as_float(payload.get("pitch_cmd", payload.get("pitch", 0.0)))
    roll_cmd = _as_float(payload.get("roll_cmd", payload.get("roll", 0.0)))
    cam_dx = _as_float(payload.get("cam_dx", payload.get("dx", 0.0)))
    cam_dy = _as_float(payload.get("cam_dy", payload.get("dy", 0.0)))
    cam_zoom = _as_float(payload.get("cam_zoom", payload.get("zoom", 0.0)))
    txt_cmd = payload.get("cmd", payload.get("command", None))
    return vx, omega, pitch_cmd, roll_cmd, cam_dx, cam_dy, cam_zoom, txt_cmd

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


def add_starry_sky_entities(scene, rng, sun_light_dir, star_count=90, sky_radius=230.0):
    dirs = rng.normal(size=(star_count, 3))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.clip(norms, 1e-9, None)

    # Slightly cool star tones read better against a black background.
    for star_idx, d in enumerate(dirs):
        pos = tuple((d * sky_radius).tolist())
        radius = float(rng.uniform(0.35, 0.95))
        star_intensity = float(rng.uniform(0.60, 1.00))
        color = (
            min(1.0, star_intensity * float(rng.uniform(0.92, 1.00))),
            min(1.0, star_intensity * float(rng.uniform(0.93, 1.00))),
            min(1.0, star_intensity * float(rng.uniform(0.96, 1.00))),
        )
        scene.add_entity(
            gs.morphs.Sphere(
                pos=pos,
                radius=radius,
                fixed=True,
                collision=False,
                visualization=True,
            ),
            surface=gs.surfaces.Emission(color=color),
            name=f"sky_star_{star_idx}",
        )

    sun_dir_vec = np.asarray(sun_light_dir, dtype=np.float64)
    sun_dir_norm = np.linalg.norm(sun_dir_vec)
    if sun_dir_norm < 1e-9:
        sun_dir_vec = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        sun_dir_norm = np.linalg.norm(sun_dir_vec)
    sun_dir_vec = sun_dir_vec / sun_dir_norm
    sun_pos = tuple((-sun_dir_vec * (sky_radius - 8.0)).tolist())

    scene.add_entity(
        gs.morphs.Sphere(
            pos=sun_pos,
            radius=10.0,
            fixed=True,
            collision=False,
            visualization=True,
        ),
        surface=gs.surfaces.Emission(color=(1.0, 0.97, 0.86)),
        name="sun_disk",
    )


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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override simulation steps. Default: video=500, headless=10000, render=unlimited. Use 0 or negative for unlimited.",
    )
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
    sun_light_dir = (-1.0, -1.0, -1.0)

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
            background_color=(0.0, 0.0, 0.0),
            ambient_light=(0.25, 0.25, 0.25),
            lights=[
                {
                    "type": "directional",
                    "dir": sun_light_dir,
                    "color": (1.0, 0.95, 0.90),
                    "intensity": 5.0,
                }
            ],
        )
    )

    add_starry_sky_entities(scene, rng, sun_light_dir)

    terrain_morph, terrain_info = generate_random_terrain_morph(rng)
    moon_albedo = generate_moon_albedo_texture(rng)
    moon_surface = gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(image_array=moon_albedo, encoding="srgb"),
        roughness=0.92,
        metallic=0.02,
    )
    terrain = scene.add_entity(terrain_morph, surface=moon_surface, name="moon_terrain")

    print("Configured lunar visual theme: starry sky + aligned sun + moon terrain surface")

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

    suspension_tune_lock = threading.RLock()
    suspension_tune = {
        "enabled": True,
        "pitch_mix_sign": float(PITCH_MIX_SIGN_DEFAULT),
        "roll_mix_sign": float(ROLL_MIX_SIGN_DEFAULT),
    }
    debug_flags = {
        "debug_pitch": True, #TODO set to false when pitch works
        "debug_roll": False,
        "debug_yaw": False,
        "debug_speed": False,
    }
    pending_actions = {"respawn": False}

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
            debug_flags["debug_pitch"] = False
            debug_flags["debug_roll"] = False
            debug_flags["debug_yaw"] = False
            debug_flags["debug_speed"] = False
        pitch_pid.reset()
        roll_pid.reset()

    if args.render:
        print("Suspension command console:")
        print("  Press '/' or 'T' to open the input line, type command, Enter to run, Esc to cancel.")
        print("  Examples: kp=0.08  ki=0.00  kd=0.02  rp=0.04  p_sign=-1  debug_pitch=on  debug_roll=on  susp=off  respawn  status  help")

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

    # Orbit target height above robot base position (meters).
    camera_center_height_offset = -0.2
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
    camera_center_height_scroll_sensitivity = 0.025
    camera_center_height_min = -1.0
    camera_center_height_max = 1.5
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
    ctrl_held = False
    command_mode = False
    command_buffer = ""
    command_history = []
    command_history_idx = None
    command_history_edit_buffer = ""
    last_cmd_result = "type help"
    pending_shift_scroll = 0.0
    pending_scroll_lon = 0.0
    pending_scroll_lat = 0.0
    pending_center_height_delta = 0.0
    zoom_input_lock = threading.Lock()
    keyboard_drive_lock = threading.Lock()
    keyboard_drive_keys = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
    }
    fps_value = 0.0
    fps_last_t = time.perf_counter()
    hud_status_text = "Susp ON\nP +0.0deg  R +0.0deg"
    hud_debug_lines = []
    hud_text_align = None
    hud_caption_font_name = "OpenSans-Regular"
    hud_caption_font_pt = 12
    hud_caption_color = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float64)

    if pyrender_viewer is not None:
        viewer_module = importlib.import_module(type(pyrender_viewer).__module__)
        pyglet_key = viewer_module.pyglet.window.key
        hud_text_align = viewer_module.TextAlign
        hud_caption_font_pt = int(max(10, round(float(getattr(viewer_module, "FONT_SIZE", 14.0)) * 0.90)))

        def _apply_message_style_override():
            text_padding = float(getattr(viewer_module, "TEXT_PADDING", 10.0))
            font_size = int(round(float(getattr(viewer_module, "FONT_SIZE", 14.0))))
            text_align = viewer_module.TextAlign

            command_help_lines = [
                "",
                "[commands]",
                "syntax:",
                "<var_name>? read value",
                "<var_name>=<value> set value",
                "[  / or t ]: open command input",
                "[  enter  ]: run command",
                "[   esc   ]: cancel command",
                "[cmd mode] up/down=history",
                "[drive] arrows up/down=fwd/back",
                "[drive] arrows left/right=yaw",
                "[camera] ctrl+scroll=target z",
                "kp/ki/kd, rp/ri/rd",
                "p_sign/r_sign, susp",
                "debug_pitch/debug_roll",
                "debug_yaw/debug_speed",
                "status, respawn, reset",
            ]

            gl_enable = getattr(viewer_module, "glEnable", None)
            gl_disable = getattr(viewer_module, "glDisable", None)
            gl_scissor = getattr(viewer_module, "glScissor", None)
            gl_clear_color = getattr(viewer_module, "glClearColor", None)
            gl_clear = getattr(viewer_module, "glClear", None)
            gl_scissor_test = getattr(viewer_module, "GL_SCISSOR_TEST", None)
            gl_color_buffer_bit = getattr(viewer_module, "GL_COLOR_BUFFER_BIT", None)

            def _render_help_text_white_on_black():
                if not pyrender_viewer._enable_help_text:
                    return

                if pyrender_viewer._message_text is not None:
                    if command_mode and all(v is not None for v in (
                        gl_enable,
                        gl_disable,
                        gl_scissor,
                        gl_clear_color,
                        gl_clear,
                        gl_scissor_test,
                        gl_color_buffer_bit,
                    )):
                        viewport_w = int(pyrender_viewer._viewport_size[0])
                        # Size the background to the actual number of visible message lines.
                        message_lines = max(1, str(pyrender_viewer._message_text).count("\n") + 1)
                        line_height = int(
                            max(
                                text_padding + font_size * float(message_lines),
                                font_size * 1.25,
                            )
                        )
                        gl_enable(gl_scissor_test)
                        gl_scissor(0, 0, viewport_w, line_height)
                        gl_clear_color(0.0, 0.0, 0.0, 1.0)
                        gl_clear(gl_color_buffer_bit)
                        gl_disable(gl_scissor_test)

                    pyrender_viewer._renderer.render_text(
                        pyrender_viewer._message_text,
                        pyrender_viewer._viewport_size[0] - text_padding,
                        text_padding,
                        font_pt=font_size,
                        color=np.array([1.0, 1.0, 1.0, np.clip(pyrender_viewer._message_opac, 0.0, 1.0)]),
                        align=text_align.BOTTOM_RIGHT,
                    )

                    if pyrender_viewer._message_opac > 1.0:
                        pyrender_viewer._message_opac -= 1.0
                    else:
                        pyrender_viewer._message_opac *= 0.90

                    if pyrender_viewer._message_opac < 0.05:
                        pyrender_viewer._message_opac = 1.0 + pyrender_viewer._ticks_till_fade
                        pyrender_viewer._message_text = None

                if pyrender_viewer._collapse_instructions:
                    pyrender_viewer._renderer.render_texts(
                        pyrender_viewer._instr_texts[0],
                        text_padding,
                        pyrender_viewer._viewport_size[1] - text_padding,
                        font_pt=font_size,
                        color=np.array([1.0, 1.0, 1.0, 0.85]),
                    )
                else:
                    expanded_lines = list(pyrender_viewer._key_instr_texts) + command_help_lines
                    pyrender_viewer._renderer.render_texts(
                        expanded_lines,
                        text_padding,
                        pyrender_viewer._viewport_size[1] - text_padding,
                        font_pt=font_size,
                        color=np.array([1.0, 1.0, 1.0, 0.85]),
                    )

                if hud_debug_lines:
                    debug_x = pyrender_viewer._viewport_size[0] - text_padding
                    debug_y = pyrender_viewer._viewport_size[1] - text_padding
                    line_step = int(max(1.0, font_size * 1.1))
                    for line_idx, line in enumerate(hud_debug_lines):
                        pyrender_viewer._renderer.render_text(
                            line,
                            debug_x,
                            int(debug_y - line_idx * line_step),
                            font_pt=font_size,
                            color=np.array([1.0, 1.0, 1.0, 0.95]),
                            align=text_align.TOP_RIGHT,
                        )

            pyrender_viewer._render_help_text = _render_help_text_white_on_black

        _apply_message_style_override()

        shifted_char_map = {
            "1": "!",
            "2": "@",
            "3": "#",
            "4": "$",
            "5": "%",
            "6": "^",
            "7": "&",
            "8": "*",
            "9": "(",
            "0": ")",
            "-": "_",
            "=": "+",
            "[": "{",
            "]": "}",
            "\\": "|",
            ";": ":",
            "'": '"',
            ",": "<",
            ".": ">",
            "/": "?",
            "`": "~",
        }
        mod_ctrl = int(getattr(pyglet_key, "MOD_CTRL", 0))
        left_ctrl = getattr(pyglet_key, "LCTRL", None)
        right_ctrl = getattr(pyglet_key, "RCTRL", None)

        def _on_key_press_with_shift(symbol, modifiers):
            nonlocal shift_held, ctrl_held, command_mode, command_buffer, last_cmd_result
            nonlocal command_history, command_history_idx, command_history_edit_buffer
            if symbol in (pyglet_key.LSHIFT, pyglet_key.RSHIFT) or (modifiers & pyglet_key.MOD_SHIFT):
                shift_held = True
            if symbol in (left_ctrl, right_ctrl) or (modifiers & mod_ctrl):
                ctrl_held = True

            if command_mode:
                num_enter = getattr(pyglet_key, "NUM_ENTER", pyglet_key.ENTER)
                if symbol in (pyglet_key.ENTER, num_enter):
                    cmd_to_send = command_buffer.strip()
                    if cmd_to_send:
                        state.push_command(cmd_to_send)
                        command_history.append(cmd_to_send)
                        if len(command_history) > COMMAND_HISTORY_MAX:
                            command_history = command_history[-COMMAND_HISTORY_MAX:]
                        last_cmd_result = f"queued: {cmd_to_send}"
                    else:
                        last_cmd_result = "empty command"
                    command_buffer = ""
                    command_history_idx = None
                    command_history_edit_buffer = ""
                    command_mode = False
                    return True

                if symbol == pyglet_key.ESCAPE:
                    command_mode = False
                    command_buffer = ""
                    command_history_idx = None
                    command_history_edit_buffer = ""
                    last_cmd_result = "command canceled"
                    return True

                if symbol == pyglet_key.UP:
                    if command_history:
                        if command_history_idx is None:
                            command_history_edit_buffer = command_buffer
                            command_history_idx = len(command_history) - 1
                        elif command_history_idx > 0:
                            command_history_idx -= 1
                        command_buffer = command_history[command_history_idx]
                    return True

                if symbol == pyglet_key.DOWN:
                    if command_history_idx is not None:
                        if command_history_idx < len(command_history) - 1:
                            command_history_idx += 1
                            command_buffer = command_history[command_history_idx]
                        else:
                            command_history_idx = None
                            command_buffer = command_history_edit_buffer
                    return True

                if symbol == pyglet_key.BACKSPACE:
                    command_buffer = command_buffer[:-1]
                    return True

                if 32 <= symbol <= 126:
                    ch = chr(symbol)
                    if ch.isalpha():
                        if modifiers & pyglet_key.MOD_SHIFT:
                            ch = ch.upper()
                        else:
                            ch = ch.lower()
                    elif modifiers & pyglet_key.MOD_SHIFT:
                        ch = shifted_char_map.get(ch, ch)

                    if len(command_buffer) < COMMAND_BUFFER_MAX:
                        command_buffer += ch
                    return True

                return True

            if symbol in (pyglet_key.SLASH, pyglet_key.T):
                command_mode = True
                command_buffer = "/" if symbol == pyglet_key.SLASH else ""
                command_history_idx = None
                command_history_edit_buffer = command_buffer
                return True

            if symbol in (pyglet_key.UP, pyglet_key.DOWN, pyglet_key.LEFT, pyglet_key.RIGHT):
                with keyboard_drive_lock:
                    if symbol == pyglet_key.UP:
                        keyboard_drive_keys["up"] = True
                    elif symbol == pyglet_key.DOWN:
                        keyboard_drive_keys["down"] = True
                    elif symbol == pyglet_key.LEFT:
                        keyboard_drive_keys["left"] = True
                    elif symbol == pyglet_key.RIGHT:
                        keyboard_drive_keys["right"] = True
                return True

            return None

        def _on_key_release_with_shift(symbol, modifiers):
            nonlocal shift_held, ctrl_held, command_mode
            if symbol in (pyglet_key.LSHIFT, pyglet_key.RSHIFT):
                shift_held = False
            elif not (modifiers & pyglet_key.MOD_SHIFT):
                shift_held = False

            if symbol in (left_ctrl, right_ctrl):
                ctrl_held = False
            elif not (modifiers & mod_ctrl):
                ctrl_held = False

            if symbol in (pyglet_key.UP, pyglet_key.DOWN, pyglet_key.LEFT, pyglet_key.RIGHT):
                with keyboard_drive_lock:
                    if symbol == pyglet_key.UP:
                        keyboard_drive_keys["up"] = False
                    elif symbol == pyglet_key.DOWN:
                        keyboard_drive_keys["down"] = False
                    elif symbol == pyglet_key.LEFT:
                        keyboard_drive_keys["left"] = False
                    elif symbol == pyglet_key.RIGHT:
                        keyboard_drive_keys["right"] = False
                return True

            if command_mode:
                return True

            return None

        def _on_mouse_scroll_camera_control(x, y, dx, dy):
            nonlocal pending_shift_scroll, pending_scroll_lon, pending_scroll_lat, pending_center_height_delta
            # Map scroll to deterministic camera state updates.
            with zoom_input_lock:
                if ctrl_held:
                    pending_center_height_delta += float(dy)
                elif shift_held:
                    pending_shift_scroll += float(dy)
                else:
                    pending_scroll_lon += float(dx)
                    pending_scroll_lat -= float(dy)
            return True

        pyrender_viewer.push_handlers(
            on_key_press=_on_key_press_with_shift,
            on_key_release=_on_key_release_with_shift,
            on_mouse_scroll=_on_mouse_scroll_camera_control,
        )

    def update_follow_camera(cam_dx=0.0, cam_dy=0.0, cam_zoom=0.0):
        nonlocal prev_forward_xy, sim_yaw_total
        nonlocal cam_radius, camera_center_height_offset
        nonlocal pending_shift_scroll, pending_scroll_lon, pending_scroll_lat, pending_center_height_delta
        nonlocal fps_value, fps_last_t
        nonlocal hud_status_text, hud_debug_lines

        if not (args.render or cam):
            return

        base_pos = _to_numpy_1d(robot.get_pos())[:3]
        forward_xy = _forward_xy_from_quat_wxyz(robot.get_quat())

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
        viewer_center_height_delta = 0.0
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
            if pending_center_height_delta != 0.0:
                viewer_center_height_delta = pending_center_height_delta
                pending_center_height_delta = 0.0

        if viewer_center_height_delta != 0.0:
            camera_center_height_offset += (
                camera_center_height_scroll_sensitivity * viewer_center_height_delta
            )
            camera_center_height_offset = float(
                np.clip(
                    camera_center_height_offset,
                    camera_center_height_min,
                    camera_center_height_max,
                )
            )

        cam_state["lon"] = _wrap_to_pi(cam_state["lon"] + camera_scroll_lon_sensitivity * viewer_scroll_lon_delta)

        camera_center = np.array(
            [
                cam_state["x"],
                cam_state["y"],
                float(base_pos[2] + camera_center_height_offset),
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
            if hud_text_align is not None:
                captions = [
                    {
                        "text": hud_status_text,
                        "location": hud_text_align.TOP_CENTER,
                        "font_name": hud_caption_font_name,
                        "font_pt": hud_caption_font_pt,
                        "color": hud_caption_color,
                        "scale": 1.0,
                    },
                ]
                pyrender_viewer.viewer_flags["caption"] = captions

            if command_mode:
                pyrender_viewer.set_message_text(f"% {command_buffer}")
            else:
                pyrender_viewer._message_text = None

    update_follow_camera()

    print("Starting WebRTC FastAPI thread...")
    flask_thread = threading.Thread(target=run_server, daemon=True)
    flask_thread.start()

    print("Starting simulation loop...")
    if args.steps is not None:
        steps = int(args.steps)
    elif args.video:
        steps = 500
    elif args.render:
        steps = 0
    else:
        steps = 10000

    if steps <= 0:
        print("Simulation loop steps: unlimited (Ctrl+C to stop)")
    else:
        print(f"Simulation loop steps: {steps}")

    i = 0
    while steps <= 0 or i < steps:
        vx, omega, pitch_cmd, roll_cmd, cam_dx, cam_dy, cam_zoom, text_cmds = state.get()

        pitch_pid.setpoint = float(np.clip(pitch_cmd, -1.0, 1.0)) * MAX_REMOTE_PITCH_SETPOINT_RAD
        roll_pid.setpoint = float(np.clip(roll_cmd, -1.0, 1.0)) * MAX_REMOTE_ROLL_SETPOINT_RAD

        with keyboard_drive_lock:
            kb_up = bool(keyboard_drive_keys["up"])
            kb_down = bool(keyboard_drive_keys["down"])
            kb_left = bool(keyboard_drive_keys["left"])
            kb_right = bool(keyboard_drive_keys["right"])

        if kb_up or kb_down or kb_left or kb_right:
            fwd_sign = (1.0 if kb_up else 0.0) - (1.0 if kb_down else 0.0)
            yaw_sign = (1.0 if kb_right else 0.0) - (1.0 if kb_left else 0.0)
            vx = fwd_sign * KEYBOARD_VX_CMD
            omega = yaw_sign * KEYBOARD_YAW_CMD

        do_respawn = False
        with suspension_tune_lock:
            for text_cmd in text_cmds:
                cmd_result = _execute_suspension_command(
                    raw_cmd=text_cmd,
                    pitch_pid=pitch_pid,
                    roll_pid=roll_pid,
                    suspension_tune=suspension_tune,
                    debug_flags=debug_flags,
                    pending_actions=pending_actions,
                    reset_defaults_fn=_reset_pid_tuning_to_defaults,
                )
                last_cmd_result = cmd_result
                print(f"[suspension-cmd] {text_cmd} -> {cmd_result}")

            if pending_actions["respawn"]:
                pending_actions["respawn"] = False
                do_respawn = True
            suspension_enabled = bool(suspension_tune["enabled"])
            pitch_mix_sign = float(suspension_tune["pitch_mix_sign"])
            roll_mix_sign = float(suspension_tune["roll_mix_sign"])
            debug_pitch_enabled = bool(debug_flags["debug_pitch"])
            debug_roll_enabled = bool(debug_flags["debug_roll"])
            debug_yaw_enabled = bool(debug_flags["debug_yaw"])
            debug_speed_enabled = bool(debug_flags["debug_speed"])

        if do_respawn:
            robot.set_qpos(spawn_qs.copy())
            pitch_pid.reset()
            roll_pid.reset()
            respawn_settle_steps = RESPAWN_SETTLE_STEPS
            state.update(0.0, 0.0)
            vx = 0.0
            omega = 0.0

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
        
        # 5. Calculate skid-steer kinematics:
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

        hud_status_text = (
            f"Susp {'ON' if suspension_enabled else 'OFF'}\n"
            f"P {np.degrees(pitch):+.1f}deg  R {np.degrees(roll):+.1f}deg"
        )
        if ctrl_held:
            hud_status_text += f"\nCam target z {camera_center_height_offset:+.3f} m"
        debug_lines = []
        if debug_pitch_enabled:
            pitch_clip = "YES" if pitch_pid.last_was_clipped else "no"
            debug_lines.extend(
                [
                    "[Pitch PID]",
                    f" Kp {pitch_pid.kp:.3f}",
                    f" Ki {pitch_pid.ki:.3f}",
                    f" Kd {pitch_pid.kd:.3f}",
                    f" angle {np.degrees(pitch):+.2f} deg",
                    f" err {pitch_pid.last_error:+.4f} rad",
                    f" P {pitch_pid.last_p_term:+.4f}",
                    f" I {pitch_pid.last_i_term:+.4f}",
                    f" D {pitch_pid.last_d_term:+.4f}",
                    f" out {pitch_pid.last_output:+.4f}",
                    f" raw {pitch_pid.last_output_unclipped:+.4f}",
                    f" clip {pitch_clip}",
                    f" dz {pitch_delta_z:+.3f} m",
                ]
            )
        if debug_roll_enabled:
            roll_clip = "YES" if roll_pid.last_was_clipped else "no"
            if debug_lines:
                debug_lines.append("")
            debug_lines.extend(
                [
                    "[Roll PID]",
                    f" Kp {roll_pid.kp:.3f}",
                    f" Ki {roll_pid.ki:.3f}",
                    f" Kd {roll_pid.kd:.3f}",
                    f" angle {np.degrees(roll):+.2f} deg",
                    f" err {roll_pid.last_error:+.4f} rad",
                    f" P {roll_pid.last_p_term:+.4f}",
                    f" I {roll_pid.last_i_term:+.4f}",
                    f" D {roll_pid.last_d_term:+.4f}",
                    f" out {roll_pid.last_output:+.4f}",
                    f" raw {roll_pid.last_output_unclipped:+.4f}",
                    f" clip {roll_clip}",
                    f" dz {roll_delta_z:+.3f} m",
                ]
            )
        if debug_yaw_enabled:
            forward_xy_debug = _forward_xy_from_quat_wxyz(robot.get_quat())
            heading_debug = np.arctan2(forward_xy_debug[1], forward_xy_debug[0])
            if debug_lines:
                debug_lines.append("")
            debug_lines.extend(
                [
                    "[Yaw Debug]",
                    f" omega_cmd {omega:+.3f} rad/s",
                    f" heading {np.degrees(heading_debug):+.2f} deg",
                    f" yawTotal {np.degrees(sim_yaw_total):+.2f} deg",
                ]
            )
        if debug_speed_enabled:
            if debug_lines:
                debug_lines.append("")
            debug_lines.extend(
                [
                    "[Speed Debug]",
                    f" vx_cmd {vx:+.3f} m/s",
                    f" omega_cmd {omega:+.3f} rad/s",
                    f" wheel_left {left_vel:+.3f}",
                    f" wheel_right {right_vel:+.3f}",
                ]
            )
        hud_debug_lines = list(debug_lines)
        
        try:
            scene.step()
        except Exception as exc:
            if args.render and "Viewer closed." in str(exc):
                print("Viewer closed. Exiting simulation loop.")
                break
            raise

        update_follow_camera(cam_dx=cam_dx, cam_dy=cam_dy, cam_zoom=cam_zoom)
        
        if cam and i % 2 == 0: 
            cam.render()

        i += 1

    if args.video and cam:
        cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
        print("Video saved to wheeled_go2.mp4")

if __name__ == "__main__":
    main()