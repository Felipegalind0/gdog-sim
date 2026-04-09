import argparse
import importlib
import os
import re
import select
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import numpy as np

try:
    import genesis as gs
except ModuleNotFoundError as exc:
    if exc.name in {"genesis", "quadrants", "gstaichi", "taichi"}:
        raise SystemExit(
            "Genesis is unavailable in this environment. "
            "Install with 'python -m pip install -r requirements.txt' on a supported platform. "
            "If you are on Linux ARM64 (e.g., Ubuntu 24 DGX Spark), run `./scripts/install_ubuntu_arm64.sh` to compile missing binaries from source."
        ) from exc
    raise

from camera_controller import FollowCameraController
from commands import CommandState, _execute_suspension_command
from math_utils import (
    _forward_xy_from_quat_wxyz,
    _ik_two_link_for_vertical_position,
    _leg_extension_from_angles,
    _roll_pitch_from_quat_wxyz,
)
from network import run_server
from pid import PIDController
from procedural_gen import (
    generate_moon_albedo_texture,
    generate_random_robot_urdf,
    generate_random_terrain_morph,
)


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
RESPAWN_SETTLE_STEPS = 20
COMMAND_BUFFER_MAX = 96
COMMAND_HISTORY_MAX = 100
KEYBOARD_VX_CMD = 24.0
KEYBOARD_YAW_CMD = -8.0
ROTATE_ONLY_VX_DEADBAND = 0.15
ROTATE_ONLY_OMEGA_MIN = 0.10
ROTATE_DRIFT_KP = 24.0
ROTATE_DRIFT_KI = 6.0
ROTATE_DRIFT_KD = 3.0
ROTATE_DRIFT_I_LIMIT = 0.30
ROTATE_DRIFT_MAX_CORRECTION = 7.0
DRIVE_VX_ACCEL_LIMIT = 40.0
DRIVE_VX_DECEL_LIMIT = 55.0
DRIVE_STATE_FILTER_ALPHA = 0.25
TRACTION_NEUTRAL_SPEED_MPS = 0.15
TRACTION_NEUTRAL_CMD = 0.5
TRACTION_NEUTRAL_PITCH_ALPHA = 0.02
TRACTION_PITCH_WARN_RAD = float(np.deg2rad(7.0))
TRACTION_PITCH_BLOCK_RAD = float(np.deg2rad(14.0))
TRACTION_PITCH_RATE_WARN_RAD_S = float(np.deg2rad(40.0))
TRACTION_PITCH_RATE_BLOCK_RAD_S = float(np.deg2rad(130.0))
TRACTION_CMD_ACCEL_WARN = 10.0
TRACTION_CMD_ACCEL_BLOCK = 45.0
TRACTION_RISK_START = 0.35
TRACTION_RISK_HARD_BLOCK = 0.90
TRACTION_MIN_SCALE = 0.05
ANTI_TIP_BALANCE_ACCEL_REF = 25.0
ANTI_TIP_PITCH_DAMP_GAIN = 0.18
ANTI_TIP_PITCH_RESTORE_GAIN = 0.55
ANTI_TIP_RISK_BOOST_GAIN = 0.60
ANTI_TIP_MAX_SETPOINT_BIAS_RAD = float(np.deg2rad(8.0))
MAX_REMOTE_PITCH_SETPOINT_RAD = float(np.deg2rad(12.0))
MAX_REMOTE_ROLL_SETPOINT_RAD = float(np.deg2rad(12.0))
VOICE_PWM_PERIOD_STEPS = 10
VOICE_MOVE_STOP_TOL_M = 0.02
VOICE_MOVE_PULSE_NEAR_M = 0.10
VOICE_MOVE_PULSE_MID_M = 0.30
VOICE_MOVE_MAX_LATERAL_ERROR_M = 0.30
VOICE_ROT_STOP_TOL_RAD = float(np.deg2rad(2.0))
VOICE_ROT_PULSE_NEAR_RAD = float(np.deg2rad(8.0))
VOICE_ROT_PULSE_MID_RAD = float(np.deg2rad(24.0))
VOICE_TASK_TIP_PITCH_RAD = float(np.deg2rad(60.0))
VOICE_TASK_TIP_ROLL_RAD = float(np.deg2rad(60.0))
VOICE_MOVE_TIMEOUT_MIN_S = 4.0
VOICE_MOVE_TIMEOUT_PER_M_S = 8.0
VOICE_ROT_TIMEOUT_MIN_S = 4.0
VOICE_ROT_TIMEOUT_PER_RAD_S = 5.0
VOICE_PROGRESS_EMIT_INTERVAL_S = 0.1
VOICE_STUCK_CHECK_INTERVAL_S = 0.1
VOICE_STUCK_GRACE_S = 0.7
VOICE_STUCK_WINDOW_S = 2.0
VOICE_MOVE_STUCK_SPEED_MPS = 0.05
VOICE_ROT_STUCK_SPEED_RAD_S = float(np.deg2rad(3.0))
DEFAULT_BACKEND_HOST = "0.0.0.0"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC = 20.0
DEFAULT_QUICK_TUNNEL_ATTEMPTS = 3
DEFAULT_QUICK_TUNNEL_PROTOCOL = "auto"
DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION = "auto"
DEFAULT_REMOTE_CONTROLLER_URL = "https://felipegalind0.github.io/gdog-remote"
DEFAULT_REMOTE_API_KEY_FILE = ".priv/api.md"
DEFAULT_REMOTE_API_KEY_QUERY_PARAM = "openai_key"


state = CommandState()


def _discover_local_ipv4_addresses():
    ips = {"127.0.0.1"}

    # Best-effort primary interface detection via UDP socket route lookup.
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        primary_ip = probe.getsockname()[0]
        if primary_ip:
            ips.add(str(primary_ip))
    except Exception:
        pass
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception:
                pass

    try:
        hostname = socket.gethostname()
        _host, _aliases, host_ips = socket.gethostbyname_ex(hostname)
        for ip in host_ips:
            if ip:
                ips.add(str(ip))
    except Exception:
        pass

    def _sort_key(ip_addr):
        if ip_addr.startswith("127."):
            return (1, ip_addr)
        return (0, ip_addr)

    return sorted(ips, key=_sort_key)


def _print_backend_endpoints(host, port):
    host_str = str(host).strip() or DEFAULT_BACKEND_HOST
    port_int = int(port)

    if host_str in ("0.0.0.0", "::"):
        addresses = _discover_local_ipv4_addresses()
    else:
        addresses = [host_str]

    print("Remote control backend endpoints:")
    print(f"  bind: {host_str}:{port_int}")
    for ip_addr in addresses:
        print(f"  backend target: {ip_addr}:{port_int}")
        print(f"    ws: ws://{ip_addr}:{port_int}/ws")
        print(f"    offer: http://{ip_addr}:{port_int}/offer")

    print("For gdog-remote backend input, use one of the 'backend target' values above.")


def _terminate_process(process, process_label):
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=3.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except Exception:
            pass


def _normalize_remote_url(remote_url):
    value = str(remote_url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    return value


def _extract_openai_api_key(raw_text):
    text = str(raw_text or "")
    match = re.search(r"(sk-[A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1).strip()
    return ""


def _load_remote_api_key(explicit_key, key_file):
    from_arg = _extract_openai_api_key(explicit_key)
    if from_arg:
        return from_arg, "--remote-api-key"

    file_path = str(key_file or "").strip()
    if not file_path:
        return "", ""

    expanded_path = os.path.expanduser(file_path)
    try:
        with open(expanded_path, "r", encoding="utf-8") as handle:
            file_contents = handle.read()
    except FileNotFoundError:
        return "", ""
    except Exception as exc:
        print(f"Failed to read remote API key file '{expanded_path}': {exc}")
        return "", ""

    from_file = _extract_openai_api_key(file_contents)
    if from_file:
        return from_file, expanded_path

    return "", ""


def _build_remote_prefilled_link(
    remote_url,
    backend_target,
    remote_api_key="",
    remote_api_key_param=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
):
    base_url = _normalize_remote_url(remote_url)
    backend_value = str(backend_target or "").strip()
    api_key_value = str(remote_api_key or "").strip()
    api_key_param = str(remote_api_key_param or "").strip() or DEFAULT_REMOTE_API_KEY_QUERY_PARAM
    if not base_url or not backend_value:
        return ""

    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        return ""

    query_map = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_map["backend"] = backend_value
    if api_key_value:
        query_map[api_key_param] = api_key_value
    new_query = urlencode(query_map, doseq=True)

    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, new_query, parts.fragment))


def _print_ascii_qr(payload):
    text = str(payload or "").strip()
    if not text:
        return False

    try:
        qrcode_mod = importlib.import_module("qrcode")
    except Exception:
        print("Terminal QR rendering unavailable (optional dependency missing: qrcode).")
        print("Install with: python -m pip install qrcode")
        qr_image_url = f"https://api.qrserver.com/v1/create-qr-code/?size=360x360&data={quote(text, safe='')}"
        print("QR image URL:")
        print(f"  {qr_image_url}")
        return False

    try:
        qr = qrcode_mod.QRCode(border=1)
        qr.add_data(text)
        qr.make(fit=True)
        print("Scan this QR with your phone camera:")
        qr.print_ascii(invert=True)
        return True
    except Exception as exc:
        print(f"Failed to render terminal QR: {exc}")
        return False


def _print_remote_controller_shortcut(
    remote_url,
    backend_target,
    print_qr=True,
    remote_api_key="",
    remote_api_key_param=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
):
    link = _build_remote_prefilled_link(
        remote_url=remote_url,
        backend_target=backend_target,
        remote_api_key=remote_api_key,
        remote_api_key_param=remote_api_key_param,
    )
    if not link:
        return ""

    if str(remote_api_key or "").strip():
        print(f"Remote controller link (backend + {remote_api_key_param} prefilled):")
    else:
        print("Remote controller link (backend prefilled):")
    print(f"  {link}")
    if bool(print_qr):
        _print_ascii_qr(link)
    return link


def _start_cloudflare_quick_tunnel(
    bind_host,
    bind_port,
    startup_timeout_sec=DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
    attempts=DEFAULT_QUICK_TUNNEL_ATTEMPTS,
    protocol=DEFAULT_QUICK_TUNNEL_PROTOCOL,
    edge_ip_version=DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
):
    cloudflared_bin = shutil.which("cloudflared")
    if cloudflared_bin is None:
        print("Cloudflare quick tunnel requested, but 'cloudflared' is not installed.")
        print("Install on macOS: brew install cloudflared")
        return None, None

    host_str = str(bind_host).strip() or DEFAULT_BACKEND_HOST
    tunnel_target_host = "127.0.0.1" if host_str in ("0.0.0.0", "::") else host_str
    tunnel_target_url = f"http://{tunnel_target_host}:{int(bind_port)}"

    cmd = [
        cloudflared_bin,
        "tunnel",
        "--url",
        tunnel_target_url,
        "--no-autoupdate",
    ]

    protocol_value = str(protocol or "").strip().lower() or DEFAULT_QUICK_TUNNEL_PROTOCOL
    if protocol_value != DEFAULT_QUICK_TUNNEL_PROTOCOL:
        cmd.extend(["--protocol", protocol_value])

    edge_ip_version_value = str(edge_ip_version or "").strip().lower() or DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION
    if edge_ip_version_value in {"4", "6"}:
        cmd.extend(["--edge-ip-version", edge_ip_version_value])

    attempts_int = max(int(attempts), 1)
    timeout_sec = max(float(startup_timeout_sec), 1.0)
    tunnel_pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.IGNORECASE)

    for attempt_idx in range(attempts_int):
        attempt_num = attempt_idx + 1
        print(
            "Starting Cloudflare quick tunnel for "
            f"{tunnel_target_url} (protocol={protocol_value}, edge_ip={edge_ip_version_value}, "
            f"attempt={attempt_num}/{attempts_int}) ..."
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(f"Failed to start cloudflared: {exc}")
            return None, None

        tunnel_url = None
        recent_output_lines = []

        def _remember_output_line(raw_line):
            line = str(raw_line or "").rstrip("\n")
            if not line:
                return
            recent_output_lines.append(line)
            if len(recent_output_lines) > 12:
                del recent_output_lines[:-12]

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if process.stdout is None:
                break

            ready, _, _ = select.select([process.stdout], [], [], 0.25)
            if not ready:
                continue

            line = process.stdout.readline()
            if not line:
                continue

            _remember_output_line(line)
            match = tunnel_pattern.search(line)
            if match:
                tunnel_url = match.group(0).rstrip("/")
                break

        if tunnel_url is not None:
            if tunnel_url.startswith("https://"):
                ws_base = f"wss://{tunnel_url[len('https://') :]}"
            else:
                ws_base = tunnel_url

            print("Cloudflare quick tunnel ready:")
            print(f"  backend target: {tunnel_url}")
            print(f"    ws: {ws_base}/ws")
            print(f"    offer: {tunnel_url}/offer")
            print("Use the 'backend target' URL above in the gdog-remote backend input.")
            return process, tunnel_url

        # Drain any remaining buffered output lines after process exit.
        if process.stdout is not None:
            while True:
                try:
                    line = process.stdout.readline()
                except Exception:
                    break
                if not line:
                    break
                _remember_output_line(line)

        if process.poll() is None:
            print("Timed out waiting for Cloudflare quick tunnel URL.")
        else:
            print(f"cloudflared exited before tunnel became ready (code {process.returncode}).")

        if recent_output_lines:
            print("Recent cloudflared output:")
            for log_line in recent_output_lines[-8:]:
                print(f"  {log_line}")

        _terminate_process(process, "cloudflared")

        if attempt_num < attempts_int:
            retry_delay = min(2.0 * attempt_num, 8.0)
            print(f"Retrying quick tunnel in {retry_delay:.1f}s...")
            time.sleep(retry_delay)
            continue

        print("You can also start tunnel manually in another terminal:")
        print(f"  {' '.join(cmd)}")
        print(
            "If this is restrictive guest Wi-Fi, retry with "
            "--quick-tunnel-protocol http2 --quick-tunnel-edge-ip-version 4."
        )
        return None, None

    return None, None


def main():
    parser = argparse.ArgumentParser(description="MVP Wheeled Robot Dog Simulator")
    parser.add_argument("--render", action="store_true", help="Enable interactive 3D viewer")
    parser.add_argument("--video", action="store_true", help="Record and save a video (mp4)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible robot and terrain randomization")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_BACKEND_HOST,
        help=f"Backend bind host (default: {DEFAULT_BACKEND_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help=f"Backend bind port (default: {DEFAULT_BACKEND_PORT})",
    )
    parser.add_argument(
        "--quick-tunnel",
        action="store_true",
        help="Start a Cloudflare Quick Tunnel for the backend and print an HTTPS URL for remote control.",
    )
    parser.add_argument(
        "--quick-tunnel-timeout",
        type=float,
        default=DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
        help=(
            "Seconds to wait for Cloudflare Quick Tunnel URL discovery "
            f"(default: {DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC})."
        ),
    )
    parser.add_argument(
        "--quick-tunnel-attempts",
        type=int,
        default=DEFAULT_QUICK_TUNNEL_ATTEMPTS,
        help=(
            "Number of Cloudflare quick tunnel startup attempts before giving up "
            f"(default: {DEFAULT_QUICK_TUNNEL_ATTEMPTS})."
        ),
    )
    parser.add_argument(
        "--quick-tunnel-protocol",
        type=str,
        choices=("auto", "http2", "quic"),
        default=DEFAULT_QUICK_TUNNEL_PROTOCOL,
        help=(
            "Transport protocol cloudflared uses to connect to Cloudflare edge "
            f"(default: {DEFAULT_QUICK_TUNNEL_PROTOCOL}). Use 'http2' on restrictive networks."
        ),
    )
    parser.add_argument(
        "--quick-tunnel-edge-ip-version",
        type=str,
        choices=("auto", "4", "6"),
        default=DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
        help=(
            "Force edge IP family for cloudflared "
            f"(default: {DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION}). Try '4' if guest Wi-Fi has broken IPv6."
        ),
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=DEFAULT_REMOTE_CONTROLLER_URL,
        help=(
            "Base URL of gdog-remote page used when printing a prefilled link/QR "
            f"(default: {DEFAULT_REMOTE_CONTROLLER_URL})."
        ),
    )
    parser.add_argument(
        "--remote-api-key",
        type=str,
        default="",
        help="OpenAI API key value to include in generated remote links as a query parameter.",
    )
    parser.add_argument(
        "--remote-api-key-file",
        type=str,
        default=DEFAULT_REMOTE_API_KEY_FILE,
        help=(
            "Path to a file containing an OpenAI API key for generated remote links "
            f"(default: {DEFAULT_REMOTE_API_KEY_FILE})."
        ),
    )
    parser.add_argument(
        "--remote-api-key-param",
        type=str,
        default=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
        help=(
            "Query parameter name used to prefill the remote API key "
            f"(default: {DEFAULT_REMOTE_API_KEY_QUERY_PARAM})."
        ),
    )
    parser.add_argument(
        "--no-qr",
        action="store_true",
        help="Disable terminal QR rendering for generated remote controller links.",
    )
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
            shadow=True,
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

    terrain_morph, terrain_info = generate_random_terrain_morph(rng)
    moon_albedo = generate_moon_albedo_texture(rng)
    moon_surface = gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(image_array=moon_albedo, encoding="srgb"),
        roughness=0.92,
        metallic=0.02,
    )
    terrain = scene.add_entity(terrain_morph, surface=moon_surface, name="moon_terrain")

    print("Configured lunar visual theme: moon terrain surface + directional sunlight")

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

    camera_controller = FollowCameraController(
        scene=scene,
        render_enabled=args.render,
        cam=cam,
        command_state=state,
        command_buffer_max=COMMAND_BUFFER_MAX,
        command_history_max=COMMAND_HISTORY_MAX,
    )
    camera_controller.update(robot)

    _print_backend_endpoints(args.host, args.port)
    print("Starting WebRTC FastAPI thread...")
    flask_thread = threading.Thread(target=run_server, args=(state, args.host, args.port), daemon=True)
    flask_thread.start()

    tunnel_process = None
    tunnel_url = ""
    remote_url = _normalize_remote_url(args.remote_url)
    remote_api_key, remote_api_key_source = _load_remote_api_key(args.remote_api_key, args.remote_api_key_file)
    remote_api_key_param = str(args.remote_api_key_param or "").strip() or DEFAULT_REMOTE_API_KEY_QUERY_PARAM

    if remote_api_key:
        print(
            "Remote controller links will include OpenAI API key query parameter "
            f"'{remote_api_key_param}' from {remote_api_key_source}."
        )

    if args.quick_tunnel:
        tunnel_process, tunnel_url = _start_cloudflare_quick_tunnel(
            bind_host=args.host,
            bind_port=args.port,
            startup_timeout_sec=args.quick_tunnel_timeout,
            attempts=args.quick_tunnel_attempts,
            protocol=args.quick_tunnel_protocol,
            edge_ip_version=args.quick_tunnel_edge_ip_version,
        )
        if tunnel_url:
            _print_remote_controller_shortcut(
                remote_url=remote_url,
                backend_target=tunnel_url,
                print_qr=not bool(args.no_qr),
                remote_api_key=remote_api_key,
                remote_api_key_param=remote_api_key_param,
            )
    elif remote_url:
        print(f"Remote controller base URL: {remote_url}")

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

    active_voice_task = None
    voice_pwm_tick = 0
    prev_pos_xy = None
    prev_pitch = None
    pitch_neutral = 0.0
    filtered_forward_speed = 0.0
    filtered_pitch_rate = 0.0
    vx_cmd_smooth = 0.0
    rotate_hold_active = False
    rotate_hold_anchor_xy = np.zeros(2, dtype=float)
    rotate_hold_integral = 0.0
    rotate_drift_err_m = 0.0
    rotate_drift_correction = 0.0
    front_unload_risk = 0.0
    traction_scale = 1.0
    front_traction_blocked = False
    forward_accel_cmd = 0.0
    forward_speed_meas = 0.0
    pitch_rate_meas = 0.0
    anti_tip_pitch_bias = 0.0

    def _emit_voice_command_result(call_id, command, status, reason="", **extra):
        call_id_text = str(call_id or "").strip()
        if not call_id_text:
            return

        payload = {
            "type": "voice_command_result",
            "call_id": call_id_text,
            "command": str(command),
            "status": str(status),
        }
        if reason:
            payload["reason"] = str(reason)
        payload.update(extra)
        state.push_outgoing(payload)

    def _emit_voice_command_progress(call_id, command, **extra):
        call_id_text = str(call_id or "").strip()
        if not call_id_text:
            return

        payload = {
            "type": "voice_command_progress",
            "call_id": call_id_text,
            "command": str(command),
        }
        payload.update(extra)
        state.push_outgoing(payload)

    def _finish_active_voice_task(task, status, reason="", **extra):
        if task is None:
            return None
        _emit_voice_command_result(
            call_id=task.get("call_id"),
            command=task.get("type", "unknown"),
            status=status,
            reason=reason,
            **extra,
        )
        return None

    try:
        i = 0
        while steps <= 0 or i < steps:
            vx, omega, pitch_cmd, roll_cmd, cam_dx, cam_dy, cam_zoom, text_cmds, voice_cmd, voice_direction, voice_amount, voice_call_id = state.get()

            remote_pitch_setpoint = float(np.clip(pitch_cmd, -1.0, 1.0)) * MAX_REMOTE_PITCH_SETPOINT_RAD
            remote_roll_setpoint = float(np.clip(roll_cmd, -1.0, 1.0)) * MAX_REMOTE_ROLL_SETPOINT_RAD

            kb_up, kb_down, kb_left, kb_right = camera_controller.get_keyboard_drive_flags()

            manual_input = (
                abs(vx) > 1e-4
                or abs(omega) > 1e-4
                or kb_up
                or kb_down
                or kb_left
                or kb_right
            )

            pos_raw = robot.get_pos()
            if hasattr(pos_raw, "cpu"):
                pos_raw = pos_raw.cpu()
            curr_pos = np.asarray(pos_raw, dtype=float).reshape(-1)

            curr_forward_xy = np.asarray(_forward_xy_from_quat_wxyz(robot.get_quat()), dtype=float)
            curr_heading = float(np.arctan2(curr_forward_xy[1], curr_forward_xy[0]))
            roll, pitch = _roll_pitch_from_quat_wxyz(robot.get_quat())

            if prev_pos_xy is None:
                prev_pos_xy = curr_pos[:2].copy()
            if prev_pitch is None:
                prev_pitch = float(pitch)
                pitch_neutral = float(pitch)

            delta_pos_xy = curr_pos[:2] - prev_pos_xy
            forward_speed_raw = float(np.dot(delta_pos_xy / sim_dt, curr_forward_xy))
            filtered_forward_speed += DRIVE_STATE_FILTER_ALPHA * (forward_speed_raw - filtered_forward_speed)
            forward_speed_meas = float(filtered_forward_speed)

            pitch_rate_raw = float((pitch - prev_pitch) / sim_dt)
            filtered_pitch_rate += DRIVE_STATE_FILTER_ALPHA * (pitch_rate_raw - filtered_pitch_rate)
            pitch_rate_meas = float(filtered_pitch_rate)

            prev_pos_xy = curr_pos[:2].copy()
            prev_pitch = float(pitch)

            received_voice_cmd = str(voice_cmd).strip().lower() if voice_cmd else ""
            if received_voice_cmd == "stop":
                if active_voice_task is not None:
                    active_voice_task = _finish_active_voice_task(
                        active_voice_task,
                        status="failed",
                        reason="Command was stopped before completion.",
                    )
                else:
                    # Always acknowledge explicit stop requests so clients can
                    # clear pending UI state even if the action just finished.
                    _emit_voice_command_result(
                        call_id=voice_call_id,
                        command="stop",
                        status="completed",
                        reason="No active command was running.",
                    )
            elif received_voice_cmd in ("move", "rotate"):
                if active_voice_task is not None:
                    active_call_id = str(active_voice_task.get("call_id") or "").strip()
                    incoming_call_id = str(voice_call_id or "").strip()

                    # Ignore duplicate start requests for the same in-flight command.
                    if not (incoming_call_id and active_call_id and incoming_call_id == active_call_id):
                        active_command = str(active_voice_task.get("type", "command"))
                        _emit_voice_command_result(
                            call_id=voice_call_id,
                            command=received_voice_cmd,
                            status="failed",
                            reason=(
                                f"Cannot start '{received_voice_cmd}' while '{active_command}' is still running. "
                                "Wait for completion or send stop first."
                            ),
                            active_command=active_command,
                        )
                else:
                    requested_dir = str(voice_direction or "").strip().lower()
                    requested_amt = max(float(voice_amount), 0.0)

                    if received_voice_cmd == "rotate" and requested_amt > (2.0 * np.pi + 0.25):
                        # Accept accidental degree payloads by auto-converting to radians.
                        requested_amt = float(np.deg2rad(requested_amt))

                    if received_voice_cmd == "move" and requested_amt > 0.0:
                        move_sign = 0.0
                        if requested_dir in ("", "forward", "fwd"):
                            move_sign = 1.0
                        elif requested_dir in ("backward", "back", "bwd", "reverse", "rev"):
                            move_sign = -1.0

                        if move_sign != 0.0:
                            timeout_s = max(
                                VOICE_MOVE_TIMEOUT_MIN_S,
                                float(requested_amt) * VOICE_MOVE_TIMEOUT_PER_M_S,
                            )
                            active_voice_task = {
                                "type": "move",
                                "dir_sign": move_sign,
                                "direction": "forward" if move_sign > 0.0 else "backward",
                                "target": requested_amt,
                                "start_pos_xy": curr_pos[:2].copy(),
                                "start_forward_xy": curr_forward_xy.copy(),
                                "call_id": voice_call_id,
                                "started_at": float(time.monotonic()),
                                "timeout_s": timeout_s,
                                "last_progress_emit_at": -1e9,
                                "last_stuck_check_progress": 0.0,
                                "last_stuck_check_time": float(time.monotonic()),
                                "stuck_time_accum": 0.0,
                                "current_speed": 0.0,
                            }
                            voice_pwm_tick = 0
                        else:
                            _emit_voice_command_result(
                                call_id=voice_call_id,
                                command="move",
                                status="failed",
                                reason="Invalid move direction. Use 'forward' or 'backward'.",
                            )

                    elif received_voice_cmd == "rotate" and requested_amt > 0.0:
                        yaw_sign = 0.0
                        if requested_dir in ("left", "l", "ccw", "counterclockwise"):
                            yaw_sign = 1.0
                        elif requested_dir in ("", "right", "r", "cw", "clockwise"):
                            yaw_sign = -1.0

                        if yaw_sign != 0.0:
                            timeout_s = max(
                                VOICE_ROT_TIMEOUT_MIN_S,
                                float(requested_amt) * VOICE_ROT_TIMEOUT_PER_RAD_S,
                            )
                            active_voice_task = {
                                "type": "rotate",
                                "dir_sign": yaw_sign,
                                "direction": "left" if yaw_sign > 0.0 else "right",
                                "target": requested_amt,
                                "progress": 0.0,
                                "prev_heading": curr_heading,
                                "call_id": voice_call_id,
                                "started_at": float(time.monotonic()),
                                "timeout_s": timeout_s,
                                "last_progress_emit_at": -1e9,
                                "last_stuck_check_progress": 0.0,
                                "last_stuck_check_time": float(time.monotonic()),
                                "stuck_time_accum": 0.0,
                                "current_speed": 0.0,
                            }
                            voice_pwm_tick = 0
                        else:
                            _emit_voice_command_result(
                                call_id=voice_call_id,
                                command="rotate",
                                status="failed",
                                reason="Invalid rotate direction. Use 'left' or 'right'.",
                            )
                    else:
                        _emit_voice_command_result(
                            call_id=voice_call_id,
                            command=received_voice_cmd,
                            status="failed",
                            reason="Invalid command amount. Value must be positive.",
                        )

            if manual_input and not received_voice_cmd:
                active_voice_task = _finish_active_voice_task(
                    active_voice_task,
                    status="failed",
                    reason="Cancelled by manual control input.",
                )

            if active_voice_task is not None:
                tipped = abs(float(pitch)) >= VOICE_TASK_TIP_PITCH_RAD or abs(float(roll)) >= VOICE_TASK_TIP_ROLL_RAD
                if tipped:
                    active_voice_task = _finish_active_voice_task(
                        active_voice_task,
                        status="failed",
                        reason=(
                            f"Robot tipped over (pitch={np.degrees(pitch):+.1f} deg, "
                            f"roll={np.degrees(roll):+.1f} deg)."
                        ),
                        pitch_deg=float(np.degrees(pitch)),
                        roll_deg=float(np.degrees(roll)),
                    )
                    vx = 0.0
                    omega = 0.0

            if kb_up or kb_down or kb_left or kb_right:
                fwd_sign = (1.0 if kb_up else 0.0) - (1.0 if kb_down else 0.0)
                yaw_sign = (1.0 if kb_right else 0.0) - (1.0 if kb_left else 0.0)
                vx = fwd_sign * KEYBOARD_VX_CMD
                omega = yaw_sign * KEYBOARD_YAW_CMD

            if active_voice_task is not None:
                elapsed_s = float(time.monotonic() - float(active_voice_task.get("started_at", 0.0)))
                timeout_s = float(active_voice_task.get("timeout_s", VOICE_MOVE_TIMEOUT_MIN_S))
                if elapsed_s > timeout_s:
                    active_voice_task = _finish_active_voice_task(
                        active_voice_task,
                        status="failed",
                        reason="Fallback safety timeout reached before completion.",
                        elapsed_s=elapsed_s,
                        timeout_s=timeout_s,
                    )
                    vx = 0.0
                    omega = 0.0

            if active_voice_task is not None:
                voice_pwm_tick += 1
                pwm_phase = voice_pwm_tick % VOICE_PWM_PERIOD_STEPS

                if active_voice_task["type"] == "move":
                    displacement_xy = curr_pos[:2] - active_voice_task["start_pos_xy"]
                    progress = float(np.dot(displacement_xy, active_voice_task["start_forward_xy"]))
                    progress *= float(active_voice_task["dir_sign"])
                    remaining = float(active_voice_task["target"]) - progress
                    lateral_axis_xy = np.array(
                        [
                            -float(active_voice_task["start_forward_xy"][1]),
                            float(active_voice_task["start_forward_xy"][0]),
                        ],
                        dtype=float,
                    )
                    lateral_error_m = abs(float(np.dot(displacement_xy, lateral_axis_xy)))

                    progress_clamped = float(np.clip(progress, 0.0, float(active_voice_task["target"])))
                    remaining_clamped = float(max(remaining, 0.0))
                    target_m = float(active_voice_task["target"])
                    now_monotonic = float(time.monotonic())

                    dt_stuck = now_monotonic - active_voice_task["last_stuck_check_time"]
                    if dt_stuck >= VOICE_STUCK_CHECK_INTERVAL_S:
                        dp = progress_clamped - active_voice_task["last_stuck_check_progress"]
                        speed = dp / max(dt_stuck, 1e-6)
                        active_voice_task["current_speed"] = float(speed)

                        should_count_as_stuck = (
                            elapsed_s >= VOICE_STUCK_GRACE_S
                            and remaining_clamped > VOICE_MOVE_STOP_TOL_M
                            and speed < VOICE_MOVE_STUCK_SPEED_MPS
                        )
                        if should_count_as_stuck:
                            active_voice_task["stuck_time_accum"] += dt_stuck
                        else:
                            active_voice_task["stuck_time_accum"] = 0.0

                        active_voice_task["last_stuck_check_progress"] = progress_clamped
                        active_voice_task["last_stuck_check_time"] = now_monotonic

                        if active_voice_task["stuck_time_accum"] > VOICE_STUCK_WINDOW_S:
                            active_voice_task = _finish_active_voice_task(
                                active_voice_task,
                                status="failed",
                                reason=(
                                    "Robot stopped making progress toward the move target "
                                    f"({speed:.2f} m/s for {VOICE_STUCK_WINDOW_S:.1f}s)."
                                ),
                                stuck_for_s=float(VOICE_STUCK_WINDOW_S),
                            )
                            vx = 0.0
                            omega = 0.0
                            continue

                    if now_monotonic - float(active_voice_task.get("last_progress_emit_at", -1e9)) >= VOICE_PROGRESS_EMIT_INTERVAL_S:
                        _emit_voice_command_progress(
                            call_id=active_voice_task.get("call_id"),
                            command="move",
                            direction=str(active_voice_task.get("direction", "forward")),
                            progress_m=progress_clamped,
                            target_m=target_m,
                            remaining_m=remaining_clamped,
                            progress_ratio=float(np.clip(progress_clamped / max(target_m, 1e-6), 0.0, 1.0)),
                            current_speed=active_voice_task.get("current_speed", 0.0),
                            lateral_error_m=float(lateral_error_m),
                            elapsed_s=float(elapsed_s),
                            timeout_s=float(active_voice_task.get("timeout_s", 0.0)),
                        )
                        active_voice_task["last_progress_emit_at"] = now_monotonic

                    if remaining <= VOICE_MOVE_STOP_TOL_M:
                        if lateral_error_m <= VOICE_MOVE_MAX_LATERAL_ERROR_M:
                            active_voice_task = _finish_active_voice_task(
                                active_voice_task,
                                status="completed",
                                progress_m=float(progress),
                                target_m=float(active_voice_task["target"]),
                                remaining_m=float(max(remaining, 0.0)),
                                lateral_error_m=float(lateral_error_m),
                            )
                        else:
                            active_voice_task = _finish_active_voice_task(
                                active_voice_task,
                                status="failed",
                                reason="Robot reached the wrong destination (path deviation too large).",
                                progress_m=float(progress),
                                target_m=float(active_voice_task["target"]),
                                remaining_m=float(max(remaining, 0.0)),
                                lateral_error_m=float(lateral_error_m),
                            )
                        vx = 0.0
                        omega = 0.0
                    else:
                        if remaining > VOICE_MOVE_PULSE_MID_M:
                            duty = 1.0
                        elif remaining > VOICE_MOVE_PULSE_NEAR_M:
                            duty = 0.6
                        else:
                            duty = 0.3
                        on_steps = max(1, int(round(VOICE_PWM_PERIOD_STEPS * duty)))
                        is_on = pwm_phase < on_steps
                        vx = float(active_voice_task["dir_sign"]) * KEYBOARD_VX_CMD if is_on else 0.0
                        omega = 0.0

                elif active_voice_task["type"] == "rotate":
                    step_delta = (curr_heading - active_voice_task["prev_heading"] + np.pi) % (2.0 * np.pi) - np.pi
                    active_voice_task["prev_heading"] = curr_heading
                    active_voice_task["progress"] += float(step_delta) * float(active_voice_task["dir_sign"])
                    remaining = float(active_voice_task["target"]) - float(active_voice_task["progress"])

                    progress_clamped_rad = float(np.clip(float(active_voice_task["progress"]), 0.0, float(active_voice_task["target"])))
                    remaining_clamped_rad = float(max(remaining, 0.0))
                    target_rad = float(active_voice_task["target"])
                    now_monotonic = float(time.monotonic())

                    dt_stuck = now_monotonic - active_voice_task["last_stuck_check_time"]
                    if dt_stuck >= VOICE_STUCK_CHECK_INTERVAL_S:
                        dp = progress_clamped_rad - active_voice_task["last_stuck_check_progress"]
                        speed = dp / max(dt_stuck, 1e-6)
                        active_voice_task["current_speed"] = float(speed)

                        should_count_as_stuck = (
                            elapsed_s >= VOICE_STUCK_GRACE_S
                            and remaining_clamped_rad > VOICE_ROT_STOP_TOL_RAD
                            and speed < VOICE_ROT_STUCK_SPEED_RAD_S
                        )
                        if should_count_as_stuck:
                            active_voice_task["stuck_time_accum"] += dt_stuck
                        else:
                            active_voice_task["stuck_time_accum"] = 0.0

                        active_voice_task["last_stuck_check_progress"] = progress_clamped_rad
                        active_voice_task["last_stuck_check_time"] = now_monotonic

                        if active_voice_task["stuck_time_accum"] > VOICE_STUCK_WINDOW_S:
                            active_voice_task = _finish_active_voice_task(
                                active_voice_task,
                                status="failed",
                                reason=(
                                    "Robot stopped making progress toward the rotate target "
                                    f"({np.degrees(speed):.1f} deg/s for {VOICE_STUCK_WINDOW_S:.1f}s)."
                                ),
                                stuck_for_s=float(VOICE_STUCK_WINDOW_S),
                            )
                            vx = 0.0
                            omega = 0.0
                            continue

                    if now_monotonic - float(active_voice_task.get("last_progress_emit_at", -1e9)) >= VOICE_PROGRESS_EMIT_INTERVAL_S:
                        _emit_voice_command_progress(
                            call_id=active_voice_task.get("call_id"),
                            command="rotate",
                            direction=str(active_voice_task.get("direction", "left")),
                            progress_rad=progress_clamped_rad,
                            target_rad=target_rad,
                            remaining_rad=remaining_clamped_rad,
                            progress_deg=float(np.degrees(progress_clamped_rad)),
                            target_deg=float(np.degrees(target_rad)),
                            remaining_deg=float(np.degrees(remaining_clamped_rad)),
                            progress_ratio=float(np.clip(progress_clamped_rad / max(target_rad, 1e-6), 0.0, 1.0)),
                            current_speed=active_voice_task.get("current_speed", 0.0),
                            elapsed_s=float(elapsed_s),
                            timeout_s=float(active_voice_task.get("timeout_s", 0.0)),
                        )
                        active_voice_task["last_progress_emit_at"] = now_monotonic

                    if remaining <= VOICE_ROT_STOP_TOL_RAD:
                        active_voice_task = _finish_active_voice_task(
                            active_voice_task,
                            status="completed",
                            progress_rad=float(active_voice_task["progress"]),
                            target_rad=float(active_voice_task["target"]),
                            remaining_rad=float(max(remaining, 0.0)),
                        )
                        vx = 0.0
                        omega = 0.0
                    else:
                        if remaining > VOICE_ROT_PULSE_MID_RAD:
                            duty = 1.0
                        elif remaining > VOICE_ROT_PULSE_NEAR_RAD:
                            duty = 0.6
                        else:
                            duty = 0.3
                        on_steps = max(1, int(round(VOICE_PWM_PERIOD_STEPS * duty)))
                        is_on = pwm_phase < on_steps
                        vx = 0.0
                        omega = -float(active_voice_task["dir_sign"]) * KEYBOARD_YAW_CMD if is_on else 0.0

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
                active_voice_task = _finish_active_voice_task(
                    active_voice_task,
                    status="failed",
                    reason="Robot was respawned before command completion.",
                )
                robot.set_qpos(spawn_qs.copy())
                pitch_pid.reset()
                roll_pid.reset()
                respawn_settle_steps = RESPAWN_SETTLE_STEPS
                state.update(0.0, 0.0)
                vx = 0.0
                omega = 0.0
                vx_cmd_smooth = 0.0
                rotate_hold_active = False
                rotate_hold_integral = 0.0
                rotate_drift_err_m = 0.0
                rotate_drift_correction = 0.0
                front_unload_risk = 0.0
                traction_scale = 1.0
                front_traction_blocked = False
                forward_accel_cmd = 0.0
                anti_tip_pitch_bias = 0.0
                prev_pos_xy = np.asarray(spawn_qs[0:2], dtype=float)
                prev_pitch = 0.0
                pitch_neutral = 0.0
                filtered_forward_speed = 0.0
                filtered_pitch_rate = 0.0
                roll = 0.0
                pitch = 0.0

            vx_cmd_raw = float(vx)
            omega_cmd_raw = float(omega)

            rotate_only_requested = (
                abs(omega_cmd_raw) > ROTATE_ONLY_OMEGA_MIN
                and abs(vx_cmd_raw) < ROTATE_ONLY_VX_DEADBAND
            )

            if rotate_only_requested:
                if not rotate_hold_active:
                    rotate_hold_active = True
                    rotate_hold_anchor_xy = curr_pos[:2].copy()
                    rotate_hold_integral = 0.0

                rotate_drift_err_m = float(np.dot(curr_pos[:2] - rotate_hold_anchor_xy, curr_forward_xy))
                rotate_hold_integral = float(
                    np.clip(
                        rotate_hold_integral + rotate_drift_err_m * sim_dt,
                        -ROTATE_DRIFT_I_LIMIT,
                        ROTATE_DRIFT_I_LIMIT,
                    )
                )
                rotate_drift_correction = float(
                    np.clip(
                        -(
                            ROTATE_DRIFT_KP * rotate_drift_err_m
                            + ROTATE_DRIFT_KI * rotate_hold_integral
                            + ROTATE_DRIFT_KD * forward_speed_meas
                        ),
                        -ROTATE_DRIFT_MAX_CORRECTION,
                        ROTATE_DRIFT_MAX_CORRECTION,
                    )
                )
            else:
                rotate_hold_active = False
                rotate_hold_anchor_xy = curr_pos[:2].copy()
                rotate_hold_integral = 0.0
                rotate_drift_err_m = 0.0
                rotate_drift_correction = 0.0

            vx_cmd_after_rotate = float(vx_cmd_raw + rotate_drift_correction)

            prev_vx_cmd_smooth = float(vx_cmd_smooth)
            vx_target_delta = vx_cmd_after_rotate - prev_vx_cmd_smooth
            if np.sign(vx_cmd_after_rotate) == np.sign(prev_vx_cmd_smooth):
                if abs(vx_cmd_after_rotate) > abs(prev_vx_cmd_smooth):
                    max_vx_step = DRIVE_VX_ACCEL_LIMIT * sim_dt
                else:
                    max_vx_step = DRIVE_VX_DECEL_LIMIT * sim_dt
            else:
                max_vx_step = DRIVE_VX_DECEL_LIMIT * sim_dt

            vx_cmd_smooth = prev_vx_cmd_smooth + float(np.clip(vx_target_delta, -max_vx_step, max_vx_step))
            forward_accel_cmd = float((vx_cmd_smooth - prev_vx_cmd_smooth) / sim_dt)

            if abs(forward_speed_meas) < TRACTION_NEUTRAL_SPEED_MPS and abs(vx_cmd_smooth) < TRACTION_NEUTRAL_CMD:
                pitch_neutral = float(
                    (1.0 - TRACTION_NEUTRAL_PITCH_ALPHA) * pitch_neutral
                    + TRACTION_NEUTRAL_PITCH_ALPHA * pitch
                )
            dynamic_pitch = float(pitch - pitch_neutral)

            pitch_risk = float(
                np.clip(
                    (abs(dynamic_pitch) - TRACTION_PITCH_WARN_RAD)
                    / max(TRACTION_PITCH_BLOCK_RAD - TRACTION_PITCH_WARN_RAD, 1e-6),
                    0.0,
                    1.0,
                )
            )
            pitch_rate_risk = float(
                np.clip(
                    (abs(pitch_rate_meas) - TRACTION_PITCH_RATE_WARN_RAD_S)
                    / max(TRACTION_PITCH_RATE_BLOCK_RAD_S - TRACTION_PITCH_RATE_WARN_RAD_S, 1e-6),
                    0.0,
                    1.0,
                )
            )
            cmd_accel_risk = float(
                np.clip(
                    (abs(forward_accel_cmd) - TRACTION_CMD_ACCEL_WARN)
                    / max(TRACTION_CMD_ACCEL_BLOCK - TRACTION_CMD_ACCEL_WARN, 1e-6),
                    0.0,
                    1.0,
                )
            )

            moving_or_driving = (
                abs(forward_speed_meas) > TRACTION_NEUTRAL_SPEED_MPS
                or abs(vx_cmd_smooth) > TRACTION_NEUTRAL_CMD
            )
            if moving_or_driving:
                front_unload_risk = float(0.55 * pitch_risk + 0.25 * pitch_rate_risk + 0.20 * cmd_accel_risk)
            else:
                front_unload_risk = 0.0

            traction_scale = 1.0
            if front_unload_risk > TRACTION_RISK_START:
                traction_scale = float(
                    np.clip(
                        1.0 - (front_unload_risk - TRACTION_RISK_START) / max(1.0 - TRACTION_RISK_START, 1e-6),
                        TRACTION_MIN_SCALE,
                        1.0,
                    )
                )

            front_traction_blocked = bool(
                front_unload_risk >= TRACTION_RISK_HARD_BLOCK
                and abs(vx_cmd_smooth) > TRACTION_NEUTRAL_CMD
            )
            vx_cmd_limited = float(vx_cmd_smooth * traction_scale)
            if front_traction_blocked:
                vx_cmd_limited = 0.0

            balance_gain = float(np.clip(abs(forward_accel_cmd) / ANTI_TIP_BALANCE_ACCEL_REF, 0.0, 1.0))
            balance_gain = max(balance_gain, front_unload_risk)
            anti_tip_pitch_bias = -(
                ANTI_TIP_PITCH_DAMP_GAIN * pitch_rate_meas
                + ANTI_TIP_PITCH_RESTORE_GAIN * dynamic_pitch
            )
            anti_tip_pitch_bias *= balance_gain * (1.0 + ANTI_TIP_RISK_BOOST_GAIN * front_unload_risk)
            anti_tip_pitch_bias = float(
                np.clip(
                    anti_tip_pitch_bias,
                    -ANTI_TIP_MAX_SETPOINT_BIAS_RAD,
                    ANTI_TIP_MAX_SETPOINT_BIAS_RAD,
                )
            )

            vx = vx_cmd_limited
            omega = omega_cmd_raw

            pitch_pid.setpoint = float(
                np.clip(
                    remote_pitch_setpoint + anti_tip_pitch_bias,
                    -MAX_REMOTE_PITCH_SETPOINT_RAD,
                    MAX_REMOTE_PITCH_SETPOINT_RAD,
                )
            )
            roll_pid.setpoint = remote_roll_setpoint

            # 1. PID roll/pitch stabilization outputs desired per-side Z adjustments.
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

                # 2. Mix pitch and roll corrections into each leg target.
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

            # 3. Apply leg position control from the suspension IK.
            robot.control_dofs_position(target_leg_pos, dofs_idx_local=leg_dofs)
        
            # 4. Calculate skid-steer kinematics:
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

            # 5. Drive the wheels with velocity control
            robot.control_dofs_velocity(target_wheel_vel, dofs_idx_local=wheel_dofs)

            hud_status_text = (
                f"Susp {'ON' if suspension_enabled else 'OFF'}\n"
                f"P {np.degrees(pitch):+.1f}deg  R {np.degrees(roll):+.1f}deg"
            )
            if front_traction_blocked:
                hud_status_text += "\nFront traction: BLOCK"
            elif traction_scale < 0.999:
                hud_status_text += f"\nFront traction: {traction_scale * 100.0:.0f}%"
            if camera_controller.is_ctrl_held():
                hud_status_text += (
                    f"\nCam target z {camera_controller.camera_center_height_offset:+.3f} m"
                )
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
                        f" yawTotal {np.degrees(camera_controller.get_sim_yaw_total()):+.2f} deg",
                    ]
                )
            if debug_speed_enabled:
                if debug_lines:
                    debug_lines.append("")
                debug_lines.extend(
                    [
                        "[Speed Debug]",
                        f" vx_raw {vx_cmd_raw:+.3f}",
                        f" vx_smooth {vx_cmd_smooth:+.3f}",
                        f" vx_cmd {vx:+.3f} m/s",
                        f" omega_cmd {omega:+.3f} rad/s",
                        f" v_forward {forward_speed_meas:+.3f} m/s",
                        f" pitch_rate {np.degrees(pitch_rate_meas):+.1f} deg/s",
                        f" accel_cmd {forward_accel_cmd:+.2f} cmd/s",
                        f" rot_drift {rotate_drift_err_m:+.3f} m",
                        f" rot_corr {rotate_drift_correction:+.3f}",
                        f" traction_risk {front_unload_risk:.2f}",
                        f" traction_scale {traction_scale:.2f}",
                        f" traction_block {'YES' if front_traction_blocked else 'no'}",
                        f" anti_tip_bias {np.degrees(anti_tip_pitch_bias):+.2f} deg",
                        f" wheel_left {left_vel:+.3f}",
                        f" wheel_right {right_vel:+.3f}",
                    ]
                )
            camera_controller.set_hud(hud_status_text, debug_lines)

            try:
                scene.step()
            except Exception as exc:
                if args.render and "Viewer closed." in str(exc):
                    print("Viewer closed. Exiting simulation loop.")
                    break
                raise

            camera_controller.update(robot, cam_dx=cam_dx, cam_dy=cam_dy, cam_zoom=cam_zoom)

            if cam and i % 2 == 0:
                cam.render()

            i += 1
    finally:
        if args.video and cam:
            cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
            print("Video saved to wheeled_go2.mp4")

        if tunnel_process is not None:
            print("Stopping Cloudflare quick tunnel...")
            _terminate_process(tunnel_process, "cloudflared")

if __name__ == "__main__":
    main()
