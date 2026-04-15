import argparse
import os
import threading

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

from bone_props import check_bone_respawn, spawn_bone_entity
from camera_controller import FollowCameraController
from commands import CommandState, TuneRegistry
from config import (
    COMMAND_BUFFER_MAX,
    COMMAND_HISTORY_MAX,
    DEFAULT_BACKEND_HOST,
    DEFAULT_BACKEND_PORT,
    DEFAULT_QUICK_TUNNEL_ATTEMPTS,
    DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
    DEFAULT_QUICK_TUNNEL_PROTOCOL,
    DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
    DEFAULT_REMOTE_API_KEY_FILE,
    DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
    DEFAULT_REMOTE_CONTROLLER_URL,
    DRIVE_ACCEL_RESPONSE_GAIN,
    DRIVE_BRAKE_RESPONSE_GAIN,
    DRIVE_INPUT_OMEGA_GAIN,
    DRIVE_INPUT_VX_GAIN,
    KEYBOARD_VX_CMD,
    KEYBOARD_YAW_CMD,
    MAX_REMOTE_PITCH_SETPOINT_RAD,
    MAX_REMOTE_ROLL_SETPOINT_RAD,
    RESPAWN_SETTLE_STEPS,
    SIM_DT,
)
from longitudinal_control import LongitudinalController
from math_utils import _forward_xy_from_quat_wxyz, _roll_pitch_from_quat_wxyz
from network import run_server
from procedural_gen import (
    generate_moon_albedo_texture,
    generate_random_robot_urdf,
    generate_random_terrain_morph,
)
from robot_model import build_spawn_qpos, compute_nominal_leg_z, discover_robot_joints
from runtime_services import (
    collect_runtime_system_info,
    load_remote_api_key,
    normalize_remote_url,
    print_backend_endpoints,
    print_remote_controller_shortcut,
    render_neofetch_banner,
    start_cloudflare_quick_tunnel,
    terminate_process,
)
from suspension_control import SuspensionController
from voice_tasks import VoiceTaskManager

state = CommandState()


def main():
    parser = argparse.ArgumentParser(description="MVP Wheeled Robot Dog Simulator")
    parser.add_argument("--render", action="store_true", help="Enable interactive 3D viewer")
    parser.add_argument("--video", action="store_true", help="Record and save a video (mp4)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible robot and terrain randomization")
    parser.add_argument(
        "--spawn-bone", action="store_true",
        help="Spawn a random bone prop somewhere on the terrain at startup.",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_BACKEND_HOST, help=f"Backend bind host (default: {DEFAULT_BACKEND_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_BACKEND_PORT, help=f"Backend bind port (default: {DEFAULT_BACKEND_PORT})")
    parser.add_argument("--quick-tunnel", action="store_true", help="Start a Cloudflare Quick Tunnel for the backend and print an HTTPS URL for remote control.")
    parser.add_argument(
        "--quick-tunnel-timeout", type=float, default=DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
        help=f"Seconds to wait for Cloudflare Quick Tunnel URL discovery (default: {DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC}).",
    )
    parser.add_argument(
        "--quick-tunnel-attempts", type=int, default=DEFAULT_QUICK_TUNNEL_ATTEMPTS,
        help=f"Number of Cloudflare quick tunnel startup attempts before giving up (default: {DEFAULT_QUICK_TUNNEL_ATTEMPTS}).",
    )
    parser.add_argument(
        "--quick-tunnel-protocol", type=str, choices=("auto", "http2", "quic"), default=DEFAULT_QUICK_TUNNEL_PROTOCOL,
        help=f"Transport protocol cloudflared uses to connect to Cloudflare edge (default: {DEFAULT_QUICK_TUNNEL_PROTOCOL}). Use 'http2' on restrictive networks.",
    )
    parser.add_argument(
        "--quick-tunnel-edge-ip-version", type=str, choices=("auto", "4", "6"), default=DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
        help=f"Force edge IP family for cloudflared (default: {DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION}). Try '4' if guest Wi-Fi has broken IPv6.",
    )
    parser.add_argument(
        "--remote-url", type=str, default=DEFAULT_REMOTE_CONTROLLER_URL,
        help=f"Base URL of gdog-remote page used when printing a prefilled link/QR (default: {DEFAULT_REMOTE_CONTROLLER_URL}).",
    )
    parser.add_argument("--remote-api-key", type=str, default="", help="OpenAI API key value to include in generated remote links as a query parameter.")
    parser.add_argument(
        "--remote-api-key-file", type=str, default=DEFAULT_REMOTE_API_KEY_FILE,
        help=f"Path to a file containing an OpenAI API key for generated remote links (default: {DEFAULT_REMOTE_API_KEY_FILE}).",
    )
    parser.add_argument(
        "--remote-api-key-param", type=str, default=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
        help=f"Query parameter name used to prefill the remote API key (default: {DEFAULT_REMOTE_API_KEY_QUERY_PARAM}).",
    )
    parser.add_argument("--no-qr", action="store_true", help="Disable terminal QR rendering for generated remote controller links.")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override simulation steps. Default: video=500, headless=10000, render=unlimited. Use 0 or negative for unlimited.",
    )
    args = parser.parse_args()

    # ── Neofetch banner & system info ──
    print("Rendering neofetch banner...", flush=True)
    banner_ok, banner_error = render_neofetch_banner()
    if not banner_ok:
        print(f"Could not render neofetch banner: {banner_error}")

    print("Collecting runtime system info using neofetch...", flush=True)
    runtime_system_info = collect_runtime_system_info()
    if runtime_system_info.get("ok"):
        print("Captured neofetch output for realtime session context.")
    else:
        print(f"Could not capture neofetch output: {runtime_system_info.get('error', 'unknown error')}")

    # ── Seed & Genesis init ──
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

    # ── Scene setup ──
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
        ),
    )

    # ── Terrain ──
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

    # ── Bone prop ──
    bone_entity = None
    bone_urdf_path = None
    if args.spawn_bone:
        bone_entity, bone_urdf_path = spawn_bone_entity(scene, rng, terrain_info, gs)

    # ── Robot ──
    urdf_path, robot_params = generate_random_robot_urdf(rng)
    robot = scene.add_entity(
        gs.morphs.URDF(file=urdf_path, pos=(0, 0, 0.5)),
    )

    cam = None
    if args.video:
        cam = scene.add_camera(res=(1920, 1080), pos=(1.5, -1.5, 1.0), lookat=(0, 0, 0.3), fov=40, GUI=False)

    scene.build()

    # Clean up temp URDFs now that Genesis has parsed them.
    os.remove(urdf_path)
    if bone_urdf_path is not None:
        os.remove(bone_urdf_path)

    # ── Joint discovery ──
    joints = discover_robot_joints(robot)
    leg_dofs = joints["leg_dofs"]
    wheel_dofs = joints["wheel_dofs"]
    wheel_names = joints["wheel_names"]
    standing_dof_pos = joints["standing_dof_pos"]
    standing_leg_pos = joints["standing_leg_pos"]
    leg_joint_dofs = joints["leg_joint_dofs"]
    leg_dof_to_local_idx = joints["leg_dof_to_local_idx"]

    thigh_length = float(robot_params["thigh_length"])
    calf_length = float(robot_params["calf_length"])
    nominal_leg_z = compute_nominal_leg_z(thigh_length, calf_length)

    print(f"Suspension IK parameters: thigh_length={thigh_length:.3f} m, calf_length={calf_length:.3f} m")

    # ── Controllers ──
    suspension = SuspensionController(
        leg_joint_dofs=leg_joint_dofs,
        leg_dof_to_local_idx=leg_dof_to_local_idx,
        standing_leg_pos=standing_leg_pos,
        nominal_leg_z=nominal_leg_z,
        thigh_length=thigh_length,
        calf_length=calf_length,
    )
    longitudinal = LongitudinalController()
    voice = VoiceTaskManager(state)

    if args.render:
        print("Suspension command console:")
        print("  Press '/' or 'T' to open the input line, type command, Enter to run, Esc to cancel.")
        print("  Examples: kp=0.08  ki=0.00  kd=0.02  rp=0.04  p_sign=-1  debug_pitch=on  debug_roll=on  susp=off  respawn  status  help")

    # ── Spawn pose ──
    spawn_qs = build_spawn_qpos(robot, standing_dof_pos)
    robot.set_qpos(spawn_qs)
    respawn_settle_steps = 0

    if args.video and cam:
        cam.start_recording()

    # ── Camera controller ──
    camera_controller = FollowCameraController(
        scene=scene,
        render_enabled=args.render,
        cam=cam,
        command_state=state,
        command_buffer_max=COMMAND_BUFFER_MAX,
        command_history_max=COMMAND_HISTORY_MAX,
    )
    camera_controller.update(robot)

    # ── Network server ──
    print_backend_endpoints(args.host, args.port)
    print("Starting WebRTC FastAPI thread...")

    tune_registry = TuneRegistry()

    flask_thread = threading.Thread(
        target=run_server,
        kwargs={
            "state": state,
            "host": args.host,
            "port": args.port,
            "runtime_info": runtime_system_info,
            "tune_registry": tune_registry,
        },
        daemon=True,
    )
    flask_thread.start()

    # ── Cloudflare tunnel ──
    tunnel_process = None
    tunnel_url = ""
    remote_url = normalize_remote_url(args.remote_url)
    remote_api_key, remote_api_key_source = load_remote_api_key(args.remote_api_key, args.remote_api_key_file)
    remote_api_key_param = str(args.remote_api_key_param or "").strip() or DEFAULT_REMOTE_API_KEY_QUERY_PARAM

    if remote_api_key:
        print(
            "Remote controller links will include OpenAI API key query parameter "
            f"'{remote_api_key_param}' from {remote_api_key_source}."
        )

    if args.quick_tunnel:
        tunnel_process, tunnel_url = start_cloudflare_quick_tunnel(
            bind_host=args.host,
            bind_port=args.port,
            startup_timeout_sec=args.quick_tunnel_timeout,
            attempts=args.quick_tunnel_attempts,
            protocol=args.quick_tunnel_protocol,
            edge_ip_version=args.quick_tunnel_edge_ip_version,
        )
        if tunnel_url:
            print_remote_controller_shortcut(
                remote_url=remote_url,
                backend_target=tunnel_url,
                print_qr=not bool(args.no_qr),
                remote_api_key=remote_api_key,
                remote_api_key_param=remote_api_key_param,
            )
    elif remote_url:
        print(f"Remote controller base URL: {remote_url}")

    # ── Simulation loop ──
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

    bone_respawn_cooldown_steps = 0

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
                or kb_up or kb_down or kb_left or kb_right
            )

            # ── Robot state ──
            pos_raw = robot.get_pos()
            if hasattr(pos_raw, "cpu"):
                pos_raw = pos_raw.cpu()
            curr_pos = np.asarray(pos_raw, dtype=float).reshape(-1)
            curr_forward_xy = np.asarray(_forward_xy_from_quat_wxyz(robot.get_quat()), dtype=float)
            curr_heading = float(np.arctan2(curr_forward_xy[1], curr_forward_xy[0]))
            roll, pitch = _roll_pitch_from_quat_wxyz(robot.get_quat())

            # ── State estimation ──
            longitudinal.estimate_state(curr_pos, curr_forward_xy, curr_heading, pitch, sim_dt)

            # ── Voice commands ──
            voice_cmd_received = voice.dispatch(voice_cmd, voice_direction, voice_amount, voice_call_id,
                                                 curr_pos, curr_forward_xy, curr_heading) if voice_cmd else False

            cancel_vx, cancel_omega, cancel_applied = voice.check_cancellations(manual_input, pitch, roll, voice_cmd_received)
            if cancel_applied:
                vx, omega = cancel_vx, cancel_omega

            if kb_up or kb_down or kb_left or kb_right:
                fwd_sign = (1.0 if kb_up else 0.0) - (1.0 if kb_down else 0.0)
                yaw_sign = (1.0 if kb_right else 0.0) - (1.0 if kb_left else 0.0)
                vx = fwd_sign * KEYBOARD_VX_CMD
                omega = yaw_sign * KEYBOARD_YAW_CMD

            timeout_vx, timeout_omega, timeout_applied = voice.check_timeout()
            if timeout_applied:
                vx, omega = timeout_vx, timeout_omega

            tick_vx, tick_omega, tick_skip = voice.tick(curr_pos, curr_forward_xy, curr_heading)
            if tick_skip:
                vx, omega = tick_vx, tick_omega
                continue
            if tick_vx is not None:
                vx, omega = tick_vx, tick_omega

            # ── Suspension commands ──
            do_respawn = suspension.process_commands(text_cmds)
            snap = suspension.get_state_snapshot()

            if do_respawn:
                voice.cancel("Robot was respawned before command completion.")
                robot.set_qpos(spawn_qs.copy())
                suspension.reset_pids()
                respawn_settle_steps = RESPAWN_SETTLE_STEPS
                state.update(0.0, 0.0)
                vx, omega = 0.0, 0.0
                longitudinal.reset_for_respawn(spawn_qs[0:2])
                roll, pitch = 0.0, 0.0

            # ── Longitudinal control ──
            vx_limited, omega_cmd, vx_cmd_raw = longitudinal.update(vx, omega, pitch, curr_pos[:2], curr_forward_xy, sim_dt)
            vx = vx_limited
            omega = omega_cmd

            # ── Suspension IK ──
            if respawn_settle_steps > 0:
                respawn_settle_steps -= 1
                respawn_settling = True
            else:
                respawn_settling = False

            target_leg_pos, pitch_delta_z, roll_delta_z = suspension.update(
                pitch=pitch,
                roll=roll,
                sim_dt=sim_dt,
                stance_shift_leg_x=longitudinal.stance_shift_leg_x,
                remote_pitch_setpoint=remote_pitch_setpoint,
                remote_roll_setpoint=remote_roll_setpoint,
                anti_tip_pitch_bias=longitudinal.anti_tip_pitch_bias,
                respawn_settling=respawn_settling,
            )

            # ── Apply leg position control ──
            robot.control_dofs_position(target_leg_pos, dofs_idx_local=leg_dofs)

            # ── Skid-steer kinematics ──
            left_vel = vx - omega
            right_vel = vx + omega

            target_wheel_vel = np.zeros(len(wheel_dofs))
            for j, joint_name in enumerate(wheel_names):
                wheel_prefix = joint_name.split("_", 1)[0].lower()
                if wheel_prefix in ("fl", "rl"):
                    target_wheel_vel[j] = left_vel
                elif wheel_prefix in ("fr", "rr"):
                    target_wheel_vel[j] = right_vel
                else:
                    target_wheel_vel[j] = vx

            robot.control_dofs_velocity(target_wheel_vel, dofs_idx_local=wheel_dofs)

            # ── HUD ──
            suspension_enabled = snap["enabled"]
            hud_status_text = (
                f"Susp {'ON' if suspension_enabled else 'OFF'}\n"
                f"P {np.degrees(pitch):+.1f}deg  R {np.degrees(roll):+.1f}deg"
            )
            if longitudinal.front_traction_blocked:
                hud_status_text += "\nFront traction: BLOCK"
            elif longitudinal.traction_scale < 0.999:
                hud_status_text += f"\nFront traction: {longitudinal.traction_scale * 100.0:.0f}%"
            if camera_controller.is_ctrl_held():
                hud_status_text += f"\nCam target z {camera_controller.camera_center_height_offset:+.3f} m"

            debug_lines = _build_debug_lines(snap, suspension, longitudinal, pitch, roll, pitch_delta_z, roll_delta_z,
                                              vx_cmd_raw, vx, omega, left_vel, right_vel, camera_controller, robot)
            camera_controller.set_hud(hud_status_text, debug_lines)

            # ── Physics step ──
            try:
                scene.step()
            except Exception as exc:
                if args.render and "Viewer closed." in str(exc):
                    print("Viewer closed. Exiting simulation loop.")
                    break
                raise

            # ── Bone respawn ──
            if bone_entity is not None:
                bone_respawn_cooldown_steps = check_bone_respawn(
                    bone_entity, robot, rng, terrain_info, bone_respawn_cooldown_steps,
                )

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
            terminate_process(tunnel_process, "cloudflared")


def _build_debug_lines(snap, suspension, longitudinal, pitch, roll,
                        pitch_delta_z, roll_delta_z, vx_cmd_raw, vx, omega,
                        left_vel, right_vel, camera_controller, robot):
    debug_lines = []
    lc = longitudinal

    if snap["debug_pitch"]:
        pp = suspension.pitch_pid
        pitch_clip = "YES" if pp.last_was_clipped else "no"
        debug_lines.extend([
            "[Pitch PID]",
            f" Kp {pp.kp:.3f}", f" Ki {pp.ki:.3f}", f" Kd {pp.kd:.3f}",
            f" angle {np.degrees(pitch):+.2f} deg",
            f" err {pp.last_error:+.4f} rad",
            f" P {pp.last_p_term:+.4f}", f" I {pp.last_i_term:+.4f}", f" D {pp.last_d_term:+.4f}",
            f" out {pp.last_output:+.4f}", f" raw {pp.last_output_unclipped:+.4f}",
            f" clip {pitch_clip}", f" dz {pitch_delta_z:+.3f} m",
        ])

    if snap["debug_roll"]:
        rp = suspension.roll_pid
        roll_clip = "YES" if rp.last_was_clipped else "no"
        if debug_lines:
            debug_lines.append("")
        debug_lines.extend([
            "[Roll PID]",
            f" Kp {rp.kp:.3f}", f" Ki {rp.ki:.3f}", f" Kd {rp.kd:.3f}",
            f" angle {np.degrees(roll):+.2f} deg",
            f" err {rp.last_error:+.4f} rad",
            f" P {rp.last_p_term:+.4f}", f" I {rp.last_i_term:+.4f}", f" D {rp.last_d_term:+.4f}",
            f" out {rp.last_output:+.4f}", f" raw {rp.last_output_unclipped:+.4f}",
            f" clip {roll_clip}", f" dz {roll_delta_z:+.3f} m",
        ])

    if snap["debug_yaw"]:
        forward_xy_debug = _forward_xy_from_quat_wxyz(robot.get_quat())
        heading_debug = np.arctan2(forward_xy_debug[1], forward_xy_debug[0])
        if debug_lines:
            debug_lines.append("")
        debug_lines.extend([
            "[Yaw Debug]",
            f" omega_cmd {omega:+.3f} rad/s",
            f" heading {np.degrees(heading_debug):+.2f} deg",
            f" yawTotal {np.degrees(camera_controller.get_sim_yaw_total()):+.2f} deg",
        ])

    if snap["debug_speed"]:
        if debug_lines:
            debug_lines.append("")
        debug_lines.extend([
            "[Speed Debug]",
            f" vx_raw {vx_cmd_raw:+.3f}",
            f" speed_ratio {lc.drive_speed_ratio:.2f}",
            f" power_scale {lc.drive_power_scale:.2f}x",
            f" vx_cap {lc.vx_power_cap:+.3f}",
            f" omega_cap {lc.omega_power_cap:+.3f}",
            f" vx_smooth {lc.vx_cmd_smooth:+.3f}",
            f" vx_cmd {vx:+.3f} m/s",
            f" omega_cmd {omega:+.3f} rad/s",
            f" v_forward {lc.forward_speed_meas:+.3f} m/s",
            f" pitch_rate {np.degrees(lc.pitch_rate_meas):+.1f} deg/s",
            f" yaw_rate {np.degrees(lc.yaw_rate_meas):+.1f} deg/s",
            f" accel_lim {lc.drive_accel_limit:.1f} cmd/s",
            f" brake_lim {lc.drive_brake_limit * lc.predictive_brake_scale * lc.brake_throttle:.1f} cmd/s",
            f" brake_ff {lc.predictive_brake_scale:.2f}x",
            f" brake_throttle {lc.brake_throttle:.2f}",
            f" vx_norm {lc.vx_cmd_norm:+.2f}",
            f" accel_des {lc.desired_accel_target:+.2f} m/s^2",
            f" tilt_des {np.degrees(lc.desired_tilt_target):+.1f} deg",
            f" accel_cmd {lc.forward_accel_cmd:+.2f} cmd/s",
            f" rot_drift {lc.rotate_drift_err_m:+.3f} m",
            f" rot_corr {lc.rotate_drift_correction:+.3f}",
            f" traction_risk {lc.front_unload_risk:.2f}",
            f" traction_scale {lc.traction_scale:.2f}",
            f" traction_block {'YES' if lc.front_traction_blocked else 'no'}",
            f" cmd_gain_vx {DRIVE_INPUT_VX_GAIN:.2f}x",
            f" cmd_gain_omega {DRIVE_INPUT_OMEGA_GAIN:.2f}x",
            f" accel_gain {DRIVE_ACCEL_RESPONSE_GAIN:.2f}x",
            f" brake_gain {DRIVE_BRAKE_RESPONSE_GAIN:.2f}x",
            f" anti_tip_bias {np.degrees(lc.anti_tip_pitch_bias):+.2f} deg",
            f" stance_target_x {lc.stance_shift_target_x:+.3f} m",
            f" stance_leg_x {lc.stance_shift_leg_x:+.3f} m",
            f" wheel_left {left_vel:+.3f}",
            f" wheel_right {right_vel:+.3f}",
        ])

    return debug_lines


if __name__ == "__main__":
    main()
