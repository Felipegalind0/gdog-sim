import argparse
import os
import threading

import genesis as gs
import numpy as np

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
KEYBOARD_YAW_CMD = 6.0
MAX_REMOTE_PITCH_SETPOINT_RAD = float(np.deg2rad(12.0))
MAX_REMOTE_ROLL_SETPOINT_RAD = float(np.deg2rad(12.0))


state = CommandState()


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

    print("Starting WebRTC FastAPI thread...")
    flask_thread = threading.Thread(target=run_server, args=(state,), daemon=True)
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

        kb_up, kb_down, kb_left, kb_right = camera_controller.get_keyboard_drive_flags()

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
                    f" vx_cmd {vx:+.3f} m/s",
                    f" omega_cmd {omega:+.3f} rad/s",
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

    if args.video and cam:
        cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
        print("Video saved to wheeled_go2.mp4")

if __name__ == "__main__":
    main()
