import threading

import numpy as np

from commands import _execute_suspension_command
from config import (
    MAX_LEG_DELTA_Z,
    MAX_PID_Z_ADJUST,
    MAX_REMOTE_PITCH_SETPOINT_RAD,
    MAX_REMOTE_ROLL_SETPOINT_RAD,
    PITCH_KD,
    PITCH_KI,
    PITCH_KP,
    PITCH_MIX_SIGN_DEFAULT,
    PITCH_SETPOINT,
    ROLL_KD,
    ROLL_KI,
    ROLL_KP,
    ROLL_MIX_SIGN_DEFAULT,
    ROLL_SETPOINT,
    STANCE_SHIFT_IK_MIN_X_M,
)
from math_utils import _ik_two_link_for_vertical_position
from pid import PIDController


class SuspensionController:
    def __init__(self, leg_joint_dofs, leg_dof_to_local_idx, standing_leg_pos,
                 nominal_leg_z, thigh_length, calf_length):
        self.leg_joint_dofs = leg_joint_dofs
        self.leg_dof_to_local_idx = leg_dof_to_local_idx
        self.standing_leg_pos = standing_leg_pos
        self.nominal_leg_z = nominal_leg_z
        self.thigh_length = thigh_length
        self.calf_length = calf_length

        self.pitch_pid = PIDController(
            kp=PITCH_KP, ki=PITCH_KI, kd=PITCH_KD,
            setpoint=PITCH_SETPOINT, output_limit=MAX_PID_Z_ADJUST,
        )
        self.roll_pid = PIDController(
            kp=ROLL_KP, ki=ROLL_KI, kd=ROLL_KD,
            setpoint=ROLL_SETPOINT, output_limit=MAX_PID_Z_ADJUST,
        )

        self.lock = threading.RLock()
        self.tune = {
            "enabled": True,
            "pitch_mix_sign": float(PITCH_MIX_SIGN_DEFAULT),
            "roll_mix_sign": float(ROLL_MIX_SIGN_DEFAULT),
        }
        self.debug_flags = {
            "debug_pitch": True,
            "debug_roll": False,
            "debug_yaw": False,
            "debug_speed": False,
        }
        self.pending_actions = {"respawn": False}

    def reset_to_defaults(self):
        with self.lock:
            self.pitch_pid.kp = float(PITCH_KP)
            self.pitch_pid.ki = float(PITCH_KI)
            self.pitch_pid.kd = float(PITCH_KD)
            self.roll_pid.kp = float(ROLL_KP)
            self.roll_pid.ki = float(ROLL_KI)
            self.roll_pid.kd = float(ROLL_KD)
            self.tune["pitch_mix_sign"] = float(PITCH_MIX_SIGN_DEFAULT)
            self.tune["roll_mix_sign"] = float(ROLL_MIX_SIGN_DEFAULT)
            self.tune["enabled"] = True
            self.debug_flags["debug_pitch"] = False
            self.debug_flags["debug_roll"] = False
            self.debug_flags["debug_yaw"] = False
            self.debug_flags["debug_speed"] = False
        self.pitch_pid.reset()
        self.roll_pid.reset()

    def process_commands(self, text_cmds):
        do_respawn = False
        with self.lock:
            for text_cmd in text_cmds:
                cmd_result = _execute_suspension_command(
                    raw_cmd=text_cmd,
                    pitch_pid=self.pitch_pid,
                    roll_pid=self.roll_pid,
                    suspension_tune=self.tune,
                    debug_flags=self.debug_flags,
                    pending_actions=self.pending_actions,
                    reset_defaults_fn=self.reset_to_defaults,
                )
                print(f"[suspension-cmd] {text_cmd} -> {cmd_result}")

            if self.pending_actions["respawn"]:
                self.pending_actions["respawn"] = False
                do_respawn = True
        return do_respawn

    def get_state_snapshot(self):
        with self.lock:
            return {
                "enabled": bool(self.tune["enabled"]),
                "pitch_mix_sign": float(self.tune["pitch_mix_sign"]),
                "roll_mix_sign": float(self.tune["roll_mix_sign"]),
                "debug_pitch": bool(self.debug_flags["debug_pitch"]),
                "debug_roll": bool(self.debug_flags["debug_roll"]),
                "debug_yaw": bool(self.debug_flags["debug_yaw"]),
                "debug_speed": bool(self.debug_flags["debug_speed"]),
            }

    def reset_pids(self):
        self.pitch_pid.reset()
        self.roll_pid.reset()

    def update(self, pitch, roll, sim_dt, stance_shift_leg_x,
               remote_pitch_setpoint, remote_roll_setpoint, anti_tip_pitch_bias,
               respawn_settling):
        snap = self.get_state_snapshot()
        suspension_enabled = snap["enabled"]
        pitch_mix_sign = snap["pitch_mix_sign"]
        roll_mix_sign = snap["roll_mix_sign"]

        self.pitch_pid.setpoint = float(
            np.clip(
                remote_pitch_setpoint + anti_tip_pitch_bias,
                -MAX_REMOTE_PITCH_SETPOINT_RAD,
                MAX_REMOTE_PITCH_SETPOINT_RAD,
            )
        )
        self.roll_pid.setpoint = remote_roll_setpoint

        pitch_delta_z = 0.0
        roll_delta_z = 0.0
        target_leg_pos = self.standing_leg_pos.copy()

        if respawn_settling:
            return target_leg_pos, pitch_delta_z, roll_delta_z

        if suspension_enabled:
            pitch_pid_output = float(np.clip(
                self.pitch_pid.update(pitch, sim_dt), -MAX_PID_Z_ADJUST, MAX_PID_Z_ADJUST,
            ))
            roll_pid_output = float(np.clip(
                self.roll_pid.update(roll, sim_dt), -MAX_PID_Z_ADJUST, MAX_PID_Z_ADJUST,
            ))
            pitch_delta_z = pitch_mix_sign * pitch_pid_output
            roll_delta_z = roll_mix_sign * roll_pid_output

        apply_stance_ik = suspension_enabled or abs(stance_shift_leg_x) > STANCE_SHIFT_IK_MIN_X_M
        if apply_stance_ik:
            leg_delta_z = {
                "fl": pitch_delta_z - roll_delta_z,
                "fr": pitch_delta_z + roll_delta_z,
                "rl": -pitch_delta_z - roll_delta_z,
                "rr": -pitch_delta_z + roll_delta_z,
            }

            for prefix, delta_z in leg_delta_z.items():
                desired_leg_z = self.nominal_leg_z + float(np.clip(delta_z, -MAX_LEG_DELTA_Z, MAX_LEG_DELTA_Z))
                hip_angle, knee_angle = _ik_two_link_for_vertical_position(
                    desired_leg_z=desired_leg_z,
                    desired_leg_x=stance_shift_leg_x,
                    thigh_length=self.thigh_length,
                    calf_length=self.calf_length,
                )

                hip_dof = self.leg_joint_dofs[prefix]["hip"]
                knee_dof = self.leg_joint_dofs[prefix]["knee"]
                target_leg_pos[self.leg_dof_to_local_idx[hip_dof]] = hip_angle
                target_leg_pos[self.leg_dof_to_local_idx[knee_dof]] = knee_angle

        return target_leg_pos, pitch_delta_z, roll_delta_z
