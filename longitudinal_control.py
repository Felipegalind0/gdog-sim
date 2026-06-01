import numpy as np

from config import (
    DRIVE_ACCEL_RESPONSE_GAIN,
    DRIVE_BRAKE_PITCH_BLOCK_RAD,
    DRIVE_BRAKE_PITCH_RATE_BLOCK_RAD_S,
    DRIVE_BRAKE_PITCH_RATE_WARN_RAD_S,
    DRIVE_BRAKE_PITCH_WARN_RAD,
    DRIVE_BRAKE_PREDICTIVE_GAIN,
    DRIVE_BRAKE_RESPONSE_GAIN,
    DRIVE_BRAKE_REVERSE_SCALE,
    DRIVE_BRAKE_THROTTLE_MIN,
    DRIVE_INPUT_OMEGA_GAIN,
    DRIVE_INPUT_VX_GAIN,
    DRIVE_OMEGA_SCALE_MAX,
    DRIVE_POWER_SCALE_MAX,
    DRIVE_POWER_SCALE_MIN,
    DRIVE_POWER_SPEED_FULL_MPS,
    DRIVE_POWER_SPEED_START_MPS,
    DRIVE_POWER_YAW_FULL_RAD_S,
    DRIVE_POWER_YAW_START_RAD_S,
    DRIVE_STATE_FILTER_ALPHA,
    DRIVE_VX_ACCEL_LIMIT,
    DRIVE_VX_ACCEL_LIMIT_STATIONARY,
    DRIVE_VX_DECEL_LIMIT,
    DRIVE_VX_DECEL_LIMIT_STATIONARY,
    KEYBOARD_VX_CMD,
    KEYBOARD_YAW_CMD,
    ROTATE_DRIFT_I_LIMIT,
    ROTATE_DRIFT_KD,
    ROTATE_DRIFT_KI,
    ROTATE_DRIFT_KP,
    ROTATE_DRIFT_MAX_CORRECTION,
    ROTATE_ONLY_OMEGA_MIN,
    ROTATE_ONLY_VX_DEADBAND,
    SIM_DT,
    STANCE_SHIFT_ACCEL_MAX_MPS2,
    STANCE_SHIFT_CMD_SPEED_MAX_MPS,
    STANCE_SHIFT_FILTER_ALPHA,
    STANCE_SHIFT_MAX_LEG_X_M,
    STANCE_SHIFT_MIN_SCALE,
    STANCE_SHIFT_RISK_REDUCTION_GAIN,
    STANCE_SHIFT_SPEED_ERROR_TO_ACCEL_GAIN,
    STANCE_SHIFT_TILT_TO_LEG_X_GAIN,
    TRACTION_CMD_ACCEL_BLOCK,
    TRACTION_CMD_ACCEL_WARN,
    TRACTION_MIN_SCALE,
    TRACTION_NEUTRAL_CMD,
    TRACTION_NEUTRAL_PITCH_ALPHA,
    TRACTION_NEUTRAL_SPEED_MPS,
    TRACTION_PITCH_BLOCK_RAD,
    TRACTION_PITCH_RATE_BLOCK_RAD_S,
    TRACTION_PITCH_RATE_WARN_RAD_S,
    TRACTION_PITCH_WARN_RAD,
    TRACTION_RISK_HARD_BLOCK,
    TRACTION_RISK_START,
    ANTI_TIP_MAX_SETPOINT_BIAS_RAD,
    ANTI_TIP_PITCH_DAMP_GAIN,
    ANTI_TIP_PITCH_RESTORE_GAIN,
    ANTI_TIP_RISK_BOOST_GAIN,
)


class LongitudinalController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_pos_xy = None
        self.prev_pitch = None
        self.prev_heading = None
        self.pitch_neutral = 0.0
        self.filtered_forward_speed = 0.0
        self.filtered_pitch_rate = 0.0
        self.filtered_yaw_rate = 0.0
        self.vx_cmd_smooth = 0.0
        self.rotate_hold_active = False
        self.rotate_hold_anchor_xy = np.zeros(2, dtype=float)
        self.rotate_hold_integral = 0.0
        self.rotate_drift_err_m = 0.0
        self.rotate_drift_correction = 0.0
        self.front_unload_risk = 0.0
        self.traction_scale = 1.0
        self.front_traction_blocked = False
        self.forward_accel_cmd = 0.0
        self.forward_speed_meas = 0.0
        self.pitch_rate_meas = 0.0
        self.yaw_rate_meas = 0.0
        self.drive_speed_ratio = 0.0
        self.drive_power_scale = 1.0
        self.vx_power_cap = float(KEYBOARD_VX_CMD)
        self.omega_power_cap = float(abs(KEYBOARD_YAW_CMD))
        self.drive_accel_limit = DRIVE_VX_ACCEL_LIMIT_STATIONARY
        self.drive_brake_limit = DRIVE_VX_DECEL_LIMIT_STATIONARY
        self.brake_throttle = 1.0
        self.predictive_brake_scale = 1.0
        self.vx_cmd_norm = 0.0
        self.desired_accel_target = 0.0
        self.desired_tilt_target = 0.0
        self.anti_tip_pitch_bias = 0.0
        self.stance_shift_target_x = 0.0
        self.stance_shift_leg_x = 0.0

    def reset_for_respawn(self, spawn_pos_xy):
        self.reset()
        self.prev_pos_xy = np.asarray(spawn_pos_xy, dtype=float).copy()
        self.prev_pitch = 0.0
        self.prev_heading = 0.0

    def estimate_state(self, curr_pos, curr_forward_xy, curr_heading, pitch, sim_dt):
        if self.prev_pos_xy is None:
            self.prev_pos_xy = curr_pos[:2].copy()
        if self.prev_pitch is None:
            self.prev_pitch = float(pitch)
            self.pitch_neutral = float(pitch)
        if self.prev_heading is None:
            self.prev_heading = float(curr_heading)

        delta_pos_xy = curr_pos[:2] - self.prev_pos_xy
        forward_speed_raw = float(np.dot(delta_pos_xy / sim_dt, curr_forward_xy))
        self.filtered_forward_speed += DRIVE_STATE_FILTER_ALPHA * (forward_speed_raw - self.filtered_forward_speed)
        self.forward_speed_meas = float(self.filtered_forward_speed)

        pitch_rate_raw = float((pitch - self.prev_pitch) / sim_dt)
        self.filtered_pitch_rate += DRIVE_STATE_FILTER_ALPHA * (pitch_rate_raw - self.filtered_pitch_rate)
        self.pitch_rate_meas = float(self.filtered_pitch_rate)

        yaw_delta = (curr_heading - self.prev_heading + np.pi) % (2.0 * np.pi) - np.pi
        yaw_rate_raw = float(yaw_delta / sim_dt)
        self.filtered_yaw_rate += DRIVE_STATE_FILTER_ALPHA * (yaw_rate_raw - self.filtered_yaw_rate)
        self.yaw_rate_meas = float(self.filtered_yaw_rate)

        self.prev_pos_xy = curr_pos[:2].copy()
        self.prev_pitch = float(pitch)
        self.prev_heading = float(curr_heading)

    def update(self, vx_raw, omega_raw, pitch, curr_pos_xy, curr_forward_xy, sim_dt):
        # Boost all operator/requested drive commands before safety envelopes.
        vx = float(vx_raw) * DRIVE_INPUT_VX_GAIN
        omega = float(omega_raw) * DRIVE_INPUT_OMEGA_GAIN

        linear_speed_ratio = float(
            np.clip(
                (abs(self.forward_speed_meas) - DRIVE_POWER_SPEED_START_MPS)
                / max(DRIVE_POWER_SPEED_FULL_MPS - DRIVE_POWER_SPEED_START_MPS, 1e-6),
                0.0,
                1.0,
            )
        )
        yaw_speed_ratio = float(
            np.clip(
                (abs(self.yaw_rate_meas) - DRIVE_POWER_YAW_START_RAD_S)
                / max(DRIVE_POWER_YAW_FULL_RAD_S - DRIVE_POWER_YAW_START_RAD_S, 1e-6),
                0.0,
                1.0,
            )
        )
        self.drive_speed_ratio = float(max(linear_speed_ratio, yaw_speed_ratio))
        self.drive_power_scale = float(
            np.clip(
                DRIVE_POWER_SCALE_MIN
                + (DRIVE_POWER_SCALE_MAX - DRIVE_POWER_SCALE_MIN) * self.drive_speed_ratio,
                DRIVE_POWER_SCALE_MIN,
                DRIVE_POWER_SCALE_MAX,
            )
        )
        omega_scale = float(
            np.clip(
                1.0 + (DRIVE_OMEGA_SCALE_MAX - 1.0) * self.drive_speed_ratio,
                1.0,
                DRIVE_OMEGA_SCALE_MAX,
            )
        )
        self.vx_power_cap = float(KEYBOARD_VX_CMD * self.drive_power_scale)
        self.omega_power_cap = float(abs(KEYBOARD_YAW_CMD) * omega_scale)

        vx_cmd_raw = float(np.clip(vx, -self.vx_power_cap, self.vx_power_cap))
        omega_cmd_raw = float(np.clip(omega, -self.omega_power_cap, self.omega_power_cap))

        rotate_only_requested = (
            abs(omega_cmd_raw) > ROTATE_ONLY_OMEGA_MIN
            and abs(vx_cmd_raw) < ROTATE_ONLY_VX_DEADBAND
        )

        if rotate_only_requested:
            if not self.rotate_hold_active:
                self.rotate_hold_active = True
                self.rotate_hold_anchor_xy = curr_pos_xy.copy()
                self.rotate_hold_integral = 0.0

            self.rotate_drift_err_m = float(np.dot(curr_pos_xy - self.rotate_hold_anchor_xy, curr_forward_xy))
            self.rotate_hold_integral = float(
                np.clip(
                    self.rotate_hold_integral + self.rotate_drift_err_m * sim_dt,
                    -ROTATE_DRIFT_I_LIMIT,
                    ROTATE_DRIFT_I_LIMIT,
                )
            )
            self.rotate_drift_correction = float(
                np.clip(
                    -(
                        ROTATE_DRIFT_KP * self.rotate_drift_err_m
                        + ROTATE_DRIFT_KI * self.rotate_hold_integral
                        + ROTATE_DRIFT_KD * self.forward_speed_meas
                    ),
                    -ROTATE_DRIFT_MAX_CORRECTION,
                    ROTATE_DRIFT_MAX_CORRECTION,
                )
            )
        else:
            self.rotate_hold_active = False
            self.rotate_hold_anchor_xy = curr_pos_xy.copy()
            self.rotate_hold_integral = 0.0
            self.rotate_drift_err_m = 0.0
            self.rotate_drift_correction = 0.0

        vx_cmd_after_rotate = float(vx_cmd_raw + self.rotate_drift_correction)

        # Predictive desired longitudinal acceleration from normalized command.
        self.vx_cmd_norm = float(np.clip(vx_cmd_after_rotate / max(self.vx_power_cap, 1e-6), -1.0, 1.0))
        desired_speed_target = float(self.vx_cmd_norm * STANCE_SHIFT_CMD_SPEED_MAX_MPS)
        speed_error = float(desired_speed_target - self.forward_speed_meas)
        self.desired_accel_target = float(
            np.clip(
                STANCE_SHIFT_SPEED_ERROR_TO_ACCEL_GAIN * speed_error,
                -STANCE_SHIFT_ACCEL_MAX_MPS2,
                STANCE_SHIFT_ACCEL_MAX_MPS2,
            )
        )
        self.desired_tilt_target = float(np.arctan2(self.desired_accel_target, 9.81))
        decel_alignment = float(-np.sign(self.forward_speed_meas) * self.desired_accel_target)
        decel_demand = float(
            np.clip(
                decel_alignment / max(STANCE_SHIFT_ACCEL_MAX_MPS2, 1e-6),
                0.0,
                1.0,
            )
        )
        self.predictive_brake_scale = float(1.0 + DRIVE_BRAKE_PREDICTIVE_GAIN * decel_demand)

        dynamic_pitch_for_brake = float(pitch - self.pitch_neutral)
        brake_pitch_risk = float(
            np.clip(
                (abs(dynamic_pitch_for_brake) - DRIVE_BRAKE_PITCH_WARN_RAD)
                / max(DRIVE_BRAKE_PITCH_BLOCK_RAD - DRIVE_BRAKE_PITCH_WARN_RAD, 1e-6),
                0.0,
                1.0,
            )
        )
        brake_pitch_rate_risk = float(
            np.clip(
                (abs(self.pitch_rate_meas) - DRIVE_BRAKE_PITCH_RATE_WARN_RAD_S)
                / max(DRIVE_BRAKE_PITCH_RATE_BLOCK_RAD_S - DRIVE_BRAKE_PITCH_RATE_WARN_RAD_S, 1e-6),
                0.0,
                1.0,
            )
        )
        self.drive_accel_limit = float(
            DRIVE_VX_ACCEL_LIMIT_STATIONARY
            + (DRIVE_VX_ACCEL_LIMIT - DRIVE_VX_ACCEL_LIMIT_STATIONARY) * self.drive_speed_ratio
        ) * DRIVE_ACCEL_RESPONSE_GAIN
        self.drive_brake_limit = float(
            DRIVE_VX_DECEL_LIMIT_STATIONARY
            + (DRIVE_VX_DECEL_LIMIT - DRIVE_VX_DECEL_LIMIT_STATIONARY) * self.drive_speed_ratio
        ) * DRIVE_BRAKE_RESPONSE_GAIN

        prev_vx_cmd_smooth = float(self.vx_cmd_smooth)
        vx_target_delta = vx_cmd_after_rotate - prev_vx_cmd_smooth
        same_direction = np.sign(vx_cmd_after_rotate) == np.sign(prev_vx_cmd_smooth)
        accelerating = same_direction and abs(vx_cmd_after_rotate) > abs(prev_vx_cmd_smooth)

        self.brake_throttle = 1.0
        if not accelerating:
            self.brake_throttle = float(
                np.clip(
                    1.0 - 0.55 * brake_pitch_risk - 0.45 * brake_pitch_rate_risk,
                    DRIVE_BRAKE_THROTTLE_MIN,
                    1.0,
                )
            )
            reversing = (
                np.sign(vx_cmd_after_rotate) != 0.0
                and np.sign(prev_vx_cmd_smooth) != 0.0
                and np.sign(vx_cmd_after_rotate) != np.sign(prev_vx_cmd_smooth)
            )
            if reversing:
                self.brake_throttle = max(
                    DRIVE_BRAKE_THROTTLE_MIN,
                    self.brake_throttle * DRIVE_BRAKE_REVERSE_SCALE,
                )

        if accelerating:
            max_vx_step = self.drive_accel_limit * sim_dt
        else:
            max_vx_step = self.drive_brake_limit * self.predictive_brake_scale * self.brake_throttle * sim_dt

        self.vx_cmd_smooth = prev_vx_cmd_smooth + float(np.clip(vx_target_delta, -max_vx_step, max_vx_step))
        self.forward_accel_cmd = float((self.vx_cmd_smooth - prev_vx_cmd_smooth) / sim_dt)

        if abs(self.forward_speed_meas) < TRACTION_NEUTRAL_SPEED_MPS and abs(self.vx_cmd_smooth) < TRACTION_NEUTRAL_CMD:
            self.pitch_neutral = float(
                (1.0 - TRACTION_NEUTRAL_PITCH_ALPHA) * self.pitch_neutral
                + TRACTION_NEUTRAL_PITCH_ALPHA * pitch
            )
        dynamic_pitch = float(pitch - self.pitch_neutral)

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
                (abs(self.pitch_rate_meas) - TRACTION_PITCH_RATE_WARN_RAD_S)
                / max(TRACTION_PITCH_RATE_BLOCK_RAD_S - TRACTION_PITCH_RATE_WARN_RAD_S, 1e-6),
                0.0,
                1.0,
            )
        )
        cmd_accel_risk = float(
            np.clip(
                (abs(self.forward_accel_cmd) - TRACTION_CMD_ACCEL_WARN)
                / max(TRACTION_CMD_ACCEL_BLOCK - TRACTION_CMD_ACCEL_WARN, 1e-6),
                0.0,
                1.0,
            )
        )

        moving_or_driving = (
            abs(self.forward_speed_meas) > TRACTION_NEUTRAL_SPEED_MPS
            or abs(self.vx_cmd_smooth) > TRACTION_NEUTRAL_CMD
        )
        if moving_or_driving:
            self.front_unload_risk = float(0.55 * pitch_risk + 0.25 * pitch_rate_risk + 0.20 * cmd_accel_risk)
        else:
            self.front_unload_risk = 0.0

        self.traction_scale = 1.0
        if self.front_unload_risk > TRACTION_RISK_START:
            self.traction_scale = float(
                np.clip(
                    1.0 - (self.front_unload_risk - TRACTION_RISK_START) / max(1.0 - TRACTION_RISK_START, 1e-6),
                    TRACTION_MIN_SCALE,
                    1.0,
                )
            )

        self.front_traction_blocked = bool(
            self.front_unload_risk >= TRACTION_RISK_HARD_BLOCK
            and abs(self.vx_cmd_smooth) > TRACTION_NEUTRAL_CMD
        )
        vx_cmd_limited = float(self.vx_cmd_smooth * self.traction_scale)
        if self.front_traction_blocked:
            vx_cmd_limited = 0.0

        balance_gain = float(
            np.clip(
                abs(self.desired_accel_target) / max(STANCE_SHIFT_ACCEL_MAX_MPS2, 1e-6),
                0.0,
                1.0,
            )
        )
        balance_gain = max(balance_gain, self.front_unload_risk)
        self.anti_tip_pitch_bias = -(
            ANTI_TIP_PITCH_DAMP_GAIN * self.pitch_rate_meas
            + ANTI_TIP_PITCH_RESTORE_GAIN * dynamic_pitch
        )
        self.anti_tip_pitch_bias *= balance_gain * (1.0 + ANTI_TIP_RISK_BOOST_GAIN * self.front_unload_risk)
        self.anti_tip_pitch_bias = float(
            np.clip(
                self.anti_tip_pitch_bias,
                -ANTI_TIP_MAX_SETPOINT_BIAS_RAD,
                ANTI_TIP_MAX_SETPOINT_BIAS_RAD,
            )
        )

        # Single inverted-pendulum target driven by desired tilt.
        stance_target_uncapped = STANCE_SHIFT_TILT_TO_LEG_X_GAIN * self.desired_tilt_target
        stance_scale = float(
            np.clip(
                1.0 - STANCE_SHIFT_RISK_REDUCTION_GAIN * self.front_unload_risk,
                STANCE_SHIFT_MIN_SCALE,
                1.0,
            )
        )

        self.stance_shift_target_x = float(
            np.clip(
                stance_target_uncapped * stance_scale,
                -STANCE_SHIFT_MAX_LEG_X_M,
                STANCE_SHIFT_MAX_LEG_X_M,
            )
        )
        self.stance_shift_leg_x += STANCE_SHIFT_FILTER_ALPHA * (
            self.stance_shift_target_x - self.stance_shift_leg_x
        )
        self.stance_shift_leg_x = float(
            np.clip(
                self.stance_shift_leg_x,
                -STANCE_SHIFT_MAX_LEG_X_M,
                STANCE_SHIFT_MAX_LEG_X_M,
            )
        )

        return vx_cmd_limited, omega_cmd_raw, vx_cmd_raw
