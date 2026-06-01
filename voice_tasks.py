import time

import numpy as np

from config import (
    KEYBOARD_VX_CMD,
    KEYBOARD_YAW_CMD,
    VOICE_MOVE_MAX_LATERAL_ERROR_M,
    VOICE_MOVE_PULSE_MID_M,
    VOICE_MOVE_PULSE_NEAR_M,
    VOICE_MOVE_STOP_TOL_M,
    VOICE_MOVE_STUCK_SPEED_MPS,
    VOICE_MOVE_TIMEOUT_MIN_S,
    VOICE_MOVE_TIMEOUT_PER_M_S,
    VOICE_PROGRESS_EMIT_INTERVAL_S,
    VOICE_PWM_PERIOD_STEPS,
    VOICE_ROT_PULSE_MID_RAD,
    VOICE_ROT_PULSE_NEAR_RAD,
    VOICE_ROT_STOP_TOL_RAD,
    VOICE_ROT_STUCK_SPEED_RAD_S,
    VOICE_ROT_TIMEOUT_MIN_S,
    VOICE_ROT_TIMEOUT_PER_RAD_S,
    VOICE_STUCK_CHECK_INTERVAL_S,
    VOICE_STUCK_GRACE_S,
    VOICE_STUCK_WINDOW_S,
    VOICE_TASK_TIP_PITCH_RAD,
    VOICE_TASK_TIP_ROLL_RAD,
)


class VoiceTaskManager:
    def __init__(self, command_state):
        self.command_state = command_state
        self.active_task = None
        self.pwm_tick = 0

    def _emit_result(self, call_id, command, status, reason="", **extra):
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
        self.command_state.push_outgoing(payload)

    def _emit_progress(self, call_id, command, **extra):
        call_id_text = str(call_id or "").strip()
        if not call_id_text:
            return
        payload = {
            "type": "voice_command_progress",
            "call_id": call_id_text,
            "command": str(command),
        }
        payload.update(extra)
        self.command_state.push_outgoing(payload)

    def _finish_task(self, task, status, reason="", **extra):
        if task is None:
            return None
        self._emit_result(
            call_id=task.get("call_id"),
            command=task.get("type", "unknown"),
            status=status,
            reason=reason,
            **extra,
        )
        return None

    def cancel(self, reason):
        self.active_task = self._finish_task(self.active_task, status="failed", reason=reason)

    @property
    def has_active_task(self):
        return self.active_task is not None

    def dispatch(self, voice_cmd, voice_direction, voice_amount, voice_call_id,
                 curr_pos, curr_forward_xy, curr_heading):
        received = str(voice_cmd).strip().lower() if voice_cmd else ""

        if received == "stop":
            if self.active_task is not None:
                self.cancel("Command was stopped before completion.")
            else:
                self._emit_result(
                    call_id=voice_call_id,
                    command="stop",
                    status="completed",
                    reason="No active command was running.",
                )
            return True

        if received not in ("move", "rotate"):
            return False

        if self.active_task is not None:
            active_call_id = str(self.active_task.get("call_id") or "").strip()
            incoming_call_id = str(voice_call_id or "").strip()
            if not (incoming_call_id and active_call_id and incoming_call_id == active_call_id):
                active_command = str(self.active_task.get("type", "command"))
                self._emit_result(
                    call_id=voice_call_id,
                    command=received,
                    status="failed",
                    reason=(
                        f"Cannot start '{received}' while '{active_command}' is still running. "
                        "Wait for completion or send stop first."
                    ),
                    active_command=active_command,
                )
            return True

        requested_dir = str(voice_direction or "").strip().lower()
        requested_amt = max(float(voice_amount), 0.0)

        if received == "rotate" and requested_amt > (2.0 * np.pi + 0.25):
            requested_amt = float(np.deg2rad(requested_amt))

        if received == "move" and requested_amt > 0.0:
            move_sign = 0.0
            if requested_dir in ("", "forward", "fwd"):
                move_sign = 1.0
            elif requested_dir in ("backward", "back", "bwd", "reverse", "rev"):
                move_sign = -1.0

            if move_sign != 0.0:
                timeout_s = max(VOICE_MOVE_TIMEOUT_MIN_S, float(requested_amt) * VOICE_MOVE_TIMEOUT_PER_M_S)
                self.active_task = {
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
                self.pwm_tick = 0
            else:
                self._emit_result(
                    call_id=voice_call_id, command="move", status="failed",
                    reason="Invalid move direction. Use 'forward' or 'backward'.",
                )

        elif received == "rotate" and requested_amt > 0.0:
            yaw_sign = 0.0
            if requested_dir in ("left", "l", "ccw", "counterclockwise"):
                yaw_sign = 1.0
            elif requested_dir in ("", "right", "r", "cw", "clockwise"):
                yaw_sign = -1.0

            if yaw_sign != 0.0:
                timeout_s = max(VOICE_ROT_TIMEOUT_MIN_S, float(requested_amt) * VOICE_ROT_TIMEOUT_PER_RAD_S)
                self.active_task = {
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
                self.pwm_tick = 0
            else:
                self._emit_result(
                    call_id=voice_call_id, command="rotate", status="failed",
                    reason="Invalid rotate direction. Use 'left' or 'right'.",
                )
        else:
            self._emit_result(
                call_id=voice_call_id, command=received, status="failed",
                reason="Invalid command amount. Value must be positive.",
            )

        return True

    def check_cancellations(self, manual_input, pitch, roll, voice_cmd_received):
        if manual_input and not voice_cmd_received:
            self.cancel("Cancelled by manual control input.")

        if self.active_task is not None:
            tipped = abs(float(pitch)) >= VOICE_TASK_TIP_PITCH_RAD or abs(float(roll)) >= VOICE_TASK_TIP_ROLL_RAD
            if tipped:
                self.active_task = self._finish_task(
                    self.active_task,
                    status="failed",
                    reason=(
                        f"Robot tipped over (pitch={np.degrees(pitch):+.1f} deg, "
                        f"roll={np.degrees(roll):+.1f} deg)."
                    ),
                    pitch_deg=float(np.degrees(pitch)),
                    roll_deg=float(np.degrees(roll)),
                )
                return 0.0, 0.0, True
        return None, None, False

    def check_timeout(self):
        if self.active_task is None:
            return None, None, False

        elapsed_s = float(time.monotonic() - float(self.active_task.get("started_at", 0.0)))
        timeout_s = float(self.active_task.get("timeout_s", VOICE_MOVE_TIMEOUT_MIN_S))
        if elapsed_s > timeout_s:
            self.active_task = self._finish_task(
                self.active_task,
                status="failed",
                reason="Fallback safety timeout reached before completion.",
                elapsed_s=elapsed_s,
                timeout_s=timeout_s,
            )
            return 0.0, 0.0, True
        return None, None, False

    def tick(self, curr_pos, curr_forward_xy, curr_heading):
        if self.active_task is None:
            return None, None, False

        self.pwm_tick += 1
        pwm_phase = self.pwm_tick % VOICE_PWM_PERIOD_STEPS
        elapsed_s = float(time.monotonic() - float(self.active_task.get("started_at", 0.0)))

        if self.active_task["type"] == "move":
            return self._tick_move(curr_pos, pwm_phase, elapsed_s)
        elif self.active_task["type"] == "rotate":
            return self._tick_rotate(curr_heading, pwm_phase, elapsed_s)

        return None, None, False

    def _tick_move(self, curr_pos, pwm_phase, elapsed_s):
        task = self.active_task
        displacement_xy = curr_pos[:2] - task["start_pos_xy"]
        progress = float(np.dot(displacement_xy, task["start_forward_xy"]))
        progress *= float(task["dir_sign"])
        remaining = float(task["target"]) - progress
        lateral_axis_xy = np.array(
            [-float(task["start_forward_xy"][1]), float(task["start_forward_xy"][0])],
            dtype=float,
        )
        lateral_error_m = abs(float(np.dot(displacement_xy, lateral_axis_xy)))

        progress_clamped = float(np.clip(progress, 0.0, float(task["target"])))
        remaining_clamped = float(max(remaining, 0.0))
        target_m = float(task["target"])
        now_monotonic = float(time.monotonic())

        dt_stuck = now_monotonic - task["last_stuck_check_time"]
        if dt_stuck >= VOICE_STUCK_CHECK_INTERVAL_S:
            dp = progress_clamped - task["last_stuck_check_progress"]
            speed = dp / max(dt_stuck, 1e-6)
            task["current_speed"] = float(speed)

            should_count_as_stuck = (
                elapsed_s >= VOICE_STUCK_GRACE_S
                and remaining_clamped > VOICE_MOVE_STOP_TOL_M
                and speed < VOICE_MOVE_STUCK_SPEED_MPS
            )
            if should_count_as_stuck:
                task["stuck_time_accum"] += dt_stuck
            else:
                task["stuck_time_accum"] = 0.0

            task["last_stuck_check_progress"] = progress_clamped
            task["last_stuck_check_time"] = now_monotonic

            if task["stuck_time_accum"] > VOICE_STUCK_WINDOW_S:
                self.active_task = self._finish_task(
                    task,
                    status="failed",
                    reason=(
                        "Robot stopped making progress toward the move target "
                        f"({speed:.2f} m/s for {VOICE_STUCK_WINDOW_S:.1f}s)."
                    ),
                    stuck_for_s=float(VOICE_STUCK_WINDOW_S),
                )
                return 0.0, 0.0, True

        if now_monotonic - float(task.get("last_progress_emit_at", -1e9)) >= VOICE_PROGRESS_EMIT_INTERVAL_S:
            self._emit_progress(
                call_id=task.get("call_id"),
                command="move",
                direction=str(task.get("direction", "forward")),
                progress_m=progress_clamped,
                target_m=target_m,
                remaining_m=remaining_clamped,
                progress_ratio=float(np.clip(progress_clamped / max(target_m, 1e-6), 0.0, 1.0)),
                current_speed=task.get("current_speed", 0.0),
                lateral_error_m=float(lateral_error_m),
                elapsed_s=float(elapsed_s),
                timeout_s=float(task.get("timeout_s", 0.0)),
            )
            task["last_progress_emit_at"] = now_monotonic

        if remaining <= VOICE_MOVE_STOP_TOL_M:
            if lateral_error_m <= VOICE_MOVE_MAX_LATERAL_ERROR_M:
                self.active_task = self._finish_task(
                    task, status="completed",
                    progress_m=float(progress), target_m=float(task["target"]),
                    remaining_m=float(max(remaining, 0.0)), lateral_error_m=float(lateral_error_m),
                )
            else:
                self.active_task = self._finish_task(
                    task, status="failed",
                    reason="Robot reached the wrong destination (path deviation too large).",
                    progress_m=float(progress), target_m=float(task["target"]),
                    remaining_m=float(max(remaining, 0.0)), lateral_error_m=float(lateral_error_m),
                )
            return 0.0, 0.0, False

        if remaining > VOICE_MOVE_PULSE_MID_M:
            duty = 1.0
        elif remaining > VOICE_MOVE_PULSE_NEAR_M:
            duty = 0.6
        else:
            duty = 0.3
        on_steps = max(1, int(round(VOICE_PWM_PERIOD_STEPS * duty)))
        is_on = pwm_phase < on_steps
        vx = float(task["dir_sign"]) * KEYBOARD_VX_CMD if is_on else 0.0
        return vx, 0.0, False

    def _tick_rotate(self, curr_heading, pwm_phase, elapsed_s):
        task = self.active_task
        step_delta = (curr_heading - task["prev_heading"] + np.pi) % (2.0 * np.pi) - np.pi
        task["prev_heading"] = curr_heading
        task["progress"] += float(step_delta) * float(task["dir_sign"])
        remaining = float(task["target"]) - float(task["progress"])

        progress_clamped_rad = float(np.clip(float(task["progress"]), 0.0, float(task["target"])))
        remaining_clamped_rad = float(max(remaining, 0.0))
        target_rad = float(task["target"])
        now_monotonic = float(time.monotonic())

        dt_stuck = now_monotonic - task["last_stuck_check_time"]
        if dt_stuck >= VOICE_STUCK_CHECK_INTERVAL_S:
            dp = progress_clamped_rad - task["last_stuck_check_progress"]
            speed = dp / max(dt_stuck, 1e-6)
            task["current_speed"] = float(speed)

            should_count_as_stuck = (
                elapsed_s >= VOICE_STUCK_GRACE_S
                and remaining_clamped_rad > VOICE_ROT_STOP_TOL_RAD
                and speed < VOICE_ROT_STUCK_SPEED_RAD_S
            )
            if should_count_as_stuck:
                task["stuck_time_accum"] += dt_stuck
            else:
                task["stuck_time_accum"] = 0.0

            task["last_stuck_check_progress"] = progress_clamped_rad
            task["last_stuck_check_time"] = now_monotonic

            if task["stuck_time_accum"] > VOICE_STUCK_WINDOW_S:
                self.active_task = self._finish_task(
                    task,
                    status="failed",
                    reason=(
                        "Robot stopped making progress toward the rotate target "
                        f"({np.degrees(speed):.1f} deg/s for {VOICE_STUCK_WINDOW_S:.1f}s)."
                    ),
                    stuck_for_s=float(VOICE_STUCK_WINDOW_S),
                )
                return 0.0, 0.0, True

        if now_monotonic - float(task.get("last_progress_emit_at", -1e9)) >= VOICE_PROGRESS_EMIT_INTERVAL_S:
            self._emit_progress(
                call_id=task.get("call_id"),
                command="rotate",
                direction=str(task.get("direction", "left")),
                progress_rad=progress_clamped_rad,
                target_rad=target_rad,
                remaining_rad=remaining_clamped_rad,
                progress_deg=float(np.degrees(progress_clamped_rad)),
                target_deg=float(np.degrees(target_rad)),
                remaining_deg=float(np.degrees(remaining_clamped_rad)),
                progress_ratio=float(np.clip(progress_clamped_rad / max(target_rad, 1e-6), 0.0, 1.0)),
                current_speed=task.get("current_speed", 0.0),
                elapsed_s=float(elapsed_s),
                timeout_s=float(task.get("timeout_s", 0.0)),
            )
            task["last_progress_emit_at"] = now_monotonic

        if remaining <= VOICE_ROT_STOP_TOL_RAD:
            self.active_task = self._finish_task(
                task, status="completed",
                progress_rad=float(task["progress"]),
                target_rad=float(task["target"]),
                remaining_rad=float(max(remaining, 0.0)),
            )
            return 0.0, 0.0, False

        if remaining > VOICE_ROT_PULSE_MID_RAD:
            duty = 1.0
        elif remaining > VOICE_ROT_PULSE_NEAR_RAD:
            duty = 0.6
        else:
            duty = 0.3
        on_steps = max(1, int(round(VOICE_PWM_PERIOD_STEPS * duty)))
        is_on = pwm_phase < on_steps
        omega = -float(task["dir_sign"]) * KEYBOARD_YAW_CMD if is_on else 0.0
        return 0.0, omega, False
