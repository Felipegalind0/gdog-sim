import difflib
import threading

import numpy as np


PID_GAIN_MAX = 2.0
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

    def update(
        self,
        vx,
        omega,
        pitch_cmd=0.0,
        roll_cmd=0.0,
        cam_dx=0.0,
        cam_dy=0.0,
        cam_zoom=0.0,
        txt_cmd=None,
    ):
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
    base = f"unknown key '{key_raw}'. valid keys: {', '.join(CANONICAL_CMD_KEYS)}"
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
            return f"invalid value for '{key}': {exc}. expected {hint}. example: {example}"

    return (
        f"parse error: '{cmd}'. expected '<key>=<value>' or '<key>?', "
        "or one of: status, help, respawn, reset, pid_reset"
    )


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
