import importlib
import threading
import time

import numpy as np

from math_utils import (
    _forward_xy_from_quat_wxyz,
    _local_offset_from_spherical,
    _pose_from_pos_lookat_up,
    _spherical_from_local_offset,
    _to_numpy_1d,
    _wrap_to_pi,
)


class FollowCameraController:
    def __init__(
        self,
        scene,
        render_enabled,
        cam,
        command_state,
        command_buffer_max=96,
        command_history_max=100,
    ):
        self.scene = scene
        self.render_enabled = bool(render_enabled)
        self.cam = cam
        self.command_state = command_state

        self.command_buffer_max = int(command_buffer_max)
        self.command_history_max = int(command_history_max)

        # Orbit target height above robot base position (meters).
        self.camera_center_height_offset = -0.2
        self.camera_default_offset_local = np.array([-1.5, 0.0, 0.8], dtype=np.float64)
        self.camera_lat_sensitivity = 0.015
        self.camera_scroll_lon_sensitivity = 0.04
        self.camera_scroll_lat_sensitivity = 0.04
        self.camera_zoom_sensitivity = 0.12
        self.camera_yaw_follow_gain = -1.0
        self.camera_min_lat = np.deg2rad(-80.0)
        self.camera_max_lat = np.deg2rad(80.0)
        self.camera_min_zoom = 0.6
        self.camera_max_zoom = 8.0
        self.camera_center_height_scroll_sensitivity = 0.025
        self.camera_center_height_min = -1.0
        self.camera_center_height_max = 1.5
        self.world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        cam_lon0, cam_lat0, self.cam_radius = _spherical_from_local_offset(self.camera_default_offset_local)
        self.cam_state = {
            "x": 0.0,
            "y": 0.0,
            "lon": float(cam_lon0),
            "lat": float(cam_lat0),
            "zoom": float(self.cam_radius),
        }
        self.prev_forward_xy = None
        self.sim_yaw_total = 0.0
        self.pyrender_viewer = (
            self.scene.viewer._pyrender_viewer
            if (self.render_enabled and self.scene.viewer is not None)
            else None
        )

        self.shift_held = False
        self.ctrl_held = False
        self.command_mode = False
        self.command_buffer = ""
        self.command_history = []
        self.command_history_idx = None
        self.command_history_edit_buffer = ""
        self.last_cmd_result = "type help"
        self.pending_shift_scroll = 0.0
        self.pending_scroll_lon = 0.0
        self.pending_scroll_lat = 0.0
        self.pending_center_height_delta = 0.0

        self.zoom_input_lock = threading.Lock()
        self.keyboard_drive_lock = threading.Lock()
        self.keyboard_drive_keys = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }

        self.fps_value = 0.0
        self.fps_last_t = time.perf_counter()

        self.hud_status_text = "Susp ON\nP +0.0deg  R +0.0deg"
        self.hud_debug_lines = []
        self.hud_text_align = None
        self.hud_caption_font_name = "OpenSans-Regular"
        self.hud_caption_font_pt = 12
        self.hud_caption_color = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float64)

        self.viewer_module = None
        self.pyglet_key = None
        self.shifted_char_map = None
        self.mod_ctrl = 0
        self.left_ctrl = None
        self.right_ctrl = None

        if self.pyrender_viewer is not None:
            self.viewer_module = importlib.import_module(type(self.pyrender_viewer).__module__)
            self.pyglet_key = self.viewer_module.pyglet.window.key
            self.hud_text_align = self.viewer_module.TextAlign
            self.hud_caption_font_pt = int(
                max(10, round(float(getattr(self.viewer_module, "FONT_SIZE", 14.0)) * 0.90))
            )

            self._apply_message_style_override()

            self.shifted_char_map = {
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
            self.mod_ctrl = int(getattr(self.pyglet_key, "MOD_CTRL", 0))
            self.left_ctrl = getattr(self.pyglet_key, "LCTRL", None)
            self.right_ctrl = getattr(self.pyglet_key, "RCTRL", None)

            self.pyrender_viewer.push_handlers(
                on_key_press=self._on_key_press_with_shift,
                on_key_release=self._on_key_release_with_shift,
                on_mouse_scroll=self._on_mouse_scroll_camera_control,
            )

    def _apply_message_style_override(self):
        viewer_module = self.viewer_module
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
            "[    h    ]: shadows on/off",
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
            if not self.pyrender_viewer._enable_help_text:
                return

            if self.pyrender_viewer._message_text is not None:
                if self.command_mode and all(
                    v is not None
                    for v in (
                        gl_enable,
                        gl_disable,
                        gl_scissor,
                        gl_clear_color,
                        gl_clear,
                        gl_scissor_test,
                        gl_color_buffer_bit,
                    )
                ):
                    viewport_w = int(self.pyrender_viewer._viewport_size[0])
                    message_lines = max(1, str(self.pyrender_viewer._message_text).count("\n") + 1)
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

                self.pyrender_viewer._renderer.render_text(
                    self.pyrender_viewer._message_text,
                    self.pyrender_viewer._viewport_size[0] - text_padding,
                    text_padding,
                    font_pt=font_size,
                    color=np.array(
                        [1.0, 1.0, 1.0, np.clip(self.pyrender_viewer._message_opac, 0.0, 1.0)]
                    ),
                    align=text_align.BOTTOM_RIGHT,
                )

                if self.pyrender_viewer._message_opac > 1.0:
                    self.pyrender_viewer._message_opac -= 1.0
                else:
                    self.pyrender_viewer._message_opac *= 0.90

                if self.pyrender_viewer._message_opac < 0.05:
                    self.pyrender_viewer._message_opac = 1.0 + self.pyrender_viewer._ticks_till_fade
                    self.pyrender_viewer._message_text = None

            if self.pyrender_viewer._collapse_instructions:
                self.pyrender_viewer._renderer.render_texts(
                    self.pyrender_viewer._instr_texts[0],
                    text_padding,
                    self.pyrender_viewer._viewport_size[1] - text_padding,
                    font_pt=font_size,
                    color=np.array([1.0, 1.0, 1.0, 0.85]),
                )
            else:
                expanded_lines = list(self.pyrender_viewer._key_instr_texts) + command_help_lines
                self.pyrender_viewer._renderer.render_texts(
                    expanded_lines,
                    text_padding,
                    self.pyrender_viewer._viewport_size[1] - text_padding,
                    font_pt=font_size,
                    color=np.array([1.0, 1.0, 1.0, 0.85]),
                )

            if self.hud_debug_lines:
                debug_x = self.pyrender_viewer._viewport_size[0] - text_padding
                debug_y = self.pyrender_viewer._viewport_size[1] - text_padding
                line_step = int(max(1.0, font_size * 1.1))
                for line_idx, line in enumerate(self.hud_debug_lines):
                    self.pyrender_viewer._renderer.render_text(
                        line,
                        debug_x,
                        int(debug_y - line_idx * line_step),
                        font_pt=font_size,
                        color=np.array([1.0, 1.0, 1.0, 0.95]),
                        align=text_align.TOP_RIGHT,
                    )

        self.pyrender_viewer._render_help_text = _render_help_text_white_on_black

    def _on_key_press_with_shift(self, symbol, modifiers):
        if symbol in (self.pyglet_key.LSHIFT, self.pyglet_key.RSHIFT) or (
            modifiers & self.pyglet_key.MOD_SHIFT
        ):
            self.shift_held = True
        if symbol in (self.left_ctrl, self.right_ctrl) or (modifiers & self.mod_ctrl):
            self.ctrl_held = True

        if self.command_mode:
            num_enter = getattr(self.pyglet_key, "NUM_ENTER", self.pyglet_key.ENTER)
            if symbol in (self.pyglet_key.ENTER, num_enter):
                cmd_to_send = self.command_buffer.strip()
                if cmd_to_send:
                    self.command_state.push_command(cmd_to_send)
                    self.command_history.append(cmd_to_send)
                    if len(self.command_history) > self.command_history_max:
                        self.command_history = self.command_history[-self.command_history_max :]
                    self.last_cmd_result = f"queued: {cmd_to_send}"
                else:
                    self.last_cmd_result = "empty command"
                self.command_buffer = ""
                self.command_history_idx = None
                self.command_history_edit_buffer = ""
                self.command_mode = False
                return True

            if symbol == self.pyglet_key.ESCAPE:
                self.command_mode = False
                self.command_buffer = ""
                self.command_history_idx = None
                self.command_history_edit_buffer = ""
                self.last_cmd_result = "command canceled"
                return True

            if symbol == self.pyglet_key.UP:
                if self.command_history:
                    if self.command_history_idx is None:
                        self.command_history_edit_buffer = self.command_buffer
                        self.command_history_idx = len(self.command_history) - 1
                    elif self.command_history_idx > 0:
                        self.command_history_idx -= 1
                    self.command_buffer = self.command_history[self.command_history_idx]
                return True

            if symbol == self.pyglet_key.DOWN:
                if self.command_history_idx is not None:
                    if self.command_history_idx < len(self.command_history) - 1:
                        self.command_history_idx += 1
                        self.command_buffer = self.command_history[self.command_history_idx]
                    else:
                        self.command_history_idx = None
                        self.command_buffer = self.command_history_edit_buffer
                return True

            if symbol == self.pyglet_key.BACKSPACE:
                self.command_buffer = self.command_buffer[:-1]
                return True

            if 32 <= symbol <= 126:
                ch = chr(symbol)
                if ch.isalpha():
                    if modifiers & self.pyglet_key.MOD_SHIFT:
                        ch = ch.upper()
                    else:
                        ch = ch.lower()
                elif modifiers & self.pyglet_key.MOD_SHIFT:
                    ch = self.shifted_char_map.get(ch, ch)

                if len(self.command_buffer) < self.command_buffer_max:
                    self.command_buffer += ch
                return True

            return True

        if symbol in (self.pyglet_key.SLASH, self.pyglet_key.T):
            self.command_mode = True
            self.command_buffer = "/" if symbol == self.pyglet_key.SLASH else ""
            self.command_history_idx = None
            self.command_history_edit_buffer = self.command_buffer
            return True

        if symbol == getattr(self.pyglet_key, "H", None):
            render_flags = getattr(self.pyrender_viewer, "render_flags", None)
            if isinstance(render_flags, dict) and "shadows" in render_flags:
                render_flags["shadows"] = not bool(render_flags["shadows"])
                self.pyrender_viewer.set_message_text(
                    "Shadows On" if render_flags["shadows"] else "Shadows Off"
                )
            return True

        if symbol in (
            self.pyglet_key.UP,
            self.pyglet_key.DOWN,
            self.pyglet_key.LEFT,
            self.pyglet_key.RIGHT,
        ):
            with self.keyboard_drive_lock:
                if symbol == self.pyglet_key.UP:
                    self.keyboard_drive_keys["up"] = True
                elif symbol == self.pyglet_key.DOWN:
                    self.keyboard_drive_keys["down"] = True
                elif symbol == self.pyglet_key.LEFT:
                    self.keyboard_drive_keys["left"] = True
                elif symbol == self.pyglet_key.RIGHT:
                    self.keyboard_drive_keys["right"] = True
            return True

        return None

    def _on_key_release_with_shift(self, symbol, modifiers):
        if symbol in (self.pyglet_key.LSHIFT, self.pyglet_key.RSHIFT):
            self.shift_held = False
        elif not (modifiers & self.pyglet_key.MOD_SHIFT):
            self.shift_held = False

        if symbol in (self.left_ctrl, self.right_ctrl):
            self.ctrl_held = False
        elif not (modifiers & self.mod_ctrl):
            self.ctrl_held = False

        if symbol in (
            self.pyglet_key.UP,
            self.pyglet_key.DOWN,
            self.pyglet_key.LEFT,
            self.pyglet_key.RIGHT,
        ):
            with self.keyboard_drive_lock:
                if symbol == self.pyglet_key.UP:
                    self.keyboard_drive_keys["up"] = False
                elif symbol == self.pyglet_key.DOWN:
                    self.keyboard_drive_keys["down"] = False
                elif symbol == self.pyglet_key.LEFT:
                    self.keyboard_drive_keys["left"] = False
                elif symbol == self.pyglet_key.RIGHT:
                    self.keyboard_drive_keys["right"] = False
            return True

        if self.command_mode:
            return True

        return None

    def _on_mouse_scroll_camera_control(self, x, y, dx, dy):
        # Map scroll to deterministic camera state updates.
        with self.zoom_input_lock:
            if self.ctrl_held:
                self.pending_center_height_delta += float(dy)
            elif self.shift_held:
                self.pending_shift_scroll += float(dy)
            else:
                self.pending_scroll_lon += float(dx)
                self.pending_scroll_lat -= float(dy)
        return True

    def get_keyboard_drive_flags(self):
        with self.keyboard_drive_lock:
            return (
                bool(self.keyboard_drive_keys["up"]),
                bool(self.keyboard_drive_keys["down"]),
                bool(self.keyboard_drive_keys["left"]),
                bool(self.keyboard_drive_keys["right"]),
            )

    def is_ctrl_held(self):
        return bool(self.ctrl_held)

    def get_sim_yaw_total(self):
        return float(self.sim_yaw_total)

    def set_hud(self, status_text, debug_lines):
        self.hud_status_text = str(status_text)
        self.hud_debug_lines = list(debug_lines)

    def update(self, robot, cam_dx=0.0, cam_dy=0.0, cam_zoom=0.0):
        if not (self.render_enabled or self.cam):
            return

        base_pos = _to_numpy_1d(robot.get_pos())[:3]
        forward_xy = _forward_xy_from_quat_wxyz(robot.get_quat())

        self.cam_state["x"] = float(base_pos[0])
        self.cam_state["y"] = float(base_pos[1])

        yaw_delta_measured = 0.0
        if self.prev_forward_xy is not None:
            dot = float(np.clip(np.dot(self.prev_forward_xy, forward_xy), -1.0, 1.0))
            cross_z = float(
                self.prev_forward_xy[0] * forward_xy[1]
                - self.prev_forward_xy[1] * forward_xy[0]
            )
            yaw_delta_measured = np.arctan2(cross_z, dot)
        self.prev_forward_xy = forward_xy

        self.sim_yaw_total += yaw_delta_measured
        self.cam_state["lon"] = _wrap_to_pi(
            self.cam_state["lon"] + self.camera_yaw_follow_gain * yaw_delta_measured
        )

        viewer_zoom_delta = 0.0
        viewer_scroll_lon_delta = 0.0
        viewer_scroll_lat_delta = 0.0
        viewer_center_height_delta = 0.0
        with self.zoom_input_lock:
            if self.pending_shift_scroll != 0.0:
                viewer_zoom_delta = self.pending_shift_scroll
                self.pending_shift_scroll = 0.0
            if self.pending_scroll_lon != 0.0:
                viewer_scroll_lon_delta = self.pending_scroll_lon
                self.pending_scroll_lon = 0.0
            if self.pending_scroll_lat != 0.0:
                viewer_scroll_lat_delta = self.pending_scroll_lat
                self.pending_scroll_lat = 0.0
            if self.pending_center_height_delta != 0.0:
                viewer_center_height_delta = self.pending_center_height_delta
                self.pending_center_height_delta = 0.0

        if viewer_center_height_delta != 0.0:
            self.camera_center_height_offset += (
                self.camera_center_height_scroll_sensitivity * viewer_center_height_delta
            )
            self.camera_center_height_offset = float(
                np.clip(
                    self.camera_center_height_offset,
                    self.camera_center_height_min,
                    self.camera_center_height_max,
                )
            )

        self.cam_state["lon"] = _wrap_to_pi(
            self.cam_state["lon"] + self.camera_scroll_lon_sensitivity * viewer_scroll_lon_delta
        )

        camera_center = np.array(
            [
                self.cam_state["x"],
                self.cam_state["y"],
                float(base_pos[2] + self.camera_center_height_offset),
            ],
            dtype=np.float64,
        )

        if self.render_enabled and self.scene.viewer is not None:
            # Only vertical orbit (lat) is user-controlled via mouse drag.
            viewer_impl = self.scene.viewer._pyrender_viewer
            viewer_mouse_pressed = bool(
                getattr(viewer_impl, "viewer_flags", {}).get("mouse_pressed", False)
            )
            trackball = getattr(viewer_impl, "_trackball", None)
            viewer_rotate_drag = False
            if trackball is not None:
                trackball_state = getattr(trackball, "_state", None)
                rotate_state = getattr(trackball, "STATE_ROTATE", 0)
                viewer_rotate_drag = viewer_mouse_pressed and trackball_state == rotate_state

            if viewer_rotate_drag:
                viewer_pos = _to_numpy_1d(self.scene.viewer.camera_pos)[:3]
                viewer_offset = viewer_pos - camera_center
                radius_obs = np.linalg.norm(viewer_offset)
                if radius_obs > 1e-9:
                    obs_lat = np.arcsin(np.clip(viewer_offset[2] / radius_obs, -1.0, 1.0))
                    self.cam_state["lat"] = float(
                        np.clip(obs_lat, self.camera_min_lat, self.camera_max_lat)
                    )

        # Remote mouse/touch input and no-shift vertical scroll both change lat.
        self.cam_state["lat"] = float(
            np.clip(
                self.cam_state["lat"]
                + self.camera_lat_sensitivity * cam_dy
                + self.camera_scroll_lat_sensitivity * viewer_scroll_lat_delta,
                self.camera_min_lat,
                self.camera_max_lat,
            )
        )

        total_zoom_delta = float(cam_zoom + viewer_zoom_delta)
        if total_zoom_delta != 0.0:
            self.cam_radius *= np.exp(-self.camera_zoom_sensitivity * total_zoom_delta)
            self.cam_radius = float(np.clip(self.cam_radius, self.camera_min_zoom, self.camera_max_zoom))
        self.cam_state["zoom"] = float(self.cam_radius)

        now_t = time.perf_counter()
        dt = now_t - self.fps_last_t
        self.fps_last_t = now_t
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            self.fps_value = inst_fps if self.fps_value == 0.0 else (0.9 * self.fps_value + 0.1 * inst_fps)

        camera_local_offset = _local_offset_from_spherical(
            self.cam_state["lon"], self.cam_state["lat"], self.cam_radius
        )
        camera_pos = camera_center + camera_local_offset
        camera_pose = _pose_from_pos_lookat_up(camera_pos, camera_center, self.world_up)

        if self.render_enabled and self.scene.viewer is not None:
            self.scene.viewer.set_camera_pose(pose=camera_pose)

        if self.cam:
            self.cam.set_pose(pos=camera_pos, lookat=camera_center, up=self.world_up)

        if self.pyrender_viewer is not None:
            if self.hud_text_align is not None:
                captions = [
                    {
                        "text": self.hud_status_text,
                        "location": self.hud_text_align.TOP_CENTER,
                        "font_name": self.hud_caption_font_name,
                        "font_pt": self.hud_caption_font_pt,
                        "color": self.hud_caption_color,
                        "scale": 1.0,
                    },
                ]
                self.pyrender_viewer.viewer_flags["caption"] = captions

            if self.command_mode:
                self.pyrender_viewer.set_message_text(f"% {self.command_buffer}")
            else:
                self.pyrender_viewer._message_text = None
