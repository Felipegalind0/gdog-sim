# gdog-sim

![gdog_v1](gdog_v1.gif)

Wheeled quadruped simulator built with [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).
The simulation randomizes robot morphology and terrain each run, provides a viewer-side command console for suspension tuning, and accepts remote drive commands over WebSocket (with optional WebRTC).

## What This Project Contains

- `main.py`: simulation entrypoint, scene setup, control loop, suspension PID + IK, wheel drive, HUD
- `procedural_gen.py`: randomized robot URDF generation and randomized terrain/albedo generation
- `gdog.urdf.jinja`: robot template used to build a temporary URDF at runtime
- `camera_controller.py`: follow camera, mouse/keyboard hooks, HUD overlays, in-view command input
- `commands.py`: command state container, remote payload parsing, suspension command parser/executor
- `network.py`: FastAPI app exposing `/ws` and optional `/offer`
- `pid.py`: PID controller with telemetry for debug overlays
- `math_utils.py`: quaternion helpers, leg IK helpers, camera math

## Runtime Behavior

On startup, `main.py` does the following:

1. Parses CLI arguments (`--render`, `--video`, `--seed`, `--steps`)
2. Initializes Genesis with GPU backend and a deterministic runtime seed
3. Builds a lunar-looking scene:
   - random terrain patch layout (center patch forced flat for stable spawn)
   - moon-style grayscale albedo texture with crater-like variation
   - directional sun-like light
4. Generates a randomized robot from `gdog.urdf.jinja`
5. Splits robot joints into leg joints (position-controlled) and wheel joints (velocity-controlled)
6. Starts FastAPI control server in a background thread (`0.0.0.0:8000`)
7. Runs the main simulation loop:
   - consumes remote and keyboard commands
   - applies roll/pitch stabilization through PID + two-link vertical IK
   - computes skid-steer wheel targets (`left = vx - omega`, `right = vx + omega`)
   - updates viewer/capture camera and overlays

## Requirements

- Python 3.10+
- Working Genesis-compatible environment (macOS Apple Silicon is a common target)
- A Python virtual environment (existing `.venv` in this repo is preferred)

## Installation

Use the existing `.venv` if present:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional WebRTC support:

```bash
python -m pip install -r requirements-webrtc-optional.txt
```

Create `.venv` only if it does not exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

Interactive viewer (continuous until window close or Ctrl+C):

```bash
python main.py --render
```

Deterministic randomized world:

```bash
python main.py --render --seed 12345
```

Fixed-length run:

```bash
python main.py --render --steps 2000
```

Unlimited loop explicitly:

```bash
python main.py --render --steps 0
```

Video capture mode:

```bash
python main.py --video
```

Video mode writes `wheeled_go2.mp4` (50 FPS) when the run completes.

### Step Defaults

- `--steps` provided: exact value is used
- `--video` without `--steps`: 500 steps
- `--render` without `--steps`: unlimited
- headless without `--steps`: 10000 steps

## Controls

### Remote Control Inputs

The backend accepts JSON command payloads with:

- `vx`: linear velocity command
- `omega`: yaw-rate command
- `pitch_cmd` or `pitch`: normalized pitch setpoint in `[-1, 1]`
- `roll_cmd` or `roll`: normalized roll setpoint in `[-1, 1]`
- `cam_dx`/`dx`, `cam_dy`/`dy`, `cam_zoom`/`zoom`: camera delta inputs
- `cmd` or `command`: text command for suspension console parser

### Keyboard Drive Override (Viewer)

When arrow keys are held in render mode, keyboard commands override remote drive commands:

- Up/Down: forward/back
- Left/Right: yaw

### Camera Controls (Viewer)

- Camera follows robot XY and yaw automatically
- Mouse rotate drag controls orbit latitude
- Mouse wheel:
  - horizontal scroll changes orbit longitude
  - vertical scroll changes orbit latitude
- Shift + wheel zooms in/out
- Ctrl + wheel adjusts camera target height offset

### Other Viewer Hotkeys

- `/` or `t`: open command input
- `Enter`: submit command
- `Esc`: cancel command input
- `h`: toggle shadows

## Suspension Command Console

Supported control/tuning commands include:

- `help`, `status`
- `respawn`, `reset`, `pid_reset`
- pitch PID: `kp=...`, `ki=...`, `kd=...` (also readable via `kp?`, etc.)
- roll PID: `rp=...`, `ri=...`, `rd=...` (also readable via `rp?`, etc.)
- mix sign toggles: `p_sign=1|-1`, `r_sign=1|-1`, `p_sign=flip`
- suspension enable: `susp=on|off`
- debug toggles: `debug_pitch=on|off`, `debug_roll=on|off`, `debug_yaw=on|off`, `debug_speed=on|off`

Unknown keys are reported with suggestions, and invalid values return per-key hints.

## HUD/Debug Overlays

- Always-on status (top-center):
  - suspension ON/OFF
  - current pitch and roll (degrees)
- Optional debug blocks (top-right) controlled by debug flags:
  - pitch PID internals
  - roll PID internals
  - yaw diagnostics
  - speed/wheel command diagnostics

## Network API

### WebSocket

- Endpoint: `ws://<host>:8000/ws`
- Behavior: accepts text JSON frames and updates command state

### WebRTC (Optional)

- Endpoint: `POST http://<host>:8000/offer`
- Requires `aiortc` installed
- If `aiortc` is missing, endpoint returns:

```json
{"error":"WebRTC not installed"}
```

### Important Note

There is currently **no** `/capabilities` endpoint in `network.py`.
The companion `gdog-remote` app probes that route and gracefully falls back to WebSocket mode when unavailable.

## Running With gdog-remote

Start simulator and remote UI in separate terminals.

1. In this repo:

```bash
source .venv/bin/activate
python main.py --render
```

2. In sibling repo `../gdog-remote`:

```bash
npm install
npm run dev
```

3. Open the Vite URL (typically `http://localhost:5173`)

The remote app streams controls at 50 Hz and prefers WebRTC data channel when connected; otherwise it sends over WebSocket.

## Dependencies

- `requirements.txt`: aggregate install (includes runtime + Genesis deps)
- `requirements-runtime.txt`: FastAPI/transport runtime packages
- `requirements-genesis.txt`: Genesis package
- `requirements-genesis-deps.txt`: Genesis-related native/python dependency set
- `requirements-webrtc-optional.txt`: optional `aiortc`

## Troubleshooting

- `aiortc not installed. WebRTC disabled. Using WebSockets as primary.`
  - Expected unless optional WebRTC dependency is installed
- Remote shows capability probe warning
  - Expected with current backend because `/capabilities` is not implemented
  - WebSocket control should still work
- Robot does not react to remote controls
  - confirm sim is running
  - confirm backend reachable at `http://localhost:8000/ws`
  - verify remote status shows WebSocket connected
- Startup/import issues
  - activate `.venv`
  - reinstall with `python -m pip install -r requirements.txt`
