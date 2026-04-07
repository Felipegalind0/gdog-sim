# gdog-sim

![gdog_v1](gdog_v1.gif)

Wheeled quadruped simulator built with [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

- randomized robot morphology and terrain 
- photo-realistic rendering with dynamic shadows
- PID-stabilized suspension control
- Smartphone remote control via [gdog-remote](https://github.com/Felipegalind0/gdog-remote) companion app with WebSocket and optional WebRTC 4 UDP

## Quick Start

### Download source code, install dependencies, and run the sim

Clone repo:
```bash
git clone https://github.com/Felipegalind0/gdog-sim
cd gdog-sim
```

Create `.venv` if it does not exist, activate it, and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the sim with interactive viewer:
```bash
python main.py --render
```

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

- Python 3.11
- A Python virtual environment (existing `.venv` in this repo is preferred)
- Upstream Genesis wheel support for your platform

Supported in this repo today:

- macOS arm64
- Linux x86_64
- Linux ARM64 (for example Ubuntu 24 on NVIDIA DGX Spark) via custom compile script

## Installation

Install simulator dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Linux ARM64 (NVIDIA DGX Spark / Ubuntu 24)

Since upstream Genesis binaries (`quadrants`, etc.) are missing for Linux ARM64, you must compile them from source. 

**Option 1: Build with Docker (Recommended for DGX/HPC clusters)**
If you do not have `sudo` privileges on the machine, you can run everything inside a container:

```bash
docker build -t gdog-sim .
docker run --rm -p 8000:8000 --gpus all gdog-sim python main.py --render --host 0.0.0.0
```
*(If your cluster uses Apptainer/Singularity or Podman instead of Docker, the same Dockerfile will compile correctly).*

**Option 2: Use the provided automated build script**
If you have local `sudo` access, you can use our one-click script for Debian/Ubuntu distributions:

```bash
./scripts/install_ubuntu_arm64.sh
```

Optional WebRTC support:

```bash
python -m pip install -r requirements-webrtc-optional.txt
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

Custom backend bind host/port:

```bash
python main.py --render --host 0.0.0.0 --port 8000
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

### Cloudflare Quick Tunnel (No API Token)

If you want to use the HTTPS GitHub Pages remote with your local simulator backend, run an ephemeral Cloudflare Quick Tunnel.

Install once (macOS):

```bash
brew install cloudflared
```

Then launch sim with tunnel:

```bash
python main.py --render --quick-tunnel
```

If your network is slow to provision the quick tunnel URL, increase wait time:

```bash
python main.py --render --quick-tunnel --quick-tunnel-timeout 60
```

On startup, the simulator prints:

- local backend targets (`<ip>:8000`)
- tunnel backend target (for example `https://random-name.trycloudflare.com`)
- a prefilled remote link (for example `https://felipegalind0.github.io/gdog-remote?backend=https://...`)
- a terminal QR code for that prefilled link

If terminal QR rendering dependency is missing, startup prints a fallback QR image URL instead.

Paste the printed tunnel URL into gdog-remote backend input when using GitHub Pages.

Notes:

- Cloudflare Quick Tunnel does not require an API key/token for this flow
- URL is temporary and usually changes each run

Useful options:

- `--remote-url <url>`: override remote page URL (default is hardcoded to `https://felipegalind0.github.io/gdog-remote`)
- `--no-qr`: print link only, disable terminal QR rendering
- `--quick-tunnel-timeout <seconds>`: wait longer for tunnel URL discovery

If quick tunnel URL discovery still times out in the simulator, run cloudflared manually in a second terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

Then copy the `https://...trycloudflare.com` URL from that terminal into gdog-remote backend input.

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

- `requirements.txt`: aggregate install (runtime + Genesis platform marker)
- `requirements-runtime.txt`: FastAPI/transport + direct Python deps used by this repo
- `requirements-genesis.txt`: Genesis package (skips Linux ARM64)
- `requirements-genesis-deps.txt`: legacy explicit Genesis transitive set (not used by default install)
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
- Linux ARM64 startup exits with Genesis unavailable message
  - Run `./scripts/install_ubuntu_arm64.sh` to compile the missing binaries.
