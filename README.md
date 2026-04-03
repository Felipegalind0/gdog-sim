# gdog-sim

![gdog_v1](gdog_v1.png)

Wheeled quadruped simulation built with [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), with a remote-control backend and a lunar visual theme.

## Features

- Procedural robot generation from `gdog.urdf.jinja` on every run
- Seeded reproducibility via `--seed` (robot + terrain + visual randomization)
- Randomized multi-patch terrain (`Terrain` morph) with flat center spawn patch
- Space-style emissive sky sphere + moon-like terrain albedo
- Follow camera that tracks robot position/yaw with orbital state (`lon`, `lat`, `zoom`)
- On-screen camera diagnostics in the viewer (`x`, `y`, `lon`, `lat`, `zoom`, `fps`, `yawTotal`, `heading`)
- FastAPI control backend:
  - WebSocket endpoint: `/ws`
  - Optional WebRTC offer endpoint: `/offer`
  - Capability endpoint: `/capabilities`

## Requirements

- Python 3.10+ (3.11 recommended)
- macOS Apple Silicon works well with Genesis GPU backend
- A Python virtual environment (the repo `.venv` is recommended)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable WebRTC transport support
pip install -r requirements-webrtc-optional.txt
```

## Run

Interactive viewer:

```bash
python main.py --render
```

Interactive viewer with deterministic world generation:

```bash
python main.py --render --seed 12345
```

Headless video render:

```bash
python main.py --video
```

## Viewer Camera Controls

- Follows robot position/yaw automatically
- Mouse rotate drag adjusts camera latitude (`lat`)
- Mouse wheel:
  - horizontal scroll changes orbit longitude (`lon`)
  - vertical scroll changes latitude (`lat`)
- Hold `Shift` + wheel to zoom (`zoom`)

## Run With Web Controller

Run the simulator and remote UI in separate terminals.

1. Terminal A (this repo):

```bash
source .venv/bin/activate
python main.py --render
```

2. Terminal B (`../gdog-remote`):

```bash
npm install
npm run dev
```

3. Open the Vite URL (usually `http://localhost:5173`)

## Transport Behavior

- WebSocket is the default command path
- WebRTC is optional and only enabled when backend capability probe reports support
- If `aiortc` is missing, backend automatically stays in WebSocket mode

## Troubleshooting

- `aiortc not installed. WebRTC disabled. Using WebSockets as primary.`
  - This is expected unless you install optional WebRTC dependencies
- Frontend shows controls but robot does not move:
  - Confirm simulator is running
  - Confirm backend responds at `http://localhost:8000/capabilities`
- Startup/import issues:
  - Activate `.venv`
  - Reinstall with `pip install -r requirements.txt`
