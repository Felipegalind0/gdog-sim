# gdog (MVP Robot Dog Simulator)

A minimal viable product (MVP) for a quadruped robot simulation using the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine. This simulator is explicitly optimized for Apple Silicon (such as the M1 MacBook Air), utilizing the Metal backend for fast, hardware-accelerated physics and rendering.

## Features
- **Custom Quadruped URDF**: Features a procedural 4-legged robot (`simple_robot.urdf`) with a central chassis and independent 2-segment legs (thighs and calves) marked in contrasting colors.
- **Active Pose Control**: The simulation automatically actuates the joints to hold a Z-fold standing posture (hips at 45°, knees at -90°).
- **Apple Silicon (M1) Optimizations**: 
  - Hardcoded internal 1080p rendering resolution to decouple from demanding Retina fullscreen constraints.
  - Disabled plane reflections to save raw GPU power and keep framerates well above 120 FPS.
  - Increased ambient lighting for better visibility on macOS displays.

## Installation

Make sure you have a generic Python virtual environment active.

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install the core requirements:
pip install -r requirements.txt

# You also need to manually install PyTorch for the Genesis engine tensor ops:
pip install torch
```

## Usage

The `main.py` script offers two distinct modes for observing the physics simulation:

**1. Interactive Viewer (GUI)**
Opens a floatable 1080p high-performance real-time 3D window.
```bash
python main.py --render
```

**2. Headless Video Render**
Steps the physics engine without launching a UI window and saves the results to `go2_standing.mp4`.
```bash
python main.py --video
```
