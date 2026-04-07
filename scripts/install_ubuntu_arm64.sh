#!/usr/bin/env bash
# Automated install script for gdog-sim on Ubuntu/Linux ARM64 engines (e.g., NVIDIA DGX Spark)
# Compiles missing PyPI wheels from source automatically.

set -e

echo "=== gdog-sim Ubuntu ARM64 Auto-Installer ==="

# 1. Install system prerequisites (C++ compilers for tetgen/libigl/quadrants)
echo "[1/4] Installing system build tools..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git python3-venv python3-dev
else
    echo "Warning: apt-get not found. Ensure cmake, git, and a C++ compiler are installed."
fi

# 2. Setup Python virtual environment
echo "[2/4] Setting up Python virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 3. Clone and natively build 'quadrants' (fork of gstaichi by Genesis team)
echo "[3/4] Building 'quadrants' compiler from source (this may take a few minutes)..."
WORK_DIR=$(mktemp -d)
cd "$WORK_DIR"
git clone --depth 1 https://github.com/Genesis-Embodied-AI/quadrants.git
cd quadrants
# Install Python build dependencies
pip install scikit-build-core ninja pybind11
# Compile and install quadrants directly into our venv
pip install .
cd - > /dev/null
rm -rf "$WORK_DIR"

# 4. Install remaining gdog-sim dependencies normally
# Since build-essential/cmake are present, pip will automatically compile tetgen/libigl from source
echo "[4/4] Installing remaining gdog-sim dependencies..."
cd "$(dirname "$0")/.."
pip install -r requirements.txt

echo "=== Install Complete! ==="
echo "Run the simulator with:"
echo "source .venv/bin/activate"
echo "python main.py --render"
