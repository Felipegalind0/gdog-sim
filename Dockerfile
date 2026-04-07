FROM python:3.11-slim

# Install system dependencies (running as root in container avoids the need for sudo)
# We include GL/GLIB libraries often required by rendering frameworks like OpenCV and Genesis.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip wheel setuptools scikit-build-core ninja pybind11

# Clone and compile 'quadrants' from source for ARM64/aarch64 support
RUN git clone --depth 1 https://github.com/Genesis-Embodied-AI/quadrants.git /tmp/quadrants && \
    cd /tmp/quadrants && \
    pip install . && \
    rm -rf /tmp/quadrants

# Copy requirements and install (will auto-compile tetgen/libigl from source since cmake is present)
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Expose the FastAPI backend port
EXPOSE 8000

# Default command. If running on a DGX cluster, you may want to pass --quick-tunnel for remote access.
CMD ["python", "main.py", "--render", "--host", "0.0.0.0"]
