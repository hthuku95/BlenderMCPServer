FROM python:3.11-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Blender headless
    blender \
    # OpenGL (required by Blender even in --background mode)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    # Virtual framebuffer (needed for Blender on headless servers)
    xvfb \
    # Minimal TeX Live for Manim (Phase 3) — ~600MB, not full 4GB
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    dvisvgm \
    dvipng \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY . /app

# Blender renders to /tmp by default; ensure it exists
RUN mkdir -p /tmp/blender_renders

# Expose MCP + REST port
EXPOSE 8000

# Start with xvfb-run so Blender has a virtual display
CMD ["xvfb-run", "-a", "python", "server.py"]
