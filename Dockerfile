FROM python:3.11-slim-bookworm

# ---- System dependencies ----
# python:3.11-slim-bookworm is pinned to Debian Bookworm (12).
# NOTE: python:3.11-slim (unversioned) now resolves to Trixie (13), which
# causes apt dependency conflicts when we force bookworm repos.
# Always use the -bookworm suffix to ensure a stable, consistent base.
#   • libgl1-mesa-glx was renamed to libgl1 in Bookworm
#   • blender needs the non-free-firmware component enabled
#   • cairosvg requires libcairo2 + libpango at runtime
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # Blender headless (requires contrib/non-free on Bookworm)
        blender \
        # OpenGL — libgl1 replaces libgl1-mesa-glx in Bookworm
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # EGL / offscreen rendering for CYCLES headless
        libegl1 \
        # Virtual framebuffer (Blender needs a display even in --background mode)
        xvfb \
        # Minimal TeX Live for Manim (NOT the full 4 GB meta-package)
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        dvisvgm \
        dvipng \
        # Cairo + Pango — runtime libraries for cairosvg and Manim
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        # Cairo dev headers — required to build pycairo from source (no wheel)
        libcairo2-dev \
        pkg-config \
        # Build tools — needed for pycairo and any other C-extension pip packages
        gcc \
        g++ \
        python3-dev \
        # FFmpeg — used by MoviePy and Manim video rendering
        ffmpeg \
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
