FROM python:3.11-slim

# ---- System dependencies ----
# python:3.11-slim is Debian Bookworm.
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
        # Cairo + Pango — required by cairosvg and Manim
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
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
