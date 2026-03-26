FROM python:3.11-slim-bookworm

# ---- System dependencies ----
# python:3.11-slim-bookworm = Debian 12 (Bookworm) slim.
# We ADD contrib + non-free repos without overwriting the existing sources.list
# (overwriting loses the security + updates repos that the slim image needs).
# blender and ffmpeg require contrib/non-free for full codec support.
RUN sed -i 's/ main/ main contrib non-free non-free-firmware/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # Blender headless
        blender \
        # OpenGL — libgl1 replaces libgl1-mesa-glx in Bookworm
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # EGL / offscreen rendering for CYCLES headless
        libegl1 \
        # Virtual framebuffer (Blender needs a display even in --background mode)
        xvfb \
        # Minimal TeX Live for Manim
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        dvisvgm \
        dvipng \
        # Cairo + Pango — runtime libraries for cairosvg and Manim
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        # Cairo dev headers + build tools — needed to compile pycairo (no wheel)
        libcairo2-dev \
        pkg-config \
        gcc \
        g++ \
        python3-dev \
        # FFmpeg — used by MoviePy and Manim
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
