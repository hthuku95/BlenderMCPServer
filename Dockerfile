FROM python:3.11-bookworm

# ---- System dependencies ----
# python:3.11-bookworm = Debian 12 (Bookworm) full image.
# Using full (non-slim) to ensure all system libs are available for blender.
# We ADD contrib + non-free without touching the existing sources.list so
# security + updates repos remain intact.
RUN echo "deb http://deb.debian.org/debian bookworm contrib non-free non-free-firmware" \
        > /etc/apt/sources.list.d/bookworm-nonfree.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # Blender headless
        blender \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # EGL / offscreen rendering for CYCLES headless
        libegl1 \
        # Virtual framebuffer (Blender needs a display even in --background mode)
        xvfb \
        # xvfb-run requires xauth to set up X11 authorization cookies
        xauth \
        # TeX Live — latex-base + fonts-extra covers all Manim default template packages
        # (tipa, mathrsfs, calligra, wasysym, dsfont required by Manim's preamble)
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-science \
        texlive-fonts-extra \
        dvisvgm \
        dvipng \
        # Cairo + Pango — runtime libraries for cairosvg and Manim
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        # Cairo + Pango dev headers — required to compile pycairo and manim (no wheel)
        libcairo2-dev \
        libpango1.0-dev \
        pkg-config \
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

# Start uvicorn server directly — xvfb is invoked per-render in blender_runner.py
CMD ["python", "server.py"]
