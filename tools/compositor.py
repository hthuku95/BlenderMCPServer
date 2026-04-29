"""
Option B Compositor — merge a transparent Manim equation clip over a Blender scene.

MoviePy CompositeVideoClip stacks:
  - Bottom layer: Blender-rendered scene (opaque MP4)
  - Top layer:   Manim equation (transparent-background .mov / .webm / .mp4 with alpha)

The equation is centred horizontally and placed at `eq_y_position` (0.0 = top, 1.0 = bottom).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH")
    return path


def _probe_duration(video_path: str) -> float:
    ffprobe = _require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def composite_manim_over_blender(
    blender_video_path: str,
    equation_video_path: str,
    output_path: str | None = None,
    eq_x_position: float = 0.5,   # 0.0=left … 1.0=right (relative to width)
    eq_y_position: float = 0.5,   # 0.0=top  … 1.0=bottom (relative to height)
    eq_scale: float = 1.0,        # Scale factor for the equation clip
    fps: int = 60,
) -> str:
    """
    Composite a Manim equation clip over a Blender scene.

    Args:
        blender_video_path:  Path to the opaque Blender-rendered scene MP4.
        equation_video_path: Path to the Manim equation clip (transparent background).
        output_path:         Destination for the composited MP4.
                             Defaults to <blender_video_path>_composited.mp4.
        eq_x_position:       Horizontal centre of the equation (0.0–1.0).
        eq_y_position:       Vertical centre of the equation (0.0–1.0).
        eq_scale:            Scale the equation relative to its natural size.
        fps:                 Output frame rate.

    Returns: Absolute path to the composited output MP4.
    Raises:  RuntimeError on failure.
    """
    if not os.path.exists(blender_video_path):
        raise FileNotFoundError(f"Blender video not found: {blender_video_path}")
    if not os.path.exists(equation_video_path):
        raise FileNotFoundError(f"Equation video not found: {equation_video_path}")

    if output_path is None:
        stem = Path(blender_video_path).stem
        output_path = str(Path(blender_video_path).parent / f"{stem}_composited.mp4")

    ffmpeg = _require_binary("ffmpeg")
    scene_duration = _probe_duration(blender_video_path)
    overlay_filter = (
        f"[1:v]scale=iw*{eq_scale}:ih*{eq_scale}[eq];"
        f"[0:v][eq]overlay="
        f"x='W*{eq_x_position}-overlay_w/2':"
        f"y='H*{eq_y_position}-overlay_h/2':"
        f"shortest=1,format=yuv420p[v]"
    )
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", blender_video_path,
            "-stream_loop", "-1",
            "-i", equation_video_path,
            "-filter_complex", overlay_filter,
            "-map", "[v]",
            "-an",
            "-r", str(fps),
            "-t", str(scene_duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg composite failed.\n"
            f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg ran but produced no composited output file")

    return os.path.abspath(output_path)


def add_audio_to_video(video_path: str, audio_path: str, output_path: str | None = None) -> str:
    """
    Mux an audio track into a video file (for future use with TTS / music).

    Args:
        video_path:  Input silent video MP4.
        audio_path:  Audio file (MP3, WAV, AAC).
        output_path: Destination MP4. Defaults to <video_path>_audio.mp4.

    Returns: Absolute path to the output MP4 with audio.
    """
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_audio.mp4")

    ffmpeg = _require_binary("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter:a", "apad",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio mux failed.\n"
            f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
        )

    return os.path.abspath(output_path)
