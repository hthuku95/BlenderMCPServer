"""FFmpeg audio/video utilities used by vibevoice.py narration pipeline."""

import os
import shutil
import subprocess
from pathlib import Path


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH")
    return path


def add_audio_to_video(video_path: str, audio_path: str, output_path: str | None = None) -> str:
    """Mux an audio track into a video file.

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
