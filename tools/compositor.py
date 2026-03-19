"""
Option B Compositor — merge a transparent Manim equation clip over a Blender scene.

MoviePy CompositeVideoClip stacks:
  - Bottom layer: Blender-rendered scene (opaque MP4)
  - Top layer:   Manim equation (transparent-background .mov / .webm / .mp4 with alpha)

The equation is centred horizontally and placed at `eq_y_position` (0.0 = top, 1.0 = bottom).
"""
from __future__ import annotations

import os
from pathlib import Path


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
    try:
        from moviepy.editor import VideoFileClip, CompositeVideoClip
    except ImportError as exc:
        raise RuntimeError(
            "moviepy is not installed. Run: pip install moviepy"
        ) from exc

    if not os.path.exists(blender_video_path):
        raise FileNotFoundError(f"Blender video not found: {blender_video_path}")
    if not os.path.exists(equation_video_path):
        raise FileNotFoundError(f"Equation video not found: {equation_video_path}")

    if output_path is None:
        stem = Path(blender_video_path).stem
        output_path = str(Path(blender_video_path).parent / f"{stem}_composited.mp4")

    # Load clips
    scene_clip = VideoFileClip(blender_video_path)
    eq_clip = VideoFileClip(equation_video_path, has_mask=True)

    # Scale if requested
    if eq_scale != 1.0:
        eq_clip = eq_clip.resize(eq_scale)

    # Trim equation to scene duration (loop if shorter, trim if longer)
    if eq_clip.duration < scene_clip.duration:
        eq_clip = eq_clip.loop(duration=scene_clip.duration)
    else:
        eq_clip = eq_clip.subclip(0, scene_clip.duration)

    # Pixel position from relative coords
    scene_w, scene_h = scene_clip.size
    eq_w, eq_h = eq_clip.size
    px = int(scene_w * eq_x_position - eq_w / 2)
    py = int(scene_h * eq_y_position - eq_h / 2)
    eq_clip = eq_clip.set_position((px, py))

    final = CompositeVideoClip([scene_clip, eq_clip], size=scene_clip.size)
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None,
    )

    scene_clip.close()
    eq_clip.close()
    final.close()

    if not os.path.exists(output_path):
        raise RuntimeError("MoviePy ran but produced no output file")

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
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError as exc:
        raise RuntimeError("moviepy is not installed") from exc

    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_audio.mp4")

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).subclip(0, video.duration)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    video.close()
    audio.close()

    return os.path.abspath(output_path)
