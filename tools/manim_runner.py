"""
Manim subprocess runner — generates LaTeX/math animation clips.

Manim is invoked as a CLI subprocess so the scene has full access to
the installed LaTeX distribution. Args are passed to the scene via the
LATEX_SCENE_ARGS environment variable (JSON string).
"""
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path


async def run_manim_scene(
    scene_file: str,
    scene_class: str,
    args: dict,
    quality: str = "m",  # l=480p fast, m=720p, h=1080p slow, k=4K
    output_path: str | None = None,
    timeout: int = 300,
) -> str:
    """
    Render a Manim scene and return the path to the MP4.

    Args:
        scene_file:   Absolute path to the .py file with the Scene subclass.
        scene_class:  Name of the Scene subclass (e.g. "LatexScene").
        args:         Dict injected into the scene via LATEX_SCENE_ARGS env var.
        quality:      Manim quality flag (l/m/h/k).
        output_path:  If given, move the rendered file here and return that path.
        timeout:      Max seconds before we kill Manim.

    Returns:
        Absolute path to the rendered MP4.
    """
    env = {**os.environ, "LATEX_SCENE_ARGS": json.dumps(args)}

    with tempfile.TemporaryDirectory(prefix="manim_work_") as work_dir:
        cmd = [
            "python3", "-m", "manim",
            f"-q{quality}",
            "--media_dir", work_dir,
            "--output_file", scene_class,   # predictable filename
            scene_file,
            scene_class,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(scene_file).parent),
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(f"Manim timed out after {timeout}s")

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(
                f"Manim failed (exit {proc.returncode}).\n"
                f"STDERR (last 2000 chars):\n{stderr[-2000:]}"
            )

        rendered = _find_rendered_mp4(work_dir, scene_class)
        if not rendered:
            raise RuntimeError(
                f"Manim finished but no MP4 found under {work_dir}.\n"
                f"STDOUT:\n{stdout[-1000:]}"
            )

        # Move out of temp dir before it's deleted
        if output_path:
            shutil.move(rendered, output_path)
            return output_path

        stable = f"/tmp/manim_{scene_class}_{os.getpid()}.mp4"
        shutil.move(rendered, stable)
        return stable


def _find_rendered_mp4(media_dir: str, scene_class: str) -> str | None:
    """Walk media_dir and return the first .mp4 matching scene_class."""
    for root, _, files in os.walk(media_dir):
        for f in files:
            if f == f"{scene_class}.mp4" or (f.endswith(".mp4") and scene_class in f):
                return os.path.join(root, f)
    # Fallback: any .mp4
    for root, _, files in os.walk(media_dir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None
