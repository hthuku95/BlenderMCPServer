"""Headless Blender subprocess executor with error capture and LLM-assisted retry."""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path


BLENDER_BIN = os.getenv("BLENDER_BIN", "blender")


def _syntax_check(script_path: str) -> None:
    """Raise ValueError if the script has a Python syntax error."""
    result = subprocess.run(
        ["python3", "-m", "py_compile", script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Script syntax error: {result.stderr}")


async def run_blender_script(
    script_path: str,
    args: dict,
    timeout: int = 600,
) -> dict:
    """
    Execute a bpy script in headless Blender.

    The script must print a line starting with 'RESULT:' followed by JSON.
    Example: print(f"RESULT:{json.dumps({'video_url': '...'})}")

    Returns the parsed RESULT dict on success; raises RuntimeError on failure.
    """
    _syntax_check(script_path)

    args_json = json.dumps(args)
    # Use xvfb-run to provide a virtual display for Blender (required even in --background mode)
    proc = await asyncio.create_subprocess_exec(
        "xvfb-run", "-a",
        BLENDER_BIN,
        "--background",
        "--python", script_path,
        "--", args_json,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"Blender timed out after {timeout}s")

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        # Surface the last 2000 chars of stderr for LLM debugging
        raise RuntimeError(
            f"Blender exited with code {proc.returncode}.\n"
            f"STDERR (last 2000 chars):\n{stderr[-2000:]}"
        )

    # Extract the RESULT line
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT:"):
            try:
                return json.loads(line[len("RESULT:"):])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Could not parse RESULT JSON: {exc}\nLine: {line}")

    raise RuntimeError(
        f"Blender finished but printed no RESULT line.\n"
        f"STDOUT (last 1000 chars):\n{stdout[-1000:]}"
    )


async def run_blender_script_with_retry(
    script_content: str,
    args: dict,
    max_attempts: int = 3,
    timeout: int = 600,
    fix_fn=None,
) -> dict:
    """
    Write script_content to a temp file, run it, and retry with LLM-assisted
    fixes on failure.  fix_fn(script_content, error_msg) -> fixed_script_content.
    """
    current_script = script_content
    last_error = None

    for attempt in range(max_attempts):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="bpy_render_"
        ) as tmp:
            tmp.write(current_script)
            tmp_path = tmp.name

        try:
            result = await run_blender_script(tmp_path, args, timeout=timeout)
            return result
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if fix_fn is not None and attempt < max_attempts - 1:
                current_script = fix_fn(current_script, last_error)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    raise RuntimeError(
        f"Blender failed after {max_attempts} attempts. Last error:\n{last_error}"
    )
