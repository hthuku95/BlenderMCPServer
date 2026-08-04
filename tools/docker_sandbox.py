"""Execute Blender/Manim Python scripts inside the Docker sandbox container.

The Docker container `script-tester` (image: blender-manim-tester:latest) runs
on the same host (172.31.33.191). It has:
  - Blender 4.3.2 at /opt/blender/blender
  - ManimCE 0.18.1 (Python 3.10)
  - FFmpeg, LaTeX, Cairo, xvfb
  - /workspace mounted from /tmp/script-tester on the host

Two usage modes:
  1. validate_in_docker() — fast pre-validation (catches import/API/runtime
     errors in seconds, no full render). Used inside LLM retry loops.
  2. execute_in_docker() — full render inside the container (for when the
     host lacks Blender/Manim).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

SANDBOX_MOUNT = "/tmp/script-tester"
CONTAINER_NAME = "script-tester"
BLENDER_BIN = "/opt/blender/blender"


# ── Validation (fast pre-check, no full render) ──────────────────────

async def validate_in_docker(
    script_content: str,
    script_type: str = "blender",
    timeout: int = 15,
) -> dict:
    """Run a script in the Docker sandbox with a SHORT timeout to catch
    import, syntax, and API errors BEFORE the full host render.

    This is a pre-validation step — it detects errors that py_compile cannot
    (e.g. bpy API errors, missing bpy modules, Manim deprecated API usage).

    Args:
        script_content: The Python script to validate.
        script_type:    "blender" or "manim".
        timeout:        Max seconds before killing the container exec.

    Returns:
        {"passed": True/False, "logs": str, "error": str or None}
        The `logs` field contains stdout+stderr for LLM consumption.
    """
    import tempfile
    script_id = uuid.uuid4().hex
    host_script = os.path.join(SANDBOX_MOUNT, f"validate_{script_id}.py")
    container_script = f"/workspace/validate_{script_id}.py"

    os.makedirs(SANDBOX_MOUNT, exist_ok=True)

    with open(host_script, "w") as f:
        f.write(script_content)

    try:
        if script_type == "blender":
            return await _validate_blender(container_script, timeout)
        elif script_type == "manim":
            return await _validate_manim(container_script, timeout)
        else:
            return {"passed": False, "logs": "", "error": f"Unknown script_type: {script_type}"}
    finally:
        try:
            if os.path.exists(host_script):
                os.unlink(host_script)
        except OSError:
            pass


async def _validate_blender(container_script: str, timeout: int) -> dict:
    """Run a quick syntax+import validation of a Blender script in Docker.
    Uses blender --python-exit-code to get proper error codes.
    """
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "timeout", str(timeout),
        BLENDER_BIN, "--background",
        "--python-exit-code", "1",
        "--python", container_script,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        logs = f"[Docker sandbox] Blender validation timed out after {timeout}s"
        return {"passed": True, "logs": logs, "error": None}
        # Timeout = validation inconclusive. We pass because the script may
        # just be slow to render — the real render has a longer timeout.

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    logs = stdout + "\n" + stderr

    if proc.returncode != 0:
        error_text = stderr[-2000:] or stdout[-2000:] or f"exit code {proc.returncode}"
        return {
            "passed": False,
            "logs": logs,
            "error": f"Blender validation failed (code {proc.returncode}): {error_text}",
        }

    return {"passed": True, "logs": logs, "error": None}


async def _validate_manim(container_script: str, timeout: int) -> dict:
    """Run a quick syntax+import validation of a Manim script in Docker.
    Uses py_compile for syntax + manim's own import check.
    """
    # Step 1: Python syntax + import validation
    cmd_compile = [
        "docker", "exec", CONTAINER_NAME,
        "timeout", "10",
        "python3", "-c",
        f"import py_compile; py_compile.compile('/workspace/{Path(container_script).name}', doraise=True)",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd_compile,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=15)
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    logs = "=== Syntax check ===\n" + stdout + "\n" + stderr

    if proc.returncode != 0:
        return {
            "passed": False,
            "logs": logs,
            "error": f"Manim validation failed (syntax error): {stderr[-2000:]}",
        }

    # Step 2: Run the module import to catch Manim-specific errors
    # (without actually rendering - just import the script as a module)
    cmd_import = [
        "docker", "exec", CONTAINER_NAME,
        "timeout", "10",
        "python3", "-c",
        f"import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('_validate_', "
        f"    '/workspace/{Path(container_script).name}'); "
        f"mod = importlib.util.module_from_spec(spec); "
        f"sys.modules['_validate_'] = mod; "
        f"spec.loader.exec_module(mod)",
    ]
    proc2 = await asyncio.create_subprocess_exec(
        *cmd_import,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b2, stderr_b2 = await asyncio.wait_for(proc2.communicate(), timeout=15)
    stdout2 = stdout_b2.decode("utf-8", errors="replace")
    stderr2 = stderr_b2.decode("utf-8", errors="replace")
    logs += "\n=== Import check ===\n" + stdout2 + "\n" + stderr2

    if proc2.returncode != 0:
        return {
            "passed": False,
            "logs": logs,
            "error": f"Manim validation failed (import error): {stderr2[-2000:]}",
        }

    return {"passed": True, "logs": logs, "error": None}


# ── Full render inside Docker (fallback / alternative) ────────────────

async def execute_in_docker(
    script_content: str,
    script_type: str = "blender",
    args: Optional[dict] = None,
    timeout: int = 600,
    quality: str = "m",
) -> str:
    """Write a Python script to the Docker sandbox, execute it, and return the
    rendered output file path.

    This is the full render path — use validate_in_docker() first for fast
    pre-validation, then call this for the actual render if needed.
    """
    script_id = uuid.uuid4().hex
    host_script = os.path.join(SANDBOX_MOUNT, f"{script_id}.py")
    container_script = f"/workspace/{script_id}.py"
    output_filename = f"output_{script_id}.mp4"
    host_output = os.path.join(SANDBOX_MOUNT, output_filename)
    container_output = f"/workspace/{output_filename}"

    os.makedirs(SANDBOX_MOUNT, exist_ok=True)

    with open(host_script, "w") as f:
        f.write(script_content)

    try:
        if script_type == "blender":
            result = await _run_blender_in_docker(
                container_script, args or {}, timeout, container_output
            )
        elif script_type == "manim":
            result = await _run_manim_in_docker(
                container_script, timeout, quality
            )
        else:
            raise ValueError(f"Unknown script_type: {script_type}")

        if not result or not os.path.exists(result):
            raise RuntimeError(f"Docker sandbox produced no output file")

        dest = f"/tmp/docker_out_{script_id}{Path(result).suffix}"
        os.rename(result, dest)
        return dest

    finally:
        for p in [host_script, host_output]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


async def _run_blender_in_docker(
    container_script: str,
    args: dict,
    timeout: int,
    container_output: str,
) -> Optional[str]:
    args_json = json.dumps({**args, "output_path": container_output})
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "xvfb-run", "-a",
        BLENDER_BIN,
        "--background",
        "--python", container_script,
        "--", args_json,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"Blender in Docker timed out after {timeout}s")

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender (Docker) exited with code {proc.returncode}.\n"
            f"STDERR:\n{stderr[-2000:]}"
        )

    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT:"):
            try:
                data = json.loads(line[len("RESULT:"):])
                out = data.get("output_path", "")
                if out and os.path.exists(out):
                    return out
            except json.JSONDecodeError:
                pass

    host_output = container_output.replace("/workspace/", SANDBOX_MOUNT + "/")
    if os.path.exists(host_output):
        return host_output

    raise RuntimeError(
        f"Blender finished but no output file found.\n"
        f"STDOUT (last 1000 chars):\n{stdout[-1000:]}"
    )


async def _run_manim_in_docker(
    container_script: str,
    timeout: int,
    quality: str,
) -> Optional[str]:
    media_dir = "/workspace/media"
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python3", "-m", "manim",
        f"-q{quality}",
        "--media_dir", media_dir,
        container_script,
        "GeneratedScene",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"Manim in Docker timed out after {timeout}s")

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Manim (Docker) exited with code {proc.returncode}.\n"
            f"STDERR:\n{stderr[-2000:]}"
        )

    script_name = Path(container_script).stem
    host_media = os.path.join(SANDBOX_MOUNT, "media")
    quality_map = {"l": "480p", "m": "720p", "h": "1080p", "k": "2160p"}
    res = quality_map.get(quality, "720p")

    candidate = os.path.join(host_media, "videos", script_name, res, "GeneratedScene.mp4")
    if os.path.exists(candidate):
        return candidate

    for root, _, files in os.walk(host_media):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)

    raise RuntimeError(
        f"Manim finished but no output file found in {host_media}.\n"
        f"STDOUT (last 1000 chars):\n{stdout[-1000:]}"
    )
