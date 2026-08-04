import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

BLENDER_BIN = os.getenv("BLENDER_BIN", "blender")


async def run_blender_script(
    script_path: str | Path,
    args: dict | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_blender_sync, str(script_path), args, timeout,
    )


async def run_blender_script_with_retry(
    script_content: str,
    args: dict | None = None,
    max_attempts: int = 3,
    timeout: int = 600,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        tmp = Path(tempfile.mkdtemp(prefix="bpy_retry_" + str(attempt) + "_")) / "script.py"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(script_content)
        try:
            return await run_blender_script(tmp, args, timeout=timeout)
        except RuntimeError as exc:
            last_exc = exc
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    raise RuntimeError("Blender script failed after " + str(max_attempts) + " attempt(s)") from last_exc


def _run_blender_sync(script_path: str, args: dict | None, timeout: int) -> dict[str, Any]:
    result_file = "/tmp/blender_result_" + uuid.uuid4().hex + ".json"
    args_file = None

    if args:
        args_file = "/tmp/blender_args_" + uuid.uuid4().hex + ".json"
        with open(args_file, "w") as f:
            json.dump(args, f)

    cmd = [BLENDER_BIN, "--background", "--python", script_path]

    if args_file:
        cmd.extend(["--", args_file])

    env = os.environ.copy()
    env["BLENDER_RESULT_FILE"] = result_file

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Blender script timed out after " + str(timeout) + "s")
    except FileNotFoundError:
        raise RuntimeError(
            "Blender binary  + BLENDER_BIN +  not found. "
            "Install Blender or set the BLENDER_BIN env var."
        )

    stderr_text = completed.stderr.decode() if completed.stderr else ""
    stdout_text = completed.stdout.decode() if completed.stdout else ""

    output_path_hint = ""
    if args and "output_path" in args:
        output_path_hint = args["output_path"]

    if completed.returncode != 0:
        err_detail = stderr_text[-2000:] or stdout_text[-2000:] or ("exit code " + str(completed.returncode))
        if result_file and os.path.exists(result_file):
            os.unlink(result_file)
        if args_file and os.path.exists(args_file):
            os.unlink(args_file)
        raise RuntimeError(
            "Blender exited with code " + str(completed.returncode) + ": " + err_detail
        )

    if output_path_hint and not os.path.exists(output_path_hint) and not os.path.exists(result_file):
        err_detail = stderr_text[-2000:] or stdout_text[-2000:] or "no output file and no result file"
        if result_file and os.path.exists(result_file):
            os.unlink(result_file)
        if args_file and os.path.exists(args_file):
            os.unlink(args_file)
        raise RuntimeError(
            "Blender script produced no output:  " + err_detail
        )

    result: dict[str, Any] = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        os.unlink(result_file)
    elif args and "output_path" in args:
        result["output_path"] = args["output_path"]

    if args_file and os.path.exists(args_file):
        os.unlink(args_file)

    return result
