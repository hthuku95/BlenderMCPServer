import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from tools.code_guards import apply_all as _apply_script_guards

BLENDER_BIN = os.getenv("BLENDER_BIN", "blender")

# GPU rendering: when the host has an NVIDIA GPU (g4dn fleet), enable Cycles
# GPU (OptiX preferred, CUDA fallback) and force Cycles as the engine. On a
# CPU-only box the probe finds no devices and everything falls back to the
# previous behavior. Kill switch: BLENDER_GPU_RENDER=off.
# NOTE: prologue must come FIRST in the script so the config applies before
# any model-written render call; a script may still override the engine
# afterwards (EEVEE also uses the GPU via EGL when a driver is present).
_GPU_RENDER_ENABLED = os.getenv("BLENDER_GPU_RENDER", "auto").lower() not in ("off", "false", "0", "no")

_GPU_PROLOGUE = r'''
# ==== harness-gpu-prologue (prepended by BlenderMCPServer) ====
import bpy as _h_bpy

def _h_gpu_setup():
    try:
        _prefs = _h_bpy.context.preferences.addons.get('cycles')
        if not _prefs:
            print("HARNESS_GPU: cycles addon unavailable")
            return
        _cp = _prefs.preferences
        _chosen = None
        for _dtype in ('OPTIX', 'CUDA'):
            try:
                _cp.compute_device_type = _dtype
                _cp.get_devices()
                if any(d.type == _dtype for d in _cp.devices):
                    _chosen = _dtype
                    break
            except Exception:
                continue
        if not _chosen:
            print("HARNESS_GPU_DISABLED no OptiX/CUDA device")
            return
        for _d in _cp.devices:
            _d.use = (_d.type == _chosen)
        _scn = _h_bpy.context.scene
        _scn.render.engine = 'CYCLES'
        _scn.cycles.device = 'GPU'
        # Denoiser kernels reload per frame on driver 580/T4 (measured 10.4s vs
        # 2.3s per 1080p frame, Aug 30 2026) — disable; simple scenes at 64+
        # samples render clean without it.
        try:
            _scn.cycles.use_denoising = False
        except Exception:
            pass
        _gpu_names = [d.name for d in _cp.devices if d.type == _chosen and d.use]
        print("HARNESS_GPU_ENABLED type=%s devices=%s" % (_chosen, _gpu_names))
    except Exception as _e:
        print("HARNESS_GPU_ERROR %s" % _e)

_h_gpu_setup()
del _h_gpu_setup
'''

_RENDER_EPILOGUE = r'''
# ==== harness-render-epilogue (appended by BlenderMCPServer) ====
# Deterministically forces a render to the harness-requested output_path so a
# successful exit always yields a media file, regardless of whether the model
# remembered to call the render operator / honour sys.argv args.
import bpy, json, os, sys as _sys
_me = __file__
_hargs = {}
for _cand in (_sys.argv[-1], _sys.argv[-2]):
    if _cand and os.path.exists(_cand):
        try:
            _maybe = json.load(open(_cand))
            if isinstance(_maybe, dict):
                _hargs = _maybe; break
        except Exception:
            pass
_out = _hargs.get("output_path", "/tmp/bpy_render.mp4")
_hdir = os.path.dirname(_out)
if _hdir:
    os.makedirs(_hdir, exist_ok=True)
_scn = bpy.context.scene
_scn.render.resolution_x = 1920
_scn.render.resolution_y = 1080
_scn.render.fps = 60
_scn.render.image_settings.file_format = 'FFMPEG'
try:
    _scn.render.ffmpeg.format = 'MPEG4'
    _scn.render.ffmpeg.codec = 'H264'
except Exception:
    pass
if not any(o.type == 'CAMERA' for o in bpy.context.scene.objects):
    _cd = bpy.data.cameras.new("HarnessCam")
    _cam = bpy.data.objects.new("HarnessCam", _cd)
    _scn.collection.objects.link(_cam)
    _cam.location = (0, -6, 3)
    _cam.rotation_euler = (1.2, 0, 0)
    _scn.camera = _cam
if not any(o.type == 'LIGHT' for o in bpy.context.scene.objects):
    _ld = bpy.data.lights.new("HarnessLight", type='POINT')
    _lt = bpy.data.objects.new("HarnessLight", _ld)
    _scn.collection.objects.link(_lt)
    _lt.location = (2, -3, 4)
_scn.render.filepath = _out
try:
    bpy.ops.render.render(animation=True, write_still=False)
except Exception as _e:
    bpy.ops.render.render(write_still=True)
_res = {"output_path": _out, "success": True}
_rf = os.environ.get("BLENDER_RESULT_FILE")
if _rf:
    with open(_rf, "w") as _f:
        _f.write(json.dumps(_res))
'''


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
    script_content = _apply_script_guards(script_content)
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

    try:
        _existing = Path(script_path).read_text()
    except OSError:
        _existing = ""
    if "harness-gpu-prologue" not in _existing and _GPU_RENDER_ENABLED:
        with open(script_path, "w") as _f:
            _f.write(_GPU_PROLOGUE + "\n" + _existing)
    _existing = Path(script_path).read_text()
    if "harness-render-epilogue" not in _existing:
        with open(script_path, "a") as _f:
            _f.write("\n" + _RENDER_EPILOGUE + "\n")

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
