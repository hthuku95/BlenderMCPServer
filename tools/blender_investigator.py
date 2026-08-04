"""
blender_investigator.py — VIGA Verifier investigator tools for BlenderMCP.

These are bpy scripts executed via blender --background to inspect and
manipulate scenes for the Verifier agent. The Verifier uses these tools
to investigate 3D scenes, check object visibility, adjust viewpoints,
and gather scene information for quality assessment.

Each function returns a JSON-serializable dict with results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional


def _build_script(inline_code: str) -> str:
    """Wrap bpy code with headless Blender boilerplate."""
    return f"""\
import bpy, json

try:
    {inline_code}
except Exception as e:
    print(f"INVESTIGATOR_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
"""


def _run_investigator_script(code: str, timeout: int = 60) -> dict:
    """Run a short bpy investigation script on the currently open .blend file.

    Reads the .blend filepath from BLENDER_CURRENT_FILE env var.
    """
    blend_path = os.environ.get("BLENDER_CURRENT_FILE", "")
    if not blend_path or not os.path.exists(blend_path):
        return {"error": "BLENDER_CURRENT_FILE not set or file not found"}

    with tempfile.NamedTemporaryFile(
        suffix=".py", prefix="investigator_", delete=False, mode="w"
    ) as f:
        f.write(_build_script(code))
        script_path = f.name

    try:
        result = subprocess.run(
            ["blender", blend_path, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Try to parse JSON output from the script
        data = {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}
        for line in stdout.split("\n"):
            line = line.strip()
            if line.startswith("INVESTIGATOR_RESULT:"):
                try:
                    payload = json.loads(line[len("INVESTIGATOR_RESULT:"):])
                    data.update(payload)
                except json.JSONDecodeError:
                    pass
            if line.startswith("INVESTIGATOR_ERROR:"):
                data["error"] = line[len("INVESTIGATOR_ERROR:"):].strip()

        return data
    except subprocess.TimeoutExpired:
        return {"error": f"Investigator script timed out after {timeout}s"}
    except FileNotFoundError:
        return {"error": "Blender executable not found in PATH"}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ── Public Tool Functions ────────────────────────────────────────────────


async def initialize_viewpoint(scene_name: str = "Scene") -> dict:
    """Reset the viewport to a default angle (top, front, side) and return current state.

    This is called before investigation to ensure a known starting viewpoint.
    """
    code = f"""\
scene = bpy.data.scenes.get('{scene_name}') or bpy.context.scene
bpy.context.window.scene = scene

# Get camera if it exists, otherwise report
cam_obj = None
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        cam_obj = obj
        break

result = {{
    "has_camera": cam_obj is not None,
    "total_objects": len(bpy.data.objects),
    "object_types": {{}},
    "scene_name": scene.name,
}}
if cam_obj:
    result["camera_location"] = list(cam_obj.location)
    result["camera_rotation"] = list(cam_obj.rotation_euler)
    result["camera_name"] = cam_obj.name

# Count by type
for obj in bpy.data.objects:
    t = obj.type
    result["object_types"][t] = result["object_types"].get(t, 0) + 1

# Also check render settings
result["render_engine"] = scene.render.engine
result["resolution_x"] = scene.render.resolution_x
result["resolution_y"] = scene.render.resolution_y
result["frame_end"] = scene.frame_end
result["fps"] = scene.render.fps

print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
"""
    return _run_investigator_script(code)


async def get_scene_info(scene_name: str = "Scene") -> dict:
    """Get comprehensive info about the current scene: objects, materials, lights, animation."""
    code = f"""\
scene = bpy.data.scenes.get('{scene_name}') or bpy.context.scene
bpy.context.window.scene = scene

info = {{
    "scene_name": scene.name,
    "total_objects": len(bpy.data.objects),
    "total_materials": len(bpy.data.materials),
    "total_meshes": len(bpy.data.meshes),
    "total_cameras": len(bpy.data.cameras),
    "total_lights": len(bpy.data.lights),
    "object_types": {{}},
    "objects": [],
    "materials": [],
    "render_engine": scene.render.engine,
    "resolution_x": scene.render.resolution_x,
    "resolution_y": scene.render.resolution_y,
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
    "current_frame": scene.frame_current,
    "fps": scene.render.fps,
}}

for obj in bpy.data.objects:
    t = obj.type
    info["object_types"][t] = info["object_types"].get(t, 0) + 1
    obj_info = {{
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "visible": obj.visible_get(),
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "parent": obj.parent.name if obj.parent else None,
        "material_count": len(obj.data.materials) if hasattr(obj, "data") and hasattr(obj.data, "materials") else 0,
    }}
    # Check if animated (has keyframes)
    if obj.animation_data and obj.animation_data.action:
        obj_info["animated"] = True
        obj_info["action_name"] = obj.animation_data.action.name
    else:
        obj_info["animated"] = False
    info["objects"].append(obj_info)

for mat in bpy.data.materials:
    info["materials"].append({{
        "name": mat.name,
        "use_nodes": mat.use_nodes,
    }})

print(f"INVESTIGATOR_RESULT:{{json.dumps(info, default=str)}}")
"""
    return _run_investigator_script(code)


async def set_viewpoint(
    camera_name: Optional[str] = None,
    location: Optional[list[float]] = None,
    rotation: Optional[list[float]] = None,
    target_object: Optional[str] = None,
) -> dict:
    """Reposition the active camera or a specific camera to a new viewpoint.

    Args:
        camera_name: Name of the camera to move. If None, use active camera.
        location: [x, y, z] camera position in world space.
        rotation: [rx, ry, rz] Euler rotation in radians.
        target_object: Object name to look at. Camera will point at this object
                       (overrides rotation if both provided).
    """
    cam_var = f'bpy.data.objects["{camera_name}"]' if camera_name else 'bpy.context.scene.camera'

    pos_code = ""
    if location:
        pos_code += f"""{cam_var}.location = {json.dumps(location)}
"""

    rot_code = ""
    if rotation:
        rot_code += f"""import math
{cam_var}.rotation_euler = {json.dumps(rotation)}
"""

    target_code = ""
    if target_object:
        target_code = f"""\
import mathutils
target = bpy.data.objects.get("{target_object}")
if target:
    direction = target.location - {cam_var}.location
    {cam_var}.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
"""

    code = f"""\
import bpy, json, math

{cam_var} = {cam_var.split("=")[1].strip() if "=" in cam_var.split(chr(10))[0] else cam_var}

{pos_code}{rot_code}{target_code}

result = {{
    "camera": {cam_var}.name,
    "location": list({cam_var}.location),
    "rotation": list({cam_var}.rotation_euler),
}}
print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
"""
    return _run_investigator_script(code)


async def toggle_visibility(
    object_name: str,
    hide: Optional[bool] = None,
    hide_render: Optional[bool] = None,
) -> dict:
    """Show or hide an object in the viewport and/or render.

    Args:
        object_name: Name of the object to toggle.
        hide: If True/False, set viewport visibility. If None, toggle.
        hide_render: If True/False, set render visibility. If None, don't change.
    """
    hide_code = ""
    if hide is not None:
        hide_code += f'obj.hide_viewport = {str(hide).lower()}\n'
    else:
        hide_code += 'obj.hide_viewport = not obj.hide_viewport\n'

    if hide_render is not None:
        hide_code += f'obj.hide_render = {str(hide_render).lower()}\n'

    code = f"""\
import bpy, json
obj = bpy.data.objects.get("{object_name}")
if not obj:
    result = {{"error": "Object '{object_name}' not found"}}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
else:
    {hide_code}
    result = {{
        "object": obj.name,
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "visible": obj.visible_get(),
    }}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
"""
    return _run_investigator_script(code)


async def set_keyframe(
    object_name: str,
    frame: int,
    location: Optional[list[float]] = None,
    rotation: Optional[list[float]] = None,
    scale: Optional[list[float]] = None,
) -> dict:
    """Set a keyframe on an object's transform at a specific frame.

    Args:
        object_name: Name of the object to keyframe.
        frame: Frame number.
        location: [x, y, z] position.
        rotation: [rx, ry, rz] Euler rotation in radians.
        scale: [sx, sy, sz] scale.
    """
    transforms = []
    if location:
        transforms.append(f"obj.location = {json.dumps(location)}; obj.keyframe_insert(data_path='location', frame={frame})")
    if rotation:
        transforms.append(f"obj.rotation_euler = {json.dumps(rotation)}; obj.keyframe_insert(data_path='rotation_euler', frame={frame})")
    if scale:
        transforms.append(f"obj.scale = {json.dumps(scale)}; obj.keyframe_insert(data_path='scale', frame={frame})")

    transforms_code = "\n    ".join(transforms)

    code = f"""\
import bpy, json
obj = bpy.data.objects.get("{object_name}")
if not obj:
    result = {{"error": "Object '{object_name}' not found"}}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
else:
    bpy.context.scene.frame_set({frame})
    {transforms_code}
    result = {{
        "object": obj.name,
        "frame": {frame},
        "location": list(obj.location),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
    }}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
"""
    return _run_investigator_script(code)


async def investigate_object(
    object_name: str,
    detailed: bool = False,
) -> dict:
    """Get detailed info about a specific object: materials, modifiers, constraints, data.

    Args:
        object_name: Name of the object.
        detailed: If True, include mesh data summary (vertex count, face count, etc.).
    """
    detailed_code = ""
    if detailed:
        detailed_code = """\
if obj.type == 'MESH' and obj.data:
    mesh = obj.data
    obj_info["mesh_data"] = {
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "uv_layers": len(mesh.uv_layers),
        "vertex_colors": len(mesh.vertex_colors),
        "materials_in_slots": len(mesh.materials),
    }
"""

    code = f"""\
import bpy, json
obj = bpy.data.objects.get("{object_name}")
if not obj:
    result = {{"error": "Object '{object_name}' not found"}}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(result)}}")
else:
    obj_info = {{
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "visible": obj.visible_get(),
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "parent": obj.parent.name if obj.parent else None,
        "children": [c.name for c in obj.children],
        "modifiers": [{{"name": m.name, "type": m.type}} for m in obj.modifiers],
        "constraints": [{{"name": c.name, "type": c.type}} for c in obj.constraints],
        "materials": [m.name for m in obj.data.materials] if hasattr(obj, "data") and hasattr(obj.data, "materials") else [],
        "animation_action": obj.animation_data.action.name if (obj.animation_data and obj.animation_data.action) else None,
    }}
    {detailed_code}
    print(f"INVESTIGATOR_RESULT:{{json.dumps(obj_info, default=str)}}")
"""
    return _run_investigator_script(code)


async def investigate_render(scene_name: str = "Scene") -> dict:
    """Check current render settings and output configuration."""
    code = f"""\
import bpy, json
scene = bpy.data.scenes.get('{scene_name}') or bpy.context.scene

info = {{
    "render_engine": scene.render.engine,
    "resolution_x": scene.render.resolution_x,
    "resolution_y": scene.render.resolution_y,
    "resolution_percentage": scene.render.resolution_percentage,
    "fps": scene.render.fps,
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
    "frame_step": scene.frame_step,
    "filepath": scene.render.filepath,
    "file_format": scene.render.image_settings.file_format,
    "color_mode": scene.render.image_settings.color_mode,

    "samples": getattr(scene.render, 'samples', None),
    "use_motion_blur": getattr(scene.render, 'use_motion_blur', False),
    "use_compositing": getattr(scene.render, 'use_compositing', True),
    "use_sequencer": getattr(scene.render, 'use_sequencer', True),
}}

# Cycles-specific
if scene.render.engine == 'CYCLES':
    info["cycles_samples"] = scene.cycles.samples if hasattr(scene, 'cycles') else None
    info["cycles_device"] = scene.cycles.device if hasattr(scene, 'cycles') else None

# Eevee-specific
if scene.render.engine == 'BLENDER_EEVEE':
    info["eevee_samples"] = scene.eevee.samples if hasattr(scene, 'eevee') else None
    info["eevee_taa_render_samples"] = scene.eevee.taa_render_samples if hasattr(scene, 'eevee') else None

print(f"INVESTIGATOR_RESULT:{{json.dumps(info, default=str)}}")
"""
    return _run_investigator_script(code)
