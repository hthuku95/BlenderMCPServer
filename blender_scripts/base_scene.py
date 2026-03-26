"""
base_scene.py — Template bpy script for procedural 3D clip generation.

Usage (called by blender_runner.py):
    blender --background --python base_scene.py -- '{"prompt":"ocean waves","duration":10,"style":"cinematic","output_path":"/tmp/out.mp4"}'

Prints: RESULT:{"output_path": "/tmp/out.mp4", "duration": 10, "resolution": "1920x1080", "frames": 240}
"""

import sys
import json
import os
import math
import tempfile

# Parse args passed after '--'
args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args = json.loads(args_json)
prompt: str = args.get("prompt", "cinematic abstract scene")
duration: float = float(args.get("duration", 10.0))
style: str = args.get("style", "cinematic")
output_path: str = args.get("output_path", "/tmp/blender_render.mp4")

# ---- Import bpy (only available inside Blender) ----
import bpy  # type: ignore

scene = bpy.context.scene

# ---- Scene settings ----
fps = 24
total_frames = int(duration * fps)
scene.frame_start = 1
scene.frame_end = total_frames
scene.render.fps = fps
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

# ---- Output settings: MP4 via FFmpeg ----
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"  # good quality
scene.render.filepath = output_path

# ---- Clear default scene ----
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ---- Style presets ----
STYLES = {
    "cinematic": {"sky_color": (0.01, 0.02, 0.08, 1.0), "accent": (0.1, 0.5, 1.0, 1.0)},
    "calm":      {"sky_color": (0.05, 0.12, 0.18, 1.0), "accent": (0.3, 0.7, 0.9, 1.0)},
    "energetic": {"sky_color": (0.08, 0.0, 0.02, 1.0),  "accent": (1.0, 0.3, 0.1, 1.0)},
    "minimal":   {"sky_color": (0.9, 0.9, 0.9, 1.0),    "accent": (0.2, 0.2, 0.2, 1.0)},
}
preset = STYLES.get(style, STYLES["cinematic"])

# ---- World (sky) ----
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = preset["sky_color"]
    bg_node.inputs["Strength"].default_value = 0.5

# ---- Floor plane ----
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.active_object
floor_mat = bpy.data.materials.new("Floor")
floor_mat.use_nodes = True
floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.05, 0.05, 0.08, 1.0
)
floor_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
floor.data.materials.append(floor_mat)

# ---- Accent spheres that orbit ----
accent_mat = bpy.data.materials.new("Accent")
accent_mat.use_nodes = True
bsdf = accent_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = preset["accent"]
bsdf.inputs["Emission"].default_value = preset["accent"]
bsdf.inputs["Emission Strength"].default_value = 2.0

NUM_ORBS = 5
for i in range(NUM_ORBS):
    angle_offset = (2 * math.pi / NUM_ORBS) * i
    radius = 3.5
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(radius, 0, 1.0))
    orb = bpy.context.active_object
    orb.data.materials.append(accent_mat)

    # Animate orbit using keyframes
    for frame in range(1, total_frames + 1):
        t = (frame - 1) / fps  # seconds
        speed = 0.4 if style != "energetic" else 0.9
        angle = angle_offset + t * speed * math.pi
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 1.0 + 0.4 * math.sin(t * 2 + angle_offset)
        orb.location = (x, y, z)
        orb.keyframe_insert(data_path="location", frame=frame)

# ---- Central icosphere ----
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(0, 0, 1.5))
center = bpy.context.active_object
center_mat = bpy.data.materials.new("Center")
center_mat.use_nodes = True
c_bsdf = center_mat.node_tree.nodes["Principled BSDF"]
c_bsdf.inputs["Base Color"].default_value = preset["accent"]
c_bsdf.inputs["Metallic"].default_value = 0.9
c_bsdf.inputs["Roughness"].default_value = 0.1
center.data.materials.append(center_mat)

# Slow rotation on center object
for frame in range(1, total_frames + 1):
    t = (frame - 1) / fps
    center.rotation_euler = (0, 0, t * 0.3)
    center.keyframe_insert(data_path="rotation_euler", frame=frame)

# ---- Key light ----
bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.data.color = (1.0, 0.95, 0.85)

# ---- Fill light ----
bpy.ops.object.light_add(type="AREA", location=(-4, -2, 4))
fill = bpy.context.active_object
fill.data.energy = 200
fill.data.color = preset["accent"][:3]
fill.data.size = 3.0

# ---- Camera ----
bpy.ops.object.camera_add(location=(7, -5, 4))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(55), 0, math.radians(55))
scene.camera = cam

# Gentle camera push-in
for frame in range(1, total_frames + 1):
    t = (frame - 1) / fps
    cam.location = (7 - t * 0.05, -5 + t * 0.02, 4 - t * 0.01)
    cam.keyframe_insert(data_path="location", frame=frame)

# ---- Render ----
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1920x1080",
    "frames": total_frames,
    "fps": fps,
}
print(f"RESULT:{json.dumps(result)}")
