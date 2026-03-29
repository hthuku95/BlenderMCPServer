"""
thumbnail.py — 3D rendered YouTube thumbnail (1280×720 PNG).

Args JSON (passed after '--'):
    prompt:      str  — scene description
    title_text:  str  — optional text to embed in the scene
    style:       str  — "youtube" | "cinematic" | "minimal"
    output_path: str  — destination PNG path

Prints: RESULT:{"output_path": "...", "width": 1280, "height": 720}
"""

import json
import math
import sys

args = {}
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args = json.loads(sys.argv[idx + 1])

prompt: str = args.get("prompt", "tech YouTube thumbnail")
title_text: str = args.get("title_text", "")
style: str = args.get("style", "youtube")
output_path: str = args.get("output_path", "/tmp/thumbnail.png")

import bpy  # type: ignore

scene = bpy.context.scene
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
# Thumbnail is a single still frame — use Cycles CPU for photorealistic quality.
# 128 samples + OIDN denoising ≈ 45s on 1 vCPU, which is acceptable.
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
try:
    # OPENIMAGEDENOISE requires SSE4.1 and must be compiled into the Blender build.
    # On some headless servers (Blender 3.4 apt package) the enum may be empty —
    # wrap in try/except so we get denoising when available, skip it when not.
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
except TypeError:
    try:
        scene.cycles.denoiser = "NLM"
    except TypeError:
        scene.cycles.use_denoising = False
try:
    _cprefs = bpy.context.preferences.addons["cycles"].preferences
    _cprefs.get_devices()
    _cprefs.compute_device_type = "NONE"
except Exception:
    pass
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = output_path
scene.frame_set(1)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

STYLES = {
    "youtube":   {"bg": (0.02, 0.02, 0.02, 1.0), "accent": (1.0, 0.2, 0.1, 1.0)},
    "cinematic": {"bg": (0.01, 0.02, 0.08, 1.0), "accent": (0.1, 0.5, 1.0, 1.0)},
    "minimal":   {"bg": (0.92, 0.92, 0.92, 1.0), "accent": (0.1, 0.1, 0.7, 1.0)},
}
p = STYLES.get(style, STYLES["youtube"])

# World background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = p["bg"]
    bg_node.inputs["Strength"].default_value = 1.0

# Three angled decorative panels (depth/layering feel)
for i in range(3):
    t = i / 2.0
    color = tuple(p["bg"][c] * (1 - t) + p["accent"][c] * t for c in range(3)) + (1.0,)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2 + i * 2.5, 1.5, -0.3 + i * 0.6))
    slab = bpy.context.active_object
    slab.scale = (1.8, 0.08, 2.5 + i * 0.5)
    slab.rotation_euler = (0, 0, math.radians(12))
    mat = bpy.data.materials.new(f"Slab{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.2
    bsdf.inputs["Emission"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = 0.4
    mat.diffuse_color = color
    slab.data.materials.append(mat)

# Hero sphere (right side)
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.3, location=(3.0, 0, 0.8))
sphere = bpy.context.active_object
sph_mat = bpy.data.materials.new("Hero")
sph_mat.use_nodes = True
s_bsdf = sph_mat.node_tree.nodes["Principled BSDF"]
s_bsdf.inputs["Base Color"].default_value = p["accent"]
s_bsdf.inputs["Metallic"].default_value = 0.8
s_bsdf.inputs["Roughness"].default_value = 0.1
s_bsdf.inputs["Emission"].default_value = p["accent"]
s_bsdf.inputs["Emission Strength"].default_value = 0.5
sph_mat.diffuse_color = p["accent"]
sphere.data.materials.append(sph_mat)

# Floor plane
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -1.5))
floor = bpy.context.active_object
fl_mat = bpy.data.materials.new("Floor")
fl_mat.use_nodes = True
fl_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.03, 0.06, 1.0)
fl_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
floor.data.materials.append(fl_mat)

# Title text overlay
if title_text:
    bpy.ops.object.text_add(location=(-5.5, 0, 0.3))
    txt = bpy.context.active_object
    txt.data.body = title_text
    txt.data.size = 0.75
    txt.data.extrude = 0.06
    txt.data.align_x = "LEFT"
    txt_mat = bpy.data.materials.new("TitleText")
    txt_mat.use_nodes = True
    t_bsdf = txt_mat.node_tree.nodes["Principled BSDF"]
    t_bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    t_bsdf.inputs["Emission"].default_value = (1.0, 1.0, 1.0, 1.0)
    t_bsdf.inputs["Emission Strength"].default_value = 1.5
    txt.data.materials.append(txt_mat)

# Key light
bpy.ops.object.light_add(type="AREA", location=(0, -6, 5))
key = bpy.context.active_object
key.data.energy = 600
key.data.size = 4.0
key.data.color = (1.0, 0.95, 0.9)

# Accent rim light (coloured)
bpy.ops.object.light_add(type="AREA", location=(5, 3, 3))
rim = bpy.context.active_object
rim.data.energy = 400
rim.data.color = p["accent"][:3]
rim.data.size = 2.0

# Camera
bpy.ops.object.camera_add(location=(0, -9, 1.5))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(80), 0, 0)
cam.data.lens = 35
scene.camera = cam

bpy.ops.render.render(write_still=True)

print(f'RESULT:{json.dumps({"output_path": output_path, "width": 1280, "height": 720})}')
