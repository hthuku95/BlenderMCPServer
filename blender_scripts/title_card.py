"""
title_card.py — Animated 3D title card MP4 (1920×1080).

Args JSON:
    title:       str   — main title text
    subtitle:    str   — optional secondary text
    duration:    float — clip length in seconds (3–8 recommended)
    style:       str   — "cinematic" | "minimal" | "bold"
    output_path: str

Prints: RESULT:{"output_path": "...", "duration": ..., "frames": ...}
"""

import json
import math
import sys

args = {}
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args = json.loads(sys.argv[idx + 1])

title: str = args.get("title", "Your Title Here")
subtitle: str = args.get("subtitle", "")
duration: float = float(args.get("duration", 5.0))
style: str = args.get("style", "cinematic")
output_path: str = args.get("output_path", "/tmp/title_card.mp4")

import bpy  # type: ignore

scene = bpy.context.scene
fps = 24
total_frames = int(duration * fps)
scene.frame_start = 1
scene.frame_end = total_frames
scene.render.fps = fps
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.engine = "BLENDER_WORKBENCH"  # Fast CPU-only
scene.display.shading.color_type = 'MATERIAL'  # CRITICAL: read mat.diffuse_color
scene.display.shading.light = 'STUDIO'
scene.display.shading.show_shadows = True
scene.display.shading.shadow_intensity = 0.3
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = output_path

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

STYLES = {
    "cinematic": {"bg": (0.01, 0.01, 0.05, 1.0), "text": (1.0, 1.0, 1.0, 1.0), "accent": (0.1, 0.5, 1.0, 1.0)},
    "minimal":   {"bg": (0.95, 0.95, 0.95, 1.0), "text": (0.05, 0.05, 0.05, 1.0), "accent": (0.1, 0.2, 0.9, 1.0)},
    "bold":      {"bg": (0.0,  0.0,  0.0,  1.0), "text": (1.0, 0.9, 0.0, 1.0),   "accent": (1.0, 0.3, 0.0, 1.0)},
}
p = STYLES.get(style, STYLES["cinematic"])

# World background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = p["bg"]
    bg.inputs["Strength"].default_value = 1.0

# Accent horizontal bar — slides in (scale X: 0 → 8) over first 10 frames
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.05, -0.9))
bar = bpy.context.active_object
bar.scale = (0.001, 0.015, 0.035)
bar.keyframe_insert(data_path="scale", frame=1)
bar.scale = (8.0, 0.015, 0.035)
bar.keyframe_insert(data_path="scale", frame=10)

bar_mat = bpy.data.materials.new("BarMat")
bar_mat.use_nodes = True
b_bsdf = bar_mat.node_tree.nodes["Principled BSDF"]
b_bsdf.inputs["Base Color"].default_value = p["accent"]
b_bsdf.inputs["Emission"].default_value = p["accent"]
b_bsdf.inputs["Emission Strength"].default_value = 3.0
bar_mat.diffuse_color = p["accent"]
bar.data.materials.append(bar_mat)

# Main title — slides in from left (frames 6–18)
bpy.ops.object.text_add(location=(-12, 0, 0.2))
title_obj = bpy.context.active_object
title_obj.data.body = title
title_obj.data.size = 0.9
title_obj.data.extrude = 0.04
title_obj.data.align_x = "LEFT"
title_obj.keyframe_insert(data_path="location", frame=6)
title_obj.location = (-5.0, 0, 0.2)
title_obj.keyframe_insert(data_path="location", frame=18)

title_mat = bpy.data.materials.new("TitleMat")
title_mat.use_nodes = True
t_bsdf = title_mat.node_tree.nodes["Principled BSDF"]
t_bsdf.inputs["Base Color"].default_value = p["text"]
t_bsdf.inputs["Emission"].default_value = p["text"]
t_bsdf.inputs["Emission Strength"].default_value = 0.8
title_mat.diffuse_color = p["text"]
title_obj.data.materials.append(title_mat)

# Subtitle — slides in after title (frames 12–24)
if subtitle:
    bpy.ops.object.text_add(location=(-12, 0, -0.65))
    sub_obj = bpy.context.active_object
    sub_obj.data.body = subtitle
    sub_obj.data.size = 0.42
    sub_obj.data.extrude = 0.015
    sub_obj.data.align_x = "LEFT"
    sub_obj.keyframe_insert(data_path="location", frame=12)
    sub_obj.location = (-5.0, 0, -0.65)
    sub_obj.keyframe_insert(data_path="location", frame=24)

    sub_mat = bpy.data.materials.new("SubMat")
    sub_mat.use_nodes = True
    s_bsdf = sub_mat.node_tree.nodes["Principled BSDF"]
    s_bsdf.inputs["Base Color"].default_value = p["accent"]
    s_bsdf.inputs["Emission"].default_value = p["accent"]
    s_bsdf.inputs["Emission Strength"].default_value = 1.2
    sub_mat.diffuse_color = p["accent"]
    sub_obj.data.materials.append(sub_mat)

# Subtle background geometric accent (low-poly diamond)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=3.0, location=(5.5, 1.5, 0))
gem = bpy.context.active_object
gem.rotation_euler = (math.radians(20), math.radians(30), math.radians(15))
gem_mat = bpy.data.materials.new("Gem")
gem_mat.use_nodes = True
g_bsdf = gem_mat.node_tree.nodes["Principled BSDF"]
g_bsdf.inputs["Base Color"].default_value = p["accent"]
g_bsdf.inputs["Metallic"].default_value = 1.0
g_bsdf.inputs["Roughness"].default_value = 0.05
g_bsdf.inputs["Alpha"].default_value = 0.15
gem_mat.blend_method = "BLEND"
gem_mat.diffuse_color = (p["accent"][0], p["accent"][1], p["accent"][2], 0.15)
gem.data.materials.append(gem_mat)

# Lighting
bpy.ops.object.light_add(type="AREA", location=(0, -8, 4))
key = bpy.context.active_object
key.data.energy = 500
key.data.size = 8.0

bpy.ops.object.light_add(type="AREA", location=(6, 4, 3))
fill = bpy.context.active_object
fill.data.energy = 200
fill.data.color = p["accent"][:3]
fill.data.size = 4.0

# Straight-on camera
bpy.ops.object.camera_add(location=(0, -12, 0))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(90), 0, 0)
cam.data.lens = 50
scene.camera = cam

bpy.ops.render.render(animation=True)

print(f'RESULT:{json.dumps({"output_path": output_path, "duration": duration, "frames": total_frames})}')
