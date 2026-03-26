"""
data_viz.py — Animated 3D bar chart data visualisation (1920×1080 MP4).

Args JSON:
    data_json:  str   — JSON array e.g. '[{"label":"A","value":42},...]'
    chart_type: str   — "bar" (others reserved for future phases)
    title:      str   — chart title text (optional)
    duration:   float — total clip length in seconds
    output_path: str

Prints: RESULT:{"output_path": "...", "duration": ..., "frames": ..., "chart_type": ...}
"""

import json
import math
import sys

args = {}
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args = json.loads(sys.argv[idx + 1])

data_json: str = args.get("data_json", '[{"label":"A","value":10},{"label":"B","value":20},{"label":"C","value":15}]')
chart_type: str = args.get("chart_type", "bar")
title: str = args.get("title", "")
duration: float = float(args.get("duration", 10.0))
output_path: str = args.get("output_path", "/tmp/data_viz.mp4")

data = json.loads(data_json)
labels = [str(d.get("label", "")) for d in data]
values = [float(d.get("value", 0)) for d in data]
max_val = max(values) if values else 1.0

import bpy  # type: ignore

scene = bpy.context.scene
fps = 24
total_frames = int(duration * fps)
scene.frame_start = 1
scene.frame_end = total_frames
scene.render.fps = fps
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"
scene.render.filepath = output_path

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Dark studio background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.05, 1.0)
    bg.inputs["Strength"].default_value = 0.4

# Colour palette (cycles)
COLORS = [
    (0.1, 0.5, 1.0, 1.0),
    (0.1, 0.9, 0.5, 1.0),
    (1.0, 0.4, 0.1, 1.0),
    (0.9, 0.1, 0.5, 1.0),
    (0.8, 0.8, 0.1, 1.0),
    (0.5, 0.1, 1.0, 1.0),
]

n = len(values)
spacing = 2.2
x_start = -(n - 1) * spacing / 2

# Bars grow from frame 1 → grow_end
grow_end = min(48, total_frames // 2)

for i, (label, value) in enumerate(zip(labels, values)):
    norm_h = (value / max_val) * 4.0  # max height 4 Blender units
    x = x_start + i * spacing
    color = COLORS[i % len(COLORS)]

    # Bar: starts flat, grows to full height
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, 0.001))
    bar = bpy.context.active_object
    bar.scale = (0.85, 0.55, 0.001)
    bar.keyframe_insert(data_path="scale", frame=1)
    bar.location = (x, 0, 0.001)
    bar.keyframe_insert(data_path="location", frame=1)

    bar.scale = (0.85, 0.55, norm_h / 2)
    bar.keyframe_insert(data_path="scale", frame=grow_end)
    bar.location = (x, 0, norm_h / 2)
    bar.keyframe_insert(data_path="location", frame=grow_end)

    mat = bpy.data.materials.new(f"Bar{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.3
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Emission"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = 0.6
    bar.data.materials.append(mat)

    # Label below bar
    bpy.ops.object.text_add(location=(x - 0.35, -0.8, -0.5))
    lbl = bpy.context.active_object
    lbl.data.body = label
    lbl.data.size = 0.32
    lbl.data.extrude = 0.015
    lbl_mat = bpy.data.materials.new(f"Lbl{i}")
    lbl_mat.use_nodes = True
    l_bsdf = lbl_mat.node_tree.nodes["Principled BSDF"]
    l_bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.85, 1.0)
    l_bsdf.inputs["Emission"].default_value = (0.85, 0.85, 0.85, 1.0)
    l_bsdf.inputs["Emission Strength"].default_value = 0.5
    lbl.data.materials.append(lbl_mat)

    # Value label that appears above bar when fully grown
    val_str = f"{int(value)}" if value == int(value) else f"{value:.1f}"
    bpy.ops.object.text_add(location=(x - 0.2, -0.8, norm_h + 0.1))
    val_lbl = bpy.context.active_object
    val_lbl.data.body = val_str
    val_lbl.data.size = 0.28
    val_lbl.data.extrude = 0.01
    val_mat = bpy.data.materials.new(f"Val{i}")
    val_mat.use_nodes = True
    v_bsdf = val_mat.node_tree.nodes["Principled BSDF"]
    v_bsdf.inputs["Base Color"].default_value = color
    v_bsdf.inputs["Emission"].default_value = color
    v_bsdf.inputs["Emission Strength"].default_value = 1.2
    val_lbl.data.materials.append(val_mat)

# Floor grid
bpy.ops.mesh.primitive_plane_add(size=24, location=(0, 0, -0.51))
floor = bpy.context.active_object
fl_mat = bpy.data.materials.new("Floor")
fl_mat.use_nodes = True
fl_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.03, 0.07, 1.0)
fl_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
floor.data.materials.append(fl_mat)

# Chart title
if title:
    bpy.ops.object.text_add(location=(-(len(title) * 0.25), -0.8, 5.2))
    t_obj = bpy.context.active_object
    t_obj.data.body = title
    t_obj.data.size = 0.65
    t_obj.data.extrude = 0.03
    t_mat = bpy.data.materials.new("ChartTitle")
    t_mat.use_nodes = True
    t_bsdf = t_mat.node_tree.nodes["Principled BSDF"]
    t_bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    t_bsdf.inputs["Emission"].default_value = (1.0, 1.0, 1.0, 1.0)
    t_bsdf.inputs["Emission Strength"].default_value = 1.0
    t_obj.data.materials.append(t_mat)

# Lighting
bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(45), 0, math.radians(45))

bpy.ops.object.light_add(type="AREA", location=(-6, -5, 7))
fill = bpy.context.active_object
fill.data.energy = 250
fill.data.color = (0.5, 0.5, 1.0)
fill.data.size = 6.0

# Camera at a slight angle for 3D depth
cam_x = n * 1.2
bpy.ops.object.camera_add(location=(cam_x, -11, 7))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(58), 0, math.radians(20))
cam.data.lens = 35
scene.camera = cam

bpy.ops.render.render(animation=True)

print(f'RESULT:{json.dumps({"output_path": output_path, "duration": duration, "frames": total_frames, "chart_type": chart_type})}')
