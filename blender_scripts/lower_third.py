"""
lower_third.py — Animated lower-third name plate (1920×1080 MP4, green-screen background).

The green-screen (#00B200) background can be chroma-keyed out in any NLE.
The panel + text slides up from below in the first ~0.4s, holds, then holds to end.

Args JSON:
    name_text:     str   — primary text (person name / topic)
    subtitle_text: str   — secondary text (job title / context), optional
    style:         str   — "modern" | "minimal" | "bold"
    duration:      float — total clip length in seconds
    output_path:   str

Prints: RESULT:{"output_path": "...", "duration": ..., "frames": ..., "keying": "green_screen"}
"""

import json
import math
import sys

args = {}
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args = json.loads(sys.argv[idx + 1])

name_text: str = args.get("name_text", "Name Here")
subtitle_text: str = args.get("subtitle_text", "")
style: str = args.get("style", "modern")
duration: float = float(args.get("duration", 5.0))
output_path: str = args.get("output_path", "/tmp/lower_third.mp4")

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

STYLES = {
    "modern":  {"panel": (0.04, 0.04, 0.12, 1.0), "accent": (0.0, 0.55, 1.0, 1.0), "text": (1.0, 1.0, 1.0, 1.0)},
    "minimal": {"panel": (1.0, 1.0, 1.0, 1.0),    "accent": (0.1, 0.1, 0.8, 1.0), "text": (0.05, 0.05, 0.05, 1.0)},
    "bold":    {"panel": (0.7, 0.05, 0.05, 1.0),  "accent": (1.0, 1.0, 1.0, 1.0), "text": (1.0, 1.0, 1.0, 1.0)},
}
p = STYLES.get(style, STYLES["modern"])

# Green screen background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.0, 0.698, 0.0, 1.0)  # #00B200
    bg.inputs["Strength"].default_value = 1.0

# Orthographic camera — no perspective distortion for 2D overlay
bpy.ops.object.camera_add(location=(0, -12, 0))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(90), 0, 0)
cam.data.type = "ORTHO"
cam.data.ortho_scale = 12.0
scene.camera = cam

# Scene coords: camera sees roughly x ±6, z ±3.375 (16:9 at ortho_scale=12)
# Lower third target: panel sits at z = -2.6 (bottom 20% of frame)
PANEL_Z = -2.6
OFFSCREEN_Z = -4.5
SLIDE_IN_END = 10  # frame at which slide finishes

# Main background panel
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.02, OFFSCREEN_Z))
panel = bpy.context.active_object
panel.scale = (10.5, 0.01, 0.55)
panel.keyframe_insert(data_path="location", frame=1)
panel.location = (0, 0.02, PANEL_Z)
panel.keyframe_insert(data_path="location", frame=SLIDE_IN_END)

p_mat = bpy.data.materials.new("Panel")
p_mat.use_nodes = True
p_bsdf = p_mat.node_tree.nodes["Principled BSDF"]
p_bsdf.inputs["Base Color"].default_value = p["panel"]
p_bsdf.inputs["Emission Color"].default_value = p["panel"]
p_bsdf.inputs["Emission Strength"].default_value = 2.5
panel.data.materials.append(p_mat)

# Left accent stripe
bpy.ops.mesh.primitive_cube_add(size=1, location=(-9.6, 0.04, OFFSCREEN_Z))
stripe = bpy.context.active_object
stripe.scale = (0.1, 0.01, 0.55)
stripe.keyframe_insert(data_path="location", frame=1)
stripe.location = (-9.6, 0.04, PANEL_Z)
stripe.keyframe_insert(data_path="location", frame=SLIDE_IN_END)

s_mat = bpy.data.materials.new("Stripe")
s_mat.use_nodes = True
s_bsdf = s_mat.node_tree.nodes["Principled BSDF"]
s_bsdf.inputs["Base Color"].default_value = p["accent"]
s_bsdf.inputs["Emission Color"].default_value = p["accent"]
s_bsdf.inputs["Emission Strength"].default_value = 4.0
stripe.data.materials.append(s_mat)

# Name text
bpy.ops.object.text_add(location=(-9.0, 0, OFFSCREEN_Z + 0.15))
name_obj = bpy.context.active_object
name_obj.data.body = name_text
name_obj.data.size = 0.38
name_obj.data.extrude = 0.01
name_obj.data.align_x = "LEFT"
name_obj.keyframe_insert(data_path="location", frame=1)
name_obj.location = (-9.0, 0, PANEL_Z + 0.15)
name_obj.keyframe_insert(data_path="location", frame=SLIDE_IN_END)

n_mat = bpy.data.materials.new("NameMat")
n_mat.use_nodes = True
n_bsdf = n_mat.node_tree.nodes["Principled BSDF"]
n_bsdf.inputs["Base Color"].default_value = p["text"]
n_bsdf.inputs["Emission Color"].default_value = p["text"]
n_bsdf.inputs["Emission Strength"].default_value = 3.0
name_obj.data.materials.append(n_mat)

# Subtitle text
if subtitle_text:
    bpy.ops.object.text_add(location=(-9.0, 0, OFFSCREEN_Z - 0.17))
    sub_obj = bpy.context.active_object
    sub_obj.data.body = subtitle_text
    sub_obj.data.size = 0.24
    sub_obj.data.extrude = 0.005
    sub_obj.data.align_x = "LEFT"
    sub_obj.keyframe_insert(data_path="location", frame=1)
    sub_obj.location = (-9.0, 0, PANEL_Z - 0.17)
    sub_obj.keyframe_insert(data_path="location", frame=SLIDE_IN_END)

    sub_mat = bpy.data.materials.new("SubMat")
    sub_mat.use_nodes = True
    sb_bsdf = sub_mat.node_tree.nodes["Principled BSDF"]
    sb_bsdf.inputs["Base Color"].default_value = p["accent"]
    sb_bsdf.inputs["Emission Color"].default_value = p["accent"]
    sb_bsdf.inputs["Emission Strength"].default_value = 2.0
    sub_obj.data.materials.append(sub_mat)

# Flat area light (bright, directionless — avoids shadows on green bg)
bpy.ops.object.light_add(type="AREA", location=(0, -8, 0))
light = bpy.context.active_object
light.data.energy = 800
light.data.size = 15.0
light.rotation_euler = (math.radians(90), 0, 0)

bpy.ops.render.render(animation=True)

print(f'RESULT:{json.dumps({"output_path": output_path, "duration": duration, "frames": total_frames, "keying": "green_screen"})}')
