"""
logo_reveal.py — 3D extruded text / logo reveal animation.

Args (JSON after '--'):
    text:         str  — main text or brand name
    tagline:      str  — optional second line
    style:        "extrude_reveal" | "typewriter" | "split" | "zoom_in"
    color:        [R, G, B, A] float list for main text material
    bg_color:     [R, G, B, A] float list for background
    duration:     float
    output_path:  str
    font:         str (optional, Blender font name)
"""
import sys
import json
import math
import bpy  # type: ignore

# Parse args
args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args       = json.loads(args_json)
text_str   = args.get("text", "BRAND")
tagline    = args.get("tagline", "")
style      = args.get("style", "extrude_reveal")
color      = args.get("color", [0.1, 0.5, 1.0, 1.0])
bg_color   = args.get("bg_color", [0.02, 0.02, 0.05, 1.0])
duration   = float(args.get("duration", 6.0))
output_path = args.get("output_path", "/tmp/logo_reveal.mp4")

# Scene setup
scene = bpy.context.scene
fps   = 30
total_frames = int(duration * fps)
scene.frame_start = 1
scene.frame_end   = total_frames
scene.render.fps  = fps
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type  = 'MATERIAL'
scene.display.shading.light        = 'STUDIO'
scene.display.shading.show_shadows = True
scene.display.shading.shadow_intensity = 0.4
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec  = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = output_path

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# World background
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = tuple(bg_color)
    bg_node.inputs["Strength"].default_value = 1.0

# Main text material
main_mat = bpy.data.materials.new("MainText")
main_mat.use_nodes = True
bsdf = main_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = tuple(color)
bsdf.inputs["Metallic"].default_value   = 0.8
bsdf.inputs["Roughness"].default_value  = 0.15
main_mat.diffuse_color = tuple(color)

# Emission/glow material for tagline
tag_mat = bpy.data.materials.new("TagText")
tag_mat.use_nodes = True
t_bsdf = tag_mat.node_tree.nodes["Principled BSDF"]
t_bsdf.inputs["Base Color"].default_value    = (1.0, 1.0, 1.0, 1.0)
t_bsdf.inputs["Emission"].default_value      = (1.0, 1.0, 1.0, 1.0)
t_bsdf.inputs["Emission Strength"].default_value = 1.5
tag_mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)

def add_text_object(text, name, extrude=0.15, bevel=0.02, scale=1.0, location=(0,0,0), mat=None):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.extrude = extrude
    obj.data.bevel_depth = bevel
    obj.scale = (scale, scale, scale)
    if mat:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return obj

# Create main text
main_obj = add_text_object(
    text_str, "MainLogo",
    extrude=0.18 if style == "extrude_reveal" else 0.05,
    bevel=0.03,
    scale=1.0,
    location=(0, 0, 0),
    mat=main_mat,
)

# Create tagline if provided
tag_obj = None
if tagline:
    tag_obj = add_text_object(
        tagline, "Tagline",
        extrude=0.03,
        bevel=0.01,
        scale=0.4,
        location=(0, -0.9, 0),
        mat=tag_mat,
    )

# Camera
bpy.ops.object.camera_add(location=(0, -8, 0))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(90), 0, 0)
scene.camera = cam

# Lighting
bpy.ops.object.light_add(type="AREA", location=(3, -4, 4))
key = bpy.context.active_object
key.data.energy = 500
key.data.color  = (1.0, 0.95, 0.9)
key.data.size   = 3.0

bpy.ops.object.light_add(type="AREA", location=(-3, -4, 2))
fill = bpy.context.active_object
fill.data.energy = 200
fill.data.color  = tuple(color[:3])
fill.data.size   = 2.0

bpy.ops.object.light_add(type="AREA", location=(0, -4, -2))
rim = bpy.context.active_object
rim.data.energy = 150
rim.data.color  = (0.5, 0.8, 1.0)

# Animate by style
if style == "extrude_reveal":
    # Scale Z from 0 to 1 (extrusion grows in)
    main_obj.scale = (1, 1, 0)
    main_obj.keyframe_insert(data_path="scale", frame=1)
    main_obj.scale = (1, 1, 1)
    main_obj.keyframe_insert(data_path="scale", frame=int(fps * 1.5))
    # Gentle Z rotation after reveal
    for f in range(int(fps * 1.5), total_frames + 1):
        t = (f - fps * 1.5) / fps
        main_obj.rotation_euler = (0, 0, t * 0.05)
        main_obj.keyframe_insert(data_path="rotation_euler", frame=f)
    if tag_obj:
        tag_obj.scale = (0.4, 0.4, 0)
        tag_obj.keyframe_insert(data_path="scale", frame=int(fps * 1.2))
        tag_obj.scale = (0.4, 0.4, 0.4)
        tag_obj.keyframe_insert(data_path="scale", frame=int(fps * 2.2))

elif style == "zoom_in":
    main_obj.scale = (4, 4, 4)
    main_obj.keyframe_insert(data_path="scale", frame=1)
    main_obj.scale = (1, 1, 1)
    main_obj.keyframe_insert(data_path="scale", frame=int(fps * 1.8))
    if tag_obj:
        tag_obj.scale = (0, 0, 0)
        tag_obj.keyframe_insert(data_path="scale", frame=int(fps * 1.5))
        tag_obj.scale = (0.4, 0.4, 0.4)
        tag_obj.keyframe_insert(data_path="scale", frame=int(fps * 2.5))

elif style == "split":
    # Left and right halves come together — approximate with X scale from left
    main_obj.location.x = -6
    main_obj.keyframe_insert(data_path="location", frame=1)
    main_obj.location.x = 0
    main_obj.keyframe_insert(data_path="location", frame=int(fps * 1.5))
    if tag_obj:
        tag_obj.location.x = 6
        tag_obj.keyframe_insert(data_path="location", frame=1)
        tag_obj.location.x = 0
        tag_obj.keyframe_insert(data_path="location", frame=int(fps * 1.8))

elif style == "typewriter":
    # Reveal characters one at a time using scale trick
    n_chars = len(text_str)
    for f in range(1, total_frames + 1):
        progress = min(1.0, (f - 1) / max(fps * 1.5, 1))
        n_visible = max(1, int(progress * n_chars))
        # Manim-style: show partial body (Blender doesn't support char reveal natively)
        # Use scale as proxy animation
        main_obj.scale = (progress, 1, 1)
        main_obj.keyframe_insert(data_path="scale", frame=f)

# Set all fcurves to smooth interpolation
for obj in [main_obj, tag_obj]:
    if obj is None:
        continue
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'BEZIER'

bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
}
print(f"RESULT:{json.dumps(result)}")
