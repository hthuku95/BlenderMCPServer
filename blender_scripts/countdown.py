"""
countdown.py — 3D animated countdown timer.

Args (JSON after '--'):
    start_number:  int   — count from (e.g. 10)
    end_number:    int   — count to (e.g. 0 or 1)
    style:         "bold" | "neon" | "minimal" | "cinematic"
    color:         [R, G, B, A]
    bg_color:      [R, G, B, A]
    show_ring:     bool  — show animated ring/circle around number
    duration:      float — total clip duration (each count gets duration/n seconds)
    output_path:   str
"""
import sys
import json
import math
import bpy  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args         = json.loads(args_json)
start_num    = int(args.get("start_number", 5))
end_num      = int(args.get("end_number", 1))
style        = args.get("style", "bold")
color        = args.get("color", [0.1, 0.6, 1.0, 1.0])
bg_color     = args.get("bg_color", [0.02, 0.02, 0.05, 1.0])
show_ring    = bool(args.get("show_ring", True))
duration     = float(args.get("duration", float(abs(start_num - end_num) + 1)))
output_path  = args.get("output_path", "/tmp/countdown.mp4")

count_dir    = 1 if end_num > start_num else -1
numbers      = list(range(start_num, end_num + count_dir, count_dir))
n_counts     = len(numbers)

scene        = bpy.context.scene
fps          = 30
total_frames = int(duration * fps)
frames_per_n = max(1, total_frames // max(n_counts, 1))

scene.frame_start = 1
scene.frame_end   = total_frames
scene.render.fps  = fps
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type   = 'MATERIAL'
scene.display.shading.light         = 'STUDIO'
scene.display.shading.show_shadows  = False
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec  = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = output_path

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = tuple(bg_color)
    bg_node.inputs["Strength"].default_value = 1.0

def make_mat(name, col, metallic=0.5, roughness=0.15, emit=2.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = tuple(col)
    bsdf.inputs["Metallic"].default_value   = metallic
    bsdf.inputs["Roughness"].default_value  = roughness
    bsdf.inputs["Emission"].default_value   = tuple(col[:3]) + (1.0,)
    bsdf.inputs["Emission Strength"].default_value = emit
    mat.diffuse_color = tuple(col)
    return mat

num_mat  = make_mat("Num",  color)
ring_mat = make_mat("Ring", [c * 0.6 for c in color[:3]] + [1.0], metallic=0.3, roughness=0.4, emit=1.0)

# For each number, create a text object and animate scale/visibility
text_objects = []
for idx, num in enumerate(numbers):
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = f"Count_{num}"
    obj.data.body = str(num)
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.extrude = 0.2
    obj.data.bevel_depth = 0.04
    obj.data.materials.append(num_mat)

    start_f = idx * frames_per_n + 1
    end_f   = min((idx + 1) * frames_per_n, total_frames)

    # Hidden outside its window
    obj.scale  = (0.001, 0.001, 0.001)
    obj.keyframe_insert(data_path="scale", frame=1)
    obj.keyframe_insert(data_path="scale", frame=max(1, start_f - 1))

    # Pop in
    obj.scale = (1.5, 1.5, 1.5)
    obj.keyframe_insert(data_path="scale", frame=start_f)

    # Settle
    obj.scale = (1.0, 1.0, 1.0)
    obj.keyframe_insert(data_path="scale", frame=start_f + max(1, frames_per_n // 4))

    # Shrink out just before next number
    obj.scale = (0.001, 0.001, 0.001)
    obj.keyframe_insert(data_path="scale", frame=end_f)
    obj.keyframe_insert(data_path="scale", frame=total_frames)

    # Smooth interpolation
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'

    text_objects.append(obj)

# Optional ring
if show_ring:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=2.2, minor_radius=0.08,
        location=(0, 0, -0.05)
    )
    ring = bpy.context.active_object
    ring.data.materials.append(ring_mat)

    # Rotate ring continuously
    for f in range(1, total_frames + 1):
        t = (f - 1) / fps
        ring.rotation_euler = (math.radians(90), 0, t * 1.2)
        ring.keyframe_insert(data_path="rotation_euler", frame=f)

# Camera
bpy.ops.object.camera_add(location=(0, -8, 0.5))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(87), 0, 0)
scene.camera = cam

# Key light
bpy.ops.object.light_add(type="AREA", location=(3, -5, 4))
key_l = bpy.context.active_object
key_l.data.energy = 600
key_l.data.color  = (1.0, 1.0, 1.0)
key_l.data.size   = 4.0

bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "numbers_shown": numbers,
}
print(f"RESULT:{json.dumps(result)}")
