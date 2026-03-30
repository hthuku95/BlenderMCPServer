"""
particle_confetti.py — Keyframe-animated confetti / snow / stars burst.

Replaces particle-system approach (unreliable in headless Blender 3.4) with
individually keyframed mesh objects — guaranteed to work with BLENDER_WORKBENCH.

Args (JSON after '--'):
    style:           "confetti" | "snow" | "stars" | "rain" | "bubbles"
    count:           int   — number of pieces (capped at 100 for performance)
    duration:        float — clip length in seconds (default: 6)
    primary_color:   [R,G,B,A]
    secondary_color: [R,G,B,A]
    bg_color:        [R,G,B,A]
    output_path:     str
"""
import sys, json, math, random
import bpy  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args = json.loads(args_json)

style         = args.get("style", "confetti")
count         = min(int(args.get("count", 80)), 100)   # cap for performance
duration      = float(args.get("duration", 6.0))
primary_color = args.get("primary_color",   [1.0, 0.3, 0.1, 1.0])
second_color  = args.get("secondary_color", [0.1, 0.5, 1.0, 1.0])
bg_color      = args.get("bg_color",        [0.02, 0.02, 0.06, 1.0])
output_path   = args.get("output_path", "/tmp/particle_confetti.mp4")

fps          = 30
total_frames = int(duration * fps)
random.seed(42)

# ── Scene ─────────────────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end   = total_frames
scene.render.fps  = fps
scene.render.resolution_x = 854
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light      = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec  = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = output_path

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# World background
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value    = tuple(bg_color[:3]) + (1.0,)
    bg_node.inputs["Strength"].default_value = 1.0

# ── Colour palette ────────────────────────────────────────────────────────────
palette = [
    [1.0, 0.15, 0.15, 1.0],
    [0.15, 0.8,  0.15, 1.0],
    [0.15, 0.15, 1.0,  1.0],
    [1.0, 0.85, 0.0,  1.0],
    [1.0, 0.5,  0.0,  1.0],
    [0.7, 0.0,  0.9,  1.0],
    primary_color,
    second_color,
]

def make_mat(name, col):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(col[:3]) + (1.0,)
    return mat

# ── Create confetti pieces ─────────────────────────────────────────────────────
pieces = []
for i in range(count):
    col = palette[i % len(palette)]

    if style == "snow":
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.07, subdivisions=1,
                                               location=(0, 0, 0))
    elif style == "stars":
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.08, subdivisions=0,
                                               location=(0, 0, 0))
    elif style == "rain":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.25,
                                             location=(0, 0, 0))
    elif style == "bubbles":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09, segments=6, rings=4,
                                              location=(0, 0, 0))
    else:  # confetti — flat rectangle
        bpy.ops.mesh.primitive_plane_add(size=0.12, location=(0, 0, 0))

    obj = bpy.context.active_object
    obj.name = f"Piece_{i}"
    mat = make_mat(f"M_{i}", col)
    obj.data.materials.append(mat)
    pieces.append(obj)

# ── Animate each piece with keyframes ─────────────────────────────────────────
# Pieces start staggered at the top and fall down (or rise for bubbles)
# Keyframe every 4 frames for performance
KF_STEP = 4

for i, obj in enumerate(pieces):
    x0 = random.uniform(-5.0, 5.0)
    y0 = random.uniform(-1.0, 1.0)
    delay = random.randint(0, max(1, total_frames // 3))  # stagger start

    spin_x = random.uniform(-3.0, 3.0)  # rotation speed rad/frame
    spin_z = random.uniform(-3.0, 3.0)

    if style == "bubbles":
        z_start = random.uniform(-4.0, -2.0)
        z_end   = random.uniform(3.0, 5.0)
    elif style == "rain":
        z_start = random.uniform(4.0, 6.0)
        z_end   = random.uniform(-5.0, -3.0)
    elif style == "snow":
        z_start = random.uniform(3.0, 5.5)
        z_end   = random.uniform(-3.0, -1.0)
        x0 += random.uniform(-0.5, 0.5)  # slight drift
    else:
        z_start = random.uniform(3.5, 6.0)
        z_end   = random.uniform(-4.0, -1.5)

    # Hide before its delay frame
    obj.hide_render   = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render",   frame=1)
    obj.keyframe_insert(data_path="hide_viewport", frame=1)

    start_f = delay + 1
    if start_f > 1:
        obj.hide_render   = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render",   frame=start_f - 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=start_f - 1)

    obj.hide_render   = False
    obj.hide_viewport = False
    obj.keyframe_insert(data_path="hide_render",   frame=start_f)
    obj.keyframe_insert(data_path="hide_viewport", frame=start_f)

    for f in range(start_f, total_frames + 1, KF_STEP):
        t = (f - start_f) / max(1, total_frames - start_f)
        z = z_start + (z_end - z_start) * t

        # Snow/confetti gentle sway
        sway = math.sin(t * math.pi * 4 + i) * 0.3 if style in ("snow", "confetti") else 0.0

        obj.location = (x0 + sway, y0, z)
        obj.rotation_euler = (
            spin_x * (f - start_f) * 0.05,
            0,
            spin_z * (f - start_f) * 0.05,
        )
        obj.keyframe_insert(data_path="location",       frame=f)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)

# ── Camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -12, 1.5))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(82), 0, 0)
scene.camera = cam

# ── Lighting ──────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(4, -6, 8))
sun = bpy.context.active_object.data
sun.energy = 4.0

bpy.ops.object.light_add(type="AREA", location=(-4, -4, 5))
fill = bpy.context.active_object.data
fill.energy = 300; fill.size = 4.0

# ── Render ────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "854x480",
    "frames": total_frames,
    "style": style,
    "count": count,
}
print(f"RESULT:{json.dumps(result)}")
