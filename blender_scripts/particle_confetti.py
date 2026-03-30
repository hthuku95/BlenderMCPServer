"""
particle_confetti.py — Particle-system confetti / snow / stars burst animation.

Args (JSON after '--'):
    style:           "confetti" | "snow" | "stars" | "rain" | "bubbles"
    count:           int   — number of particles (default: 400)
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
count         = int(args.get("count", 400))
duration      = float(args.get("duration", 6.0))
primary_color = args.get("primary_color",   [1.0, 0.3, 0.1, 1.0])
second_color  = args.get("secondary_color", [0.1, 0.5, 1.0, 1.0])
bg_color      = args.get("bg_color",        [0.02, 0.02, 0.06, 1.0])
output_path   = args.get("output_path", "/tmp/particle_confetti.mp4")

fps          = 30
total_frames = int(duration * fps)

# ── Scene setup ──────────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end   = total_frames
scene.render.fps  = fps
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light      = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
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

# ── Palette ──────────────────────────────────────────────────────────────────
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

# ── Create particle instance objects ─────────────────────────────────────────
instances = []
for i, col in enumerate(palette[:6]):
    if style == "snow":
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.06, subdivisions=1,
                                              location=(1000, 1000, 1000))
    elif style == "stars":
        bpy.ops.mesh.primitive_circle_add(radius=0.07, vertices=5,
                                          location=(1000, 1000, 1000))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')
    elif style == "rain":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.2,
                                             location=(1000, 1000, 1000))
    elif style == "bubbles":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, segments=8, rings=6,
                                              location=(1000, 1000, 1000))
    else:  # confetti – flat rectangle
        bpy.ops.mesh.primitive_plane_add(size=0.1, location=(1000, 1000, 1000))

    inst = bpy.context.active_object
    inst.name = f"Inst_{i}"
    mat = bpy.data.materials.new(f"PMat_{i}")
    mat.diffuse_color = tuple(col)
    inst.data.materials.append(mat)
    inst.hide_viewport = False
    inst.hide_render   = False
    instances.append(inst)

# ── Emitter plane ────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 4.5))
emitter = bpy.context.active_object
emitter.name = "Emitter"
em_mat = bpy.data.materials.new("EmMat")
em_mat.diffuse_color = tuple(bg_color[:3]) + (1.0,)
emitter.data.materials.append(em_mat)

ps_mod = emitter.modifiers.new("Particles", type='PARTICLE_SYSTEM')
ps     = emitter.particle_systems[0]
s      = ps.settings

s.count       = count
s.frame_start = 1
s.frame_end   = max(1, int(total_frames * 0.6))
s.lifetime    = total_frames
s.lifetime_random = 0.3
s.emit_from   = "FACE"
s.distribution = "RAND"

# Physics
s.physics_type   = "NEWTON"
s.normal_factor  = 0.0
s.factor_random  = 3.0
s.gravity        = 0.5 if style not in ("snow",) else 0.2
s.drag_factor    = 0.08 if style != "rain" else 0.02

# Rotation (confetti tumbling)
s.use_rotations          = True
s.rotation_mode          = "NONE"
s.rotation_factor_random = 1.0
s.phase_factor_random    = 1.0

# Instance render
s.render_type    = "OBJECT"
s.instance_object = instances[0]
s.particle_size  = 1.0
s.size_random    = 0.4

# ── Floor collider ───────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -4))
floor = bpy.context.active_object
floor.name = "Floor"
fl_mat = bpy.data.materials.new("FlMat")
fl_mat.diffuse_color = tuple(bg_color[:3]) + (1.0,)
floor.data.materials.append(fl_mat)

# Add collision modifier to floor
floor.modifiers.new("Collision", type="COLLISION")

# ── Camera ───────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -12, 1.5))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(82), 0, 0)
scene.camera = cam

# ── Lighting ─────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="AREA", location=(4, -6, 8))
key = bpy.context.active_object.data
key.energy = 800; key.size = 6.0

bpy.ops.object.light_add(type="AREA", location=(-4, -4, 5))
fill = bpy.context.active_object.data
fill.energy = 300; fill.size = 4.0

# ── Render ───────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "style": style,
    "count": count,
}
print(f"RESULT:{json.dumps(result)}")
