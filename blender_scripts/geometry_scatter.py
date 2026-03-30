"""
geometry_scatter.py — Procedural instance scatter using Geometry Nodes.

Scatters instanced objects across a surface (plane, sphere, torus) with optional
animated wave displacement — great for particle-like fields without physics sim.

Args (JSON after '--'):
    instance_type: "cubes" | "spheres" | "stars" | "arrows" | "crystals"
    surface:       "plane" | "sphere" | "torus" | "grid"
    count:         int  (default: 200)
    primary_color: [R,G,B,A]
    secondary_color:[R,G,B,A]
    bg_color:      [R,G,B,A]
    animated:      bool (default: True — wave displacement over time)
    scale:         float (instance scale, default: 1.0)
    duration:      float (seconds, default: 8)
    output_path:   str
"""
import sys, json, math, random
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args          = json.loads(args_json)
inst_type     = args.get("instance_type", "spheres")
surface       = args.get("surface",       "plane")
count         = int(args.get("count", 200))
primary_color = args.get("primary_color",   [0.1, 0.6, 1.0, 1.0])
second_color  = args.get("secondary_color", [1.0, 0.3, 0.1, 1.0])
bg_color      = args.get("bg_color",        [0.02, 0.02, 0.06, 1.0])
animated      = bool(args.get("animated", True))
inst_scale    = float(args.get("scale", 1.0))
duration      = float(args.get("duration", 8.0))
output_path   = args.get("output_path", "/tmp/geometry_scatter.mp4")

fps          = 30
total_frames = int(duration * fps)
random.seed(42)

# ── Scene ─────────────────────────────────────────────────────────────────────
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
scene.display.shading.show_cavity  = True
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
    bg_node.inputs["Color"].default_value    = tuple(bg_color[:3]) + (1.0,)
    bg_node.inputs["Strength"].default_value = 1.0

def make_mat(name, col, metallic=0.3, roughness=0.25):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = tuple(col[:4])
        bsdf.inputs["Metallic"].default_value   = metallic
        bsdf.inputs["Roughness"].default_value  = roughness
    mat.diffuse_color = tuple(col[:4])
    return mat

# ── Instance object ─────────────────────────────────────────────────────────
base_scale = 0.12 * inst_scale

if inst_type == "cubes":
    bpy.ops.mesh.primitive_cube_add(size=base_scale * 2, location=(1000, 0, 0))
elif inst_type == "stars":
    # Icosphere as "star" (low-poly)
    bpy.ops.mesh.primitive_ico_sphere_add(radius=base_scale, subdivisions=1, location=(1000, 0, 0))
elif inst_type == "arrows":
    bpy.ops.mesh.primitive_cone_add(radius1=base_scale * 0.6, radius2=0,
                                     depth=base_scale * 3, location=(1000, 0, 0))
elif inst_type == "crystals":
    bpy.ops.mesh.primitive_ico_sphere_add(radius=base_scale, subdivisions=0, location=(1000, 0, 0))
else:  # spheres
    bpy.ops.mesh.primitive_uv_sphere_add(radius=base_scale, segments=6, rings=4, location=(1000, 0, 0))

instance_obj = bpy.context.active_object
instance_obj.name = "InstanceObj"
inst_mat = make_mat("InstMat", primary_color, metallic=0.5, roughness=0.2)
instance_obj.data.materials.append(inst_mat)
instance_obj.hide_viewport = False
instance_obj.hide_render   = False

# Second color variant
if inst_type == "cubes":
    bpy.ops.mesh.primitive_cube_add(size=base_scale * 2, location=(1001, 0, 0))
elif inst_type == "crystals":
    bpy.ops.mesh.primitive_ico_sphere_add(radius=base_scale, subdivisions=0, location=(1001, 0, 0))
else:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=base_scale * 0.7, segments=6, rings=4, location=(1001, 0, 0))
inst2 = bpy.context.active_object
inst2.name = "InstanceObj2"
inst_mat2 = make_mat("InstMat2", second_color, metallic=0.5, roughness=0.2)
inst2.data.materials.append(inst_mat2)

# ── Scatter logic (manual — no Geometry Nodes required) ─────────────────────
# We manually place instances since GN setup is complex in headless mode.
# Animate Z via a sine wave keyed every 3 frames for performance.
scatter_objs = []

if surface == "sphere":
    # Fibonacci sphere distribution
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        y_off = 1 - (i / (count - 1)) * 2
        r     = math.sqrt(1 - y_off * y_off) * 5
        theta = golden * i
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        z = y_off * 5
        use_inst = instance_obj if i % 3 != 0 else inst2
        dup = use_inst.copy()
        dup.data = use_inst.data  # linked data
        bpy.context.collection.objects.link(dup)
        bpy.context.view_layer.objects.active = dup
        dup.location = (x, y, z)
        dup.hide_viewport = False; dup.hide_render = False
        scatter_objs.append((dup, x, y, z))

elif surface == "torus":
    major_r = 4.0
    for i in range(count):
        theta = i / count * 2 * math.pi
        phi   = (i * 7) / count * 2 * math.pi
        minor_r = 1.5
        x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
        y = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
        z = minor_r * math.sin(phi)
        use_inst = instance_obj if i % 3 != 0 else inst2
        dup = use_inst.copy()
        dup.data = use_inst.data  # linked data
        bpy.context.collection.objects.link(dup)
        bpy.context.view_layer.objects.active = dup
        dup.location = (x, y, z)
        dup.hide_viewport = False; dup.hide_render = False
        scatter_objs.append((dup, x, y, z))

elif surface == "grid":
    side = int(math.sqrt(count)) + 1
    spacing = 8.0 / side
    for ix in range(side):
        for iy in range(side):
            if ix * side + iy >= count:
                break
            x = (ix - side / 2) * spacing
            y = (iy - side / 2) * spacing
            z = 0
            use_inst = instance_obj if (ix + iy) % 2 == 0 else inst2
            bpy.ops.object.duplicate({'selected_objects': [use_inst]}, linked=True)
            dup = bpy.context.active_object
            dup.location = (x, y, z)
            dup.hide_viewport = False; dup.hide_render = False
            scatter_objs.append((dup, x, y, z))

else:  # plane — random distribution
    for i in range(count):
        x = random.uniform(-6, 6)
        y = random.uniform(-4, 4)
        z = 0
        use_inst = instance_obj if i % 3 != 0 else inst2
        dup = use_inst.copy()
        dup.data = use_inst.data  # linked data
        bpy.context.collection.objects.link(dup)
        bpy.context.view_layer.objects.active = dup
        dup.location = (x, y, z)
        dup.rotation_euler = (random.uniform(0, 6.28),
                               random.uniform(0, 6.28),
                               random.uniform(0, 6.28))
        dup.hide_viewport = False; dup.hide_render = False
        scatter_objs.append((dup, x, y, z))

# ── Animated wave displacement ─────────────────────────────────────────────
if animated:
    # Key every 3 frames for performance
    keyframe_interval = 3
    for f in range(1, total_frames + 1, keyframe_interval):
        t = (f - 1) / fps
        for dup, ox, oy, oz in scatter_objs:
            wave_z = math.sin(ox * 0.8 + t * 2.0) * math.cos(oy * 0.8 + t * 1.5) * 0.4
            dup.location.z = oz + wave_z
            dup.keyframe_insert(data_path="location", frame=f)

    # Smooth fcurves
    for action in bpy.data.actions:
        for fc in action.fcurves:
            fc.extrapolation = 'LINEAR'

# ── Camera ────────────────────────────────────────────────────────────────────
if surface == "sphere":
    cam_loc = (0, -18, 4)
    cam_rot = (math.radians(80), 0, 0)
elif surface == "torus":
    cam_loc = (0, -15, 5)
    cam_rot = (math.radians(75), 0, 0)
else:
    cam_loc = (0, -14, 8)
    cam_rot = (math.radians(65), 0, 0)

bpy.ops.object.camera_add(location=cam_loc)
cam = bpy.context.active_object
cam.rotation_euler = cam_rot
cam.data.lens = 40
scene.camera = cam

# Slow orbit for sphere/torus
if animated and surface in ("sphere", "torus"):
    for f in range(1, total_frames + 1, 5):
        t     = (f - 1) / total_frames
        angle = t * math.pi * 0.5  # quarter turn
        r     = math.sqrt(cam_loc[0]**2 + cam_loc[1]**2)
        cam.location.x = r * math.sin(angle)
        cam.location.y = -r * math.cos(angle)
        dx, dy = -cam.location.x, -cam.location.y
        cam.rotation_euler.z = math.atan2(dx, -dy)
        cam.keyframe_insert(data_path="location",       frame=f)
        cam.keyframe_insert(data_path="rotation_euler", frame=f)

# ── Lighting ──────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
sun = bpy.context.active_object.data
sun.energy = 4.0; sun.angle = math.radians(10)

bpy.ops.object.light_add(type="AREA", location=(-5, -5, 6))
fill = bpy.context.active_object.data
fill.energy = 200; fill.size = 8

# ── Render ────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "instance_type": inst_type,
    "surface": surface,
    "count": len(scatter_objs),
}
print(f"RESULT:{json.dumps(result)}")
