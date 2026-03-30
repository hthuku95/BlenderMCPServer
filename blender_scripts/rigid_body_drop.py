"""
rigid_body_drop.py — Physics rigid-body drop animation (logo/text/objects fall and collide).

Args (JSON after '--'):
    text:          str  — text to extrude as falling 3D letters (default: "DROP")
    object_type:   "text" | "spheres" | "cubes" | "mixed" (default: "text")
    count:         int  — number of objects if not text (default: 12)
    color:         [R,G,B,A]
    bg_color:      [R,G,B,A]
    style:         "dark" | "bright" | "neon"
    duration:      float (seconds, default: 5)
    output_path:   str
"""
import sys, json, math, random
import bpy  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args        = json.loads(args_json)
text        = args.get("text", "DROP")
obj_type    = args.get("object_type", "text")
count       = int(args.get("count", 12))
color       = args.get("color",    [0.1, 0.6, 1.0, 1.0])
bg_color    = args.get("bg_color", [0.02, 0.02, 0.05, 1.0])
style       = args.get("style", "dark")
duration    = float(args.get("duration", 5.0))
output_path = args.get("output_path", "/tmp/rigid_body_drop.mp4")

fps          = 30
total_frames = int(duration * fps)

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

# ── Material helper ───────────────────────────────────────────────────────────
palette = [
    color,
    [1.0, 0.3, 0.1, 1.0],
    [0.1, 0.9, 0.4, 1.0],
    [0.9, 0.8, 0.0, 1.0],
    [0.6, 0.1, 0.9, 1.0],
    [1.0, 0.5, 0.8, 1.0],
]

def make_mat(name, col):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = tuple(col[:4])
        bsdf.inputs["Metallic"].default_value   = 0.6
        bsdf.inputs["Roughness"].default_value  = 0.2
    mat.diffuse_color = tuple(col[:4])
    return mat

# ── Create rigid-body objects ──────────────────────────────────────────────────
rigid_objects = []

if obj_type == "text" and text:
    # One text object per character, dropped staggered
    chars = list(text.upper()[:8])  # cap at 8 chars
    spacing = 1.4
    start_x = -(len(chars) - 1) * spacing / 2.0
    for ci, ch in enumerate(chars):
        bpy.ops.object.text_add(location=(start_x + ci * spacing, 0, 6 + ci * 1.5))
        obj = bpy.context.active_object
        obj.name = f"Char_{ch}_{ci}"
        obj.data.body    = ch
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.extrude = 0.3
        obj.data.bevel_depth = 0.05
        obj.data.size    = 1.2
        bpy.ops.object.convert(target='MESH')
        mat = make_mat(f"CharMat_{ci}", palette[ci % len(palette)])
        obj.data.materials.append(mat)
        obj.rotation_euler = (random.uniform(-0.2, 0.2),
                               random.uniform(-0.2, 0.2),
                               random.uniform(-0.1, 0.1))
        rigid_objects.append(obj)
else:
    # Generic objects
    for i in range(min(count, 20)):
        x = random.uniform(-4, 4)
        y = random.uniform(-1, 1)
        z = 4 + i * 0.8
        if obj_type == "spheres":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(x, y, z))
        elif obj_type == "cubes":
            bpy.ops.mesh.primitive_cube_add(size=0.7, location=(x, y, z))
        else:  # mixed
            if i % 3 == 0:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(x, y, z))
            elif i % 3 == 1:
                bpy.ops.mesh.primitive_cube_add(size=0.6, location=(x, y, z))
            else:
                bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.6, location=(x, y, z))

        obj = bpy.context.active_object
        obj.name = f"Obj_{i}"
        mat = make_mat(f"ObjMat_{i}", palette[i % len(palette)])
        obj.data.materials.append(mat)
        obj.rotation_euler = (random.uniform(0, 6.28),
                               random.uniform(0, 6.28),
                               random.uniform(0, 6.28))
        rigid_objects.append(obj)

# ── Floor (passive rigid body) ────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -1.5))
floor = bpy.context.active_object
floor.name = "Floor"
fl_mat = make_mat("FlMat", [c * 0.3 for c in bg_color[:3]] + [1.0])
floor.data.materials.append(fl_mat)

# ── Rigid body world ──────────────────────────────────────────────────────────
bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.enabled = True
rbw.substeps_per_frame = 6
rbw.solver_iterations  = 12

# Make floor passive
bpy.context.view_layer.objects.active = floor
floor.select_set(True)
bpy.ops.rigidbody.object_add()
floor.rigid_body.type = "PASSIVE"
floor.rigid_body.collision_shape = "BOX"
floor.select_set(False)

# Make each object active
for obj in rigid_objects:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type        = "ACTIVE"
    obj.rigid_body.mass        = 1.0
    obj.rigid_body.restitution = 0.3
    obj.rigid_body.friction    = 0.6
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.select_set(False)

# ── Bake rigid body by advancing frames ───────────────────────────────────────
# This records actual positions so render doesn't need live sim
scene.frame_set(1)
bpy.context.view_layer.update()
try:
    bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=total_frames)
except Exception:
    # Fallback: manually step and keyframe
    for f in range(1, total_frames + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for obj in rigid_objects:
            if obj.rigid_body:
                obj.keyframe_insert(data_path="location",       frame=f)
                obj.keyframe_insert(data_path="rotation_euler", frame=f)

scene.frame_set(1)

# ── Camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -14, 3))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(82), 0, 0)
cam.data.lens = 45
scene.camera = cam

# ── Lighting ──────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="AREA", location=(5, -6, 8))
key = bpy.context.active_object.data; key.energy = 1000; key.size = 6
bpy.ops.object.light_add(type="AREA", location=(-5, -4, 5))
fill = bpy.context.active_object.data; fill.energy = 300; fill.size = 4
bpy.ops.object.light_add(type="SPOT", location=(0, 5, 6))
back = bpy.context.active_object.data
back.energy = 500; back.spot_size = math.radians(60)

# ── Render ────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "object_type": obj_type,
    "text": text,
}
print(f"RESULT:{json.dumps(result)}")
