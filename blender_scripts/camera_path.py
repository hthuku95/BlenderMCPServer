"""
camera_path.py — Smooth camera fly-through / orbit animation on a spline path.

Args (JSON after '--'):
    path_type:   "orbit" | "helix" | "arc" | "dolly_zoom" | "flythrough"
    subject:     "spheres" | "cubes" | "text" | "abstract" | "landscape"
    title:       str — optional 3D text in scene centre
    color:       [R,G,B,A]
    bg_color:    [R,G,B,A]
    style:       "cinematic" | "minimal" | "neon"
    duration:    float (seconds, default: 8)
    output_path: str
"""
import sys, json, math
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args        = json.loads(args_json)
path_type   = args.get("path_type",  "orbit")
subject     = args.get("subject",    "abstract")
title       = args.get("title",      "")
color       = args.get("color",      [0.1, 0.6, 1.0, 1.0])
bg_color    = args.get("bg_color",   [0.02, 0.02, 0.06, 1.0])
style       = args.get("style",      "cinematic")
duration    = float(args.get("duration", 8.0))
output_path = args.get("output_path", "/tmp/camera_path.mp4")

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
scene.display.shading.show_object_outline = (style == "neon")
if style == "neon":
    scene.display.shading.object_outline_color = tuple(color[:3])
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

palette = [
    color,
    [1.0, 0.3, 0.1, 1.0], [0.1, 0.9, 0.3, 1.0],
    [0.9, 0.8, 0.0, 1.0], [0.8, 0.1, 0.9, 1.0],
    [0.0, 0.8, 0.9, 1.0],
]

# ── Subject objects ────────────────────────────────────────────────────────────
target_loc = Vector((0, 0, 0))

if subject == "text" and title:
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.body    = title.upper()
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.extrude = 0.3
    obj.data.bevel_depth = 0.05
    obj.data.size    = 1.5
    mat = make_mat("TitleMat", color, metallic=0.7, roughness=0.15)
    obj.data.materials.append(mat)

elif subject == "spheres":
    positions = [(0,0,0),(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2)]
    for i, pos in enumerate(positions):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=pos)
        obj = bpy.context.active_object
        mat = make_mat(f"SpMat_{i}", palette[i % len(palette)], metallic=0.5)
        obj.data.materials.append(mat)

elif subject == "cubes":
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            h = abs(ix * iy) + 0.5
            bpy.ops.mesh.primitive_cube_add(size=0.8,
                                            location=(ix * 1.5, iy * 1.5, h / 2))
            obj = bpy.context.active_object
            obj.scale.z = h
            mat = make_mat(f"CuMat_{ix}{iy}", palette[(ix + iy + 2) % len(palette)])
            obj.data.materials.append(mat)

elif subject == "landscape":
    # Simple flat terrain with varied height cubes
    for ix in range(-3, 4):
        for iy in range(-2, 3):
            h = math.sin(ix * 0.8) * math.cos(iy * 0.8) * 1.5 + 1.6
            bpy.ops.mesh.primitive_cube_add(size=0.9,
                                            location=(ix * 1.0, iy * 1.0, h / 2))
            obj = bpy.context.active_object
            obj.scale.z = h
            ci = int((h / 3.0) * len(palette)) % len(palette)
            mat = make_mat(f"TrMat_{ix}{iy}", palette[ci])
            obj.data.materials.append(mat)

else:  # abstract
    import random; random.seed(42)
    shapes = ['cube', 'sphere', 'cylinder', 'torus']
    for i in range(18):
        angle = i / 18 * 2 * math.pi
        r = random.uniform(2, 5)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = random.uniform(-1.5, 2.5)
        shape = shapes[i % len(shapes)]
        if shape == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.4, 0.9), location=(x, y, z))
        elif shape == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.3, 0.7), location=(x, y, z))
        elif shape == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=random.uniform(0.5, 1.5), location=(x, y, z))
        else:
            bpy.ops.mesh.primitive_torus_add(major_radius=0.4, minor_radius=0.1, location=(x, y, z))
        obj = bpy.context.active_object
        obj.rotation_euler = (random.uniform(0, 6.28), random.uniform(0, 6.28), 0)
        mat = make_mat(f"AbMat_{i}", palette[i % len(palette)])
        obj.data.materials.append(mat)

# ── Camera ─────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -8, 2))
cam_obj = bpy.context.active_object
cam_obj.name = "Camera"
scene.camera = cam_obj
cam_obj.data.lens = 40

# Animated camera — keyframe along path type
if path_type == "orbit":
    radius = 9.0
    tilt   = math.radians(25)
    for f in range(1, total_frames + 1):
        t     = (f - 1) / total_frames
        angle = t * 2 * math.pi
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 3.0 + math.sin(t * math.pi) * 1.5
        cam_obj.location = (x, y, z)
        # Point at origin
        dx, dy, dz = (0 - x), (0 - y), (0 - z)
        cam_obj.rotation_euler = (
            math.atan2(math.sqrt(dx**2 + dy**2), -dz) - math.pi,
            0,
            math.atan2(dx, -dy) + math.pi,
        )
        cam_obj.keyframe_insert(data_path="location",       frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif path_type == "helix":
    for f in range(1, total_frames + 1):
        t     = (f - 1) / total_frames
        angle = t * 3 * math.pi
        r     = 8.0
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = -2 + t * 6
        cam_obj.location = (x, y, z)
        dx, dy, dz = (0 - x), (0 - y), (0 - z)
        cam_obj.rotation_euler = (
            math.atan2(math.sqrt(dx**2 + dy**2), -dz) - math.pi,
            0,
            math.atan2(dx, -dy) + math.pi,
        )
        cam_obj.keyframe_insert(data_path="location",       frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif path_type == "arc":
    # Swing from left to right in a smooth arc
    for f in range(1, total_frames + 1):
        t = (f - 1) / total_frames
        # Ease in-out
        t_ease = t * t * (3 - 2 * t)
        angle  = math.radians(-60) + t_ease * math.radians(120)
        x = 10 * math.sin(angle)
        y = -10 * math.cos(angle)
        z = 3.0
        cam_obj.location = (x, y, z)
        dx, dy, dz = (0 - x), (0 - y), (0 - z)
        cam_obj.rotation_euler = (
            math.atan2(math.sqrt(dx**2 + dy**2), -dz) - math.pi,
            0,
            math.atan2(dx, -dy) + math.pi,
        )
        cam_obj.keyframe_insert(data_path="location",       frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif path_type == "dolly_zoom":
    for f in range(1, total_frames + 1):
        t = (f - 1) / total_frames
        dist = 14 - t * 10   # camera moves in
        fov  = 20 + t * 55   # fov widens
        cam_obj.location       = (0, -dist, 1.5)
        cam_obj.rotation_euler = (math.radians(87), 0, 0)
        cam_obj.data.lens      = 70 - t * 50  # zoom out while pushing in
        cam_obj.keyframe_insert(data_path="location",       frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)
        cam_obj.data.keyframe_insert(data_path="lens",       frame=f)

else:  # flythrough — linear push-in
    for f in range(1, total_frames + 1):
        t = (f - 1) / total_frames
        cam_obj.location       = (0, -12 + t * 8, 2 - t * 1.5)
        cam_obj.rotation_euler = (math.radians(88), 0, 0)
        cam_obj.keyframe_insert(data_path="location",       frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)

# Smooth fcurves
for action in bpy.data.actions:
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR' if total_frames > 60 else 'BEZIER'

# ── Lighting ───────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
sun = bpy.context.active_object.data
sun.energy = 4.0; sun.angle = math.radians(5)

bpy.ops.object.light_add(type="AREA", location=(-5, -5, 5))
fill = bpy.context.active_object.data; fill.energy = 200; fill.size = 8

# ── Render ─────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "path_type": path_type,
    "subject": subject,
}
print(f"RESULT:{json.dumps(result)}")
