"""
toon_scene.py — NPR cartoon / toon-shaded Blender scene with object outlines.

Args (JSON after '--'):
    subject:        "characters" | "robots" | "landscape" | "abstract" | "logo"
    title:          str — optional 3D text
    outline_color:  [R,G,B,A]
    primary_color:  [R,G,B,A]
    bg_color:       [R,G,B,A]
    outline_width:  float (default: 1.5, range 0.5–5.0)
    flat_shading:   bool (default: True, uses FLAT color mode — true cartoon look)
    animated:       bool (default: True, objects slowly rotate/bob)
    duration:       float (seconds, default: 6)
    output_path:    str
"""
import sys, json, math, random
import bpy  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args          = json.loads(args_json)
subject       = args.get("subject",       "abstract")
title         = args.get("title",         "")
outline_color = args.get("outline_color", [0.0, 0.0, 0.0, 1.0])
primary_color = args.get("primary_color", [0.2, 0.6, 1.0, 1.0])
bg_color      = args.get("bg_color",      [1.0, 0.97, 0.88, 1.0])
outline_width = float(args.get("outline_width", 1.5))
flat_shading  = bool(args.get("flat_shading", True))
animated      = bool(args.get("animated", True))
duration      = float(args.get("duration", 6.0))
output_path   = args.get("output_path", "/tmp/toon_scene.mp4")

fps          = 30
total_frames = int(duration * fps)
random.seed(7)

# ── Scene ─────────────────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end   = total_frames
scene.render.fps  = fps
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_WORKBENCH"

# Workbench NPR settings
shading = scene.display.shading
shading.light      = "FLAT" if flat_shading else "STUDIO"
shading.color_type = "MATERIAL"
shading.show_object_outline   = True
shading.object_outline_color  = tuple(outline_color[:3])
shading.show_shadows          = not flat_shading
shading.show_cavity           = not flat_shading

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

# ── Palette ─────────────────────────────────────────────────────────────────
palette = [
    primary_color,
    [1.0, 0.35, 0.25, 1.0],
    [0.25, 0.8,  0.35, 1.0],
    [1.0, 0.85, 0.15, 1.0],
    [0.8, 0.25,  0.85, 1.0],
    [0.15, 0.75, 0.85, 1.0],
    [1.0, 0.55,  0.15, 1.0],
]

def make_mat(name, col):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(col[:4])
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = tuple(col[:4])
        bsdf.inputs["Roughness"].default_value  = 1.0  # flat / no gloss
        bsdf.inputs["Metallic"].default_value   = 0.0
        bsdf.inputs["Specular"].default_value   = 0.0
    return mat

# ── Subject objects ────────────────────────────────────────────────────────────
all_objs = []

if subject == "robots":
    # Simple blocky robot from primitives
    for ri in range(2):
        ox = -2.5 + ri * 5
        # Body
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(ox, 0, 0.5))
        body = bpy.context.active_object
        body.scale = (0.9, 0.6, 1.2)
        mat = make_mat(f"Body_{ri}", palette[ri * 2])
        body.data.materials.append(mat)
        # Head
        bpy.ops.mesh.primitive_cube_add(size=0.7, location=(ox, 0, 1.7))
        head = bpy.context.active_object
        mat2 = make_mat(f"Head_{ri}", palette[ri * 2 + 1])
        head.data.materials.append(mat2)
        # Eyes
        for ey, ex in [(-0.15, -0.18), (-0.15, 0.18)]:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08,
                                                  location=(ox + ex, -0.35, 1.75))
            eye = bpy.context.active_object
            em = make_mat("Eye", [1.0, 1.0, 0.0, 1.0])
            eye.data.materials.append(em)
            all_objs.append(eye)
        # Legs
        for lx in (-0.3, 0.3):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=0.9,
                                                 location=(ox + lx, 0, -0.45))
            leg = bpy.context.active_object
            leg.data.materials.append(mat)
            all_objs.append(leg)
        all_objs += [body, head]

elif subject == "landscape":
    # Cartoon hills
    for i in range(7):
        x = i * 2 - 6
        h = 1 + math.sin(i * 1.2) * 0.8
        bpy.ops.mesh.primitive_uv_sphere_add(radius=h, location=(x, 1.5, 0))
        hill = bpy.context.active_object
        hill.scale.z = 0.5
        mat = make_mat(f"Hill_{i}", palette[i % len(palette)])
        hill.data.materials.append(mat)
        all_objs.append(hill)
    # Ground
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 2, -0.3))
    gnd = bpy.context.active_object
    gnd.data.materials.append(make_mat("Gnd", [0.2, 0.7, 0.2, 1.0]))
    all_objs.append(gnd)

elif subject == "logo" and title:
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.data.body    = title.upper()
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.extrude = 0.25
    obj.data.size    = 1.5
    mat = make_mat("LogoMat", primary_color)
    obj.data.materials.append(mat)
    all_objs.append(obj)
    # Halo ring
    bpy.ops.mesh.primitive_torus_add(major_radius=3.0, minor_radius=0.08, location=(0, 0, 0))
    ring = bpy.context.active_object
    ring.data.materials.append(make_mat("Ring", [1.0, 0.85, 0.1, 1.0]))
    all_objs.append(ring)

else:  # abstract / characters
    shapes = ['cube', 'sphere', 'cylinder', 'torus', 'cone']
    for i in range(14):
        angle = i / 14 * 2 * math.pi
        r = 3.5 if i > 6 else 1.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = random.uniform(-0.5, 1.0)
        s = shapes[i % len(shapes)]
        if s == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.5, 1.0), location=(x, y, z))
        elif s == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.3, 0.65), location=(x, y, z))
        elif s == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=random.uniform(0.6, 1.4), location=(x, y, z))
        elif s == 'torus':
            bpy.ops.mesh.primitive_torus_add(major_radius=0.5, minor_radius=0.15, location=(x, y, z))
        else:
            bpy.ops.mesh.primitive_cone_add(radius1=0.4, depth=0.9, location=(x, y, z))
        obj = bpy.context.active_object
        mat = make_mat(f"AbsMat_{i}", palette[i % len(palette)])
        obj.data.materials.append(mat)
        all_objs.append(obj)

# ── Optional title text ────────────────────────────────────────────────────────
if title and subject != "logo":
    bpy.ops.object.text_add(location=(0, -3.5, 2.5))
    t_obj = bpy.context.active_object
    t_obj.data.body    = title.upper()
    t_obj.data.align_x = "CENTER"
    t_obj.data.size    = 0.7
    t_obj.data.extrude = 0.08
    mat = make_mat("TitleM", primary_color)
    t_obj.data.materials.append(mat)
    all_objs.append(t_obj)

# ── Animation — gentle float / rotation ───────────────────────────────────────
if animated:
    for i, obj in enumerate(all_objs[:10]):
        offset = i * 0.4
        for f in range(1, total_frames + 1):
            t = (f - 1) / fps
            obj.location.z = obj.location.z + math.sin(t * 1.5 + offset) * 0.0015
            obj.rotation_euler.z = (t * 0.3 + offset) % (2 * math.pi)
            if f % 5 == 0:  # keyframe every 5 frames for performance
                obj.keyframe_insert(data_path="location", frame=f)
                obj.keyframe_insert(data_path="rotation_euler", frame=f)

# ── Camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -10, 3))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(80), 0, 0)
scene.camera = cam

# ── Lighting ──────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
sun = bpy.context.active_object.data
sun.energy = 5.0

# ── Render ─────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "subject": subject,
}
print(f"RESULT:{json.dumps(result)}")
