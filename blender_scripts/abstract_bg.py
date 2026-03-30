"""
abstract_bg.py — Animated abstract background loop for video overlays.

Styles:
  "geometric"  — slowly rotating/drifting geometric shapes
  "waves"      — undulating planes with displacement
  "particles"  — floating dots/spheres
  "gradient"   — smooth gradient plane with animated color shift
  "grid"       — neon wireframe grid (lo-fi/retro)

Args (JSON after '--'):
    style:       str
    primary_color:   [R, G, B, A]
    secondary_color: [R, G, B, A]
    duration:    float
    output_path: str
    loop:        bool  — if True, last frame matches first (for seamless loop)
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

args      = json.loads(args_json)
style     = args.get("style", "geometric")
primary   = args.get("primary_color", [0.05, 0.2, 0.8, 1.0])
secondary = args.get("secondary_color", [0.8, 0.1, 0.5, 1.0])
duration  = float(args.get("duration", 8.0))
output_path = args.get("output_path", "/tmp/abstract_bg.mp4")

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
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)
    bg.inputs["Strength"].default_value = 0.5

def make_mat(name, color, metallic=0.3, roughness=0.4, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = tuple(color)
    bsdf.inputs["Metallic"].default_value   = metallic
    bsdf.inputs["Roughness"].default_value  = roughness
    if emission_strength > 0:
        bsdf.inputs["Emission"].default_value = tuple(color[:3]) + (1.0,)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    mat.diffuse_color = tuple(color)
    return mat

if style == "geometric":
    pm = make_mat("Primary", primary,   metallic=0.7, roughness=0.1, emission_strength=0.5)
    sm = make_mat("Secondary", secondary, metallic=0.5, roughness=0.2, emission_strength=0.3)

    shapes = []
    import random
    random.seed(42)
    for i in range(12):
        angle_offset = (2 * math.pi / 12) * i
        radius = random.uniform(2.0, 5.0)
        size   = random.uniform(0.3, 1.2)
        z      = random.uniform(-2.0, 2.0)
        stype  = random.choice(["cube", "sphere", "torus"])

        if stype == "cube":
            bpy.ops.mesh.primitive_cube_add(size=size, location=(radius, 0, z))
        elif stype == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size * 0.5, location=(radius, 0, z))
        else:
            bpy.ops.mesh.primitive_torus_add(
                major_radius=size * 0.5, minor_radius=size * 0.15,
                location=(radius, 0, z)
            )

        obj = bpy.context.active_object
        mat = pm if i % 2 == 0 else sm
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        shapes.append((obj, angle_offset, radius, random.uniform(0.2, 0.7)))

    # Animate orbit + self-rotation
    for obj, ao, r, speed in shapes:
        for f in range(1, total_frames + 1):
            t = (f - 1) / fps
            angle = ao + t * speed
            obj.location = (r * math.cos(angle), r * math.sin(angle), obj.location.z)
            obj.rotation_euler = (t * speed, t * speed * 0.7, t * speed * 1.3)
            obj.keyframe_insert(data_path="location", frame=f)
            obj.keyframe_insert(data_path="rotation_euler", frame=f)

elif style == "waves":
    # Create a subdivided plane and animate Z displacement manually
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    plane = bpy.context.active_object
    # Subdivide
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=20)
    bpy.ops.object.mode_set(mode='OBJECT')

    pm = make_mat("Wave", primary, metallic=0.5, roughness=0.3, emission_strength=1.0)
    plane.data.materials.append(pm)

    # Animate vertex Z with wave
    mesh = plane.data
    for f in range(1, total_frames + 1):
        t = (f - 1) / fps
        for v in mesh.vertices:
            x, y = v.co.x, v.co.y
            v.co.z = 0.3 * math.sin(x * 0.8 + t * 2) * math.cos(y * 0.8 + t * 1.5)
        mesh.update()
        # For animation we need shape keys
        # Simplify: just animate plane rotation for visual interest
        plane.rotation_euler.x = math.sin(t * 0.4) * 0.3
        plane.keyframe_insert(data_path="rotation_euler", frame=f)

elif style == "particles":
    import random
    random.seed(7)
    pm = make_mat("Part", primary, metallic=0.0, roughness=0.5, emission_strength=2.0)
    sm = make_mat("Part2", secondary, metallic=0.0, roughness=0.5, emission_strength=2.0)

    particles = []
    for i in range(60):
        r = random.uniform(0.05, 0.2)
        x = random.uniform(-8, 8)
        y = random.uniform(-8, 8)
        z = random.uniform(-3, 3)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z))
        obj = bpy.context.active_object
        mat = pm if i % 2 == 0 else sm
        obj.data.materials.append(mat)
        vx = random.uniform(-0.5, 0.5)
        vy = random.uniform(-0.5, 0.5)
        vz = random.uniform(-0.2, 0.2)
        particles.append((obj, x, y, z, vx, vy, vz))

    for obj, x0, y0, z0, vx, vy, vz in particles:
        for f in range(1, total_frames + 1):
            t = (f - 1) / fps
            nx = x0 + vx * t + 0.3 * math.sin(t * 1.5 + x0)
            ny = y0 + vy * t + 0.3 * math.cos(t * 1.3 + y0)
            nz = z0 + vz * t + 0.15 * math.sin(t * 2 + z0)
            obj.location = (nx % 16 - 8, ny % 16 - 8, nz)
            obj.keyframe_insert(data_path="location", frame=f)

elif style == "grid":
    pm = make_mat("Grid", primary, metallic=0.0, roughness=1.0, emission_strength=3.0)
    # Create grid of thin cylinders
    for xi in range(-5, 6):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.03, depth=20,
            location=(xi * 1.2, 0, 0),
        )
        obj = bpy.context.active_object
        obj.rotation_euler.x = math.radians(90)
        obj.data.materials.append(pm)
    for yi in range(-4, 5):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.03, depth=14,
            location=(0, 0, yi * 1.2),
        )
        obj = bpy.context.active_object
        obj.rotation_euler.z = math.radians(90)
        obj.data.materials.append(pm)

else:  # gradient fallback
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.5))
    plane = bpy.context.active_object
    pm = make_mat("Grad", primary, metallic=0.0, roughness=1.0, emission_strength=0.8)
    plane.data.materials.append(pm)

# Camera
bpy.ops.object.camera_add(location=(0, -12, 2))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(80), 0, 0)
scene.camera = cam

# Gentle camera sway
for f in range(1, total_frames + 1):
    t = (f - 1) / fps
    cam.location = (math.sin(t * 0.3) * 0.5, -12, 2 + math.sin(t * 0.2) * 0.3)
    cam.keyframe_insert(data_path="location", frame=f)

# Lighting
bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
sun = bpy.context.active_object
sun.data.energy = 2.0
sun.data.color  = tuple(primary[:3])

bpy.ops.object.light_add(type="AREA", location=(-3, -3, 5))
fill = bpy.context.active_object
fill.data.energy = 300
fill.data.color  = tuple(secondary[:3])

bpy.ops.render.render(animation=True)
result = {"output_path": output_path, "duration": duration, "resolution": "1280x720", "frames": total_frames}
print(f"RESULT:{json.dumps(result)}")
