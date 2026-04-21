"""
Reference-mode Blender scene — reproduces a look derived from a reference image.

Three modes (determined by vision_agent analysis):
  Mode 1 — Empty/Image overlay:  reference shown as a flat image plane in the scene.
            Best for: logos, UI mockups, flat designs.
  Mode 2 — Camera Background:    reference set as the camera background (viewport only,
            shows through during render when 'Background Images' is enabled).
            Best for: product shots, portraits, real-scene references.
  Mode 3 — World / HDRI proxy:   reference mapped as the world environment texture.
            Best for: outdoor scenes, skies, reflective surfaces.

In all modes the script also applies:
  - dominant_colors from vision analysis (drives material palette)
  - lighting_type (studio → area lights; natural → sun; dramatic → spot; neon → emission)
  - camera_angle (front / top / isometric / orbit / close_up)

Args (JSON after --):
    output_path: str
    duration: float
    fps: int
    reference_image_path: str          — local path to reference image
    mode: int                          — 1 | 2 | 3
    dominant_colors: list[str]         — hex codes from vision analysis
    lighting_type: str
    camera_angle: str
    mood: str
    key_objects: list[str]
    corrections: dict                  — from QA loop (applied on retry)
    prompt: str                        — original user prompt
"""
import bpy
import sys
import json
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

argv = sys.argv
args_json = argv[argv.index("--") + 1] if "--" in argv else "{}"
args = json.loads(args_json)

OUTPUT_PATH    = args.get("output_path", "/tmp/ref_render.mp4")
DURATION       = float(args.get("duration", 10.0))
FPS            = int(args.get("fps", 60))
REF_IMG        = args.get("reference_image_path", "")
MODE           = int(args.get("mode", 2))
DOM_COLORS     = args.get("dominant_colors", ["#1a1a2e", "#16213e", "#0f3460"])
LIGHTING       = args.get("lighting_type", "studio")
CAM_ANGLE      = args.get("camera_angle", "front")
MOOD           = args.get("mood", "cinematic")
KEY_OBJECTS    = args.get("key_objects", [])
COMP_FOCUS     = args.get("composition_focus", "centered")
MOVE_STYLE     = args.get("movement_style", "slow_push")
MATERIAL_STYLE = args.get("material_style", "mixed")
SCENE_LAYERS   = args.get("scene_layers", [])
VERIFY_FOCUS   = args.get("verification_focus", [])
NOTES          = args.get("notes", "")
CORRECTIONS    = args.get("corrections", {})
PROMPT         = args.get("prompt", "")
TOTAL_FRAMES   = int(DURATION * FPS)


# ---------------------------------------------------------------------------
# Utility: hex → linear RGB
# ---------------------------------------------------------------------------

def hex_to_linear(hex_str: str) -> tuple[float, float, float, float]:
    hex_str = hex_str.lstrip("#")
    r, g, b = (int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
    # gamma decode (approx)
    r, g, b = r**2.2, g**2.2, b**2.2
    return (r, g, b, 1.0)


# ---------------------------------------------------------------------------
# Scene reset
# ---------------------------------------------------------------------------

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES

# ---------------------------------------------------------------------------
# World / background
# ---------------------------------------------------------------------------

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")

primary_color = hex_to_linear(DOM_COLORS[0]) if DOM_COLORS else (0.02, 0.02, 0.04, 1.0)

if MODE == 3 and REF_IMG and os.path.exists(REF_IMG):
    # World HDRI: map the reference image as the environment
    env_tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    try:
        env_tex.image = bpy.data.images.load(REF_IMG)
    except Exception:
        pass
    world.node_tree.links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
    bg_node.inputs["Strength"].default_value = 1.5
else:
    bg_node.inputs["Color"].default_value = primary_color
    bg_node.inputs["Strength"].default_value = 0.8

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

CAM_PRESETS = {
    "front":      ((0, -12, 0),  (math.radians(90), 0, 0)),
    "top":        ((0, 0, 15),   (0, 0, 0)),
    "isometric":  ((-7, -7, 7),  (math.radians(54.7), 0, math.radians(-45))),
    "orbit":      ((10, -10, 5), (math.radians(65), 0, math.radians(45))),
    "close_up":   ((0, -5, 0),   (math.radians(90), 0, 0)),
}
loc, rot = CAM_PRESETS.get(CAM_ANGLE, CAM_PRESETS["front"])
cam_obj.location = loc
cam_obj.rotation_euler = rot

focus_offsets = {
    "centered": 0.0,
    "left_weighted": -2.0,
    "right_weighted": 2.0,
    "layered_depth": 0.0,
}
cam_target_x = focus_offsets.get(COMP_FOCUS, 0.0)

track_empty = bpy.data.objects.new("CameraTarget", None)
track_empty.location = (cam_target_x, 0, 0.5)
scene.collection.objects.link(track_empty)
track = cam_obj.constraints.new(type="TRACK_TO")
track.target = track_empty
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"

# Mode 2: set reference as camera background image
if MODE == 2 and REF_IMG and os.path.exists(REF_IMG):
    cam_data.show_background_images = True
    bg_img = cam_data.background_images.new()
    try:
        bg_img.image = bpy.data.images.load(REF_IMG)
    except Exception:
        pass
    bg_img.alpha = 0.5
    bg_img.display_depth = "BACK"

# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def add_light(name, light_type, energy, location, rotation=None):
    ld = bpy.data.lights.new(name, type=light_type)
    ld.energy = energy
    lo = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo)
    lo.location = location
    if rotation:
        lo.rotation_euler = rotation
    return lo

if LIGHTING == "studio":
    add_light("Key",  "AREA",  400, (5, -5, 8))
    add_light("Fill", "AREA",  150, (-5, -3, 4))
    add_light("Rim",  "AREA",  200, (0,  6, 6))
elif LIGHTING == "natural":
    sun = add_light("Sun", "SUN", 3, (5, -5, 10))
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))
elif LIGHTING == "dramatic":
    add_light("Spot", "SPOT", 1000, (3, -8, 8),
              rotation=(math.radians(35), 0, math.radians(25)))
elif LIGHTING == "neon":
    # Coloured emission lights
    for i, col in enumerate(DOM_COLORS[:3]):
        lo = add_light(f"Neon{i}", "AREA", 200, (
            (i - 1) * 5, -4, 3
        ))
        lo.data.color = hex_to_linear(col)[:3]
else:
    add_light("Default", "SUN", 2, (5, -5, 10))

# ---------------------------------------------------------------------------
# Key objects (simple geometry proxies)
# ---------------------------------------------------------------------------

def make_material(name: str, color: tuple, roughness: float = 0.4):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        if MATERIAL_STYLE == "glossy":
            bsdf.inputs["Roughness"].default_value = 0.18
            bsdf.inputs["Specular IOR Level"].default_value = 0.7
        elif MATERIAL_STYLE == "matte":
            bsdf.inputs["Roughness"].default_value = 0.72
        elif MATERIAL_STYLE == "glass":
            bsdf.inputs["Transmission Weight"].default_value = 0.78
            bsdf.inputs["Roughness"].default_value = 0.08
    return mat

# Ground plane gives the scene more depth and keeps product/demo renders from floating.
floor_size = 26 if COMP_FOCUS == "layered_depth" else 18
bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(0, 0, -1.2))
floor = bpy.context.active_object
floor_mat = make_material("FloorMat", hex_to_linear(DOM_COLORS[min(1, len(DOM_COLORS) - 1)]), roughness=0.85)
floor.data.materials.append(floor_mat)

# Place simple proxy objects for key_objects (sphere / cube / torus)
SHAPES = ["sphere", "cube", "torus", "plane", "cylinder"]
combined_objects = list(KEY_OBJECTS[:5])
for layer in SCENE_LAYERS[:4]:
    if layer and layer not in combined_objects:
        combined_objects.append(layer)

for i, obj_name in enumerate(combined_objects[:6]):
    col = hex_to_linear(DOM_COLORS[i % len(DOM_COLORS)]) if DOM_COLORS else (0.5, 0.5, 0.5, 1.0)
    x_pos = (i - len(combined_objects) // 2) * 2.8
    y_pos = 0.0
    z_pos = 0.0

    if COMP_FOCUS == "left_weighted":
        x_pos -= 1.6
    elif COMP_FOCUS == "right_weighted":
        x_pos += 1.6
    elif COMP_FOCUS == "layered_depth":
        y_pos = -2.4 + (i * 1.15)
        z_pos = 0.18 * i

    shape = SHAPES[i % len(SHAPES)]

    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(x_pos, y_pos, z_pos))
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.5, location=(x_pos, y_pos, z_pos))
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(location=(x_pos, y_pos, z_pos))
    elif shape == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(x_pos, y_pos, z_pos))
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.8, depth=2.0, location=(x_pos, y_pos, z_pos))

    obj = bpy.context.active_object
    mat = make_material(f"Mat_{obj_name}", col)
    obj.data.materials.append(mat)
    if i == 0:
        obj.scale = (1.25, 1.25, 1.25)
        track_empty.location = (obj.location.x, obj.location.y, obj.location.z + 0.3)

notes_lower = NOTES.lower()
if "close" in notes_lower or "macro" in notes_lower:
    cam_obj.location.y += 2.0
    cam_obj.location.z += 0.3
elif "wide" in notes_lower or "spacious" in notes_lower:
    cam_obj.location.y -= 2.5
elif "minimal" in notes_lower:
    bg_node.inputs["Strength"].default_value = 0.65

# Apply corrections from QA loop if present
if CORRECTIONS.get("lighting_correction"):
    # Re-adjust key light energy as a simple proxy for lighting correction
    for obj in scene.objects:
        if obj.type == "LIGHT" and obj.name == "Key":
            obj.data.energy = obj.data.energy * 1.3

if CORRECTIONS.get("color_correction"):
    # Shift dominant color by tinting background
    bg_node.inputs["Strength"].default_value = 1.2

if CORRECTIONS.get("composition_correction"):
    cam_obj.location.y = cam_obj.location.y - 1.5

if CORRECTIONS.get("object_correction") and combined_objects:
    track_empty.location.z += 0.4

# Simple camera animation adds product-demo motion rather than a static proxy render.
if MOVE_STYLE == "slow_push":
    cam_obj.keyframe_insert(data_path="location", frame=1)
    cam_obj.location.y += 2.0
    cam_obj.location.z += 0.4
    cam_obj.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)
elif MOVE_STYLE == "orbit":
    cam_obj.keyframe_insert(data_path="location", frame=1)
    cam_obj.location.x = loc[0] * -0.85
    cam_obj.location.y = abs(loc[1]) * 0.8
    cam_obj.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)
elif MOVE_STYLE == "parallax":
    cam_obj.keyframe_insert(data_path="location", frame=1)
    track_empty.keyframe_insert(data_path="location", frame=1)
    cam_obj.location.x += 2.5
    track_empty.location.x -= 1.1
    cam_obj.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)
    track_empty.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)

for obj in scene.objects:
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for key in fcurve.keyframe_points:
                key.interpolation = "BEZIER"

# Mode 1: add reference image as flat image plane in the scene
if MODE == 1 and REF_IMG and os.path.exists(REF_IMG):
    try:
        bpy.ops.import_image.to_plane(files=[{"name": REF_IMG}], directory="")
    except AttributeError:
        # Fallback: create a plane with image texture
        bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
        plane = bpy.context.active_object
        mat = bpy.data.materials.new("RefImage")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        tex = nodes.new("ShaderNodeTexImage")
        try:
            tex.image = bpy.data.images.load(REF_IMG)
        except Exception:
            pass
        bsdf = nodes.get("Principled BSDF")
        if bsdf and tex.image:
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        plane.data.materials.append(mat)

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

render = scene.render
render.engine = "CYCLES"
render.resolution_x = 1920
render.resolution_y = 1080
render.image_settings.file_format = "FFMPEG"
render.ffmpeg.format = "MPEG4"
render.ffmpeg.codec = "H264"
render.ffmpeg.constant_rate_factor = "HIGH"
render.filepath = OUTPUT_PATH

scene.cycles.samples = 128
scene.cycles.use_denoising = True

bpy.ops.render.render(animation=True)

print(f"RESULT:{json.dumps({
    'output_path': OUTPUT_PATH,
    'frames': TOTAL_FRAMES,
    'fps': FPS,
    'mode': MODE,
    'composition_focus': COMP_FOCUS,
    'movement_style': MOVE_STYLE,
    'material_style': MATERIAL_STYLE,
    'verification_focus': VERIFY_FOCUS,
})}")
