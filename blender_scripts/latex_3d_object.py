"""
Option A — Import a LaTeX SVG as a 3D curve object in Blender.

Workflow:
  1. Import SVG via bpy.ops.import_curve.svg()            (creates Curve objects)
  2. Convert curves → mesh                                  (optional, for extrude)
  3. Extrude the mesh to give depth                         (3D effect)
  4. Centre in scene, apply emission material               (glowing equation look)
  5. Animate: fade-in via material alpha keyframes
  6. Render to MP4

Args (JSON string as last argv):
    svg_path: str           — absolute path to the LaTeX SVG file
    output_path: str        — destination MP4 path
    duration: float         — clip length in seconds (default 8.0)
    fps: int                — frame rate (default 60)
    background_style: str   — "dark" | "light" | "transparent"
    extrude_depth: float    — extrusion depth in Blender units (default 0.05)
    emit_color: list[float] — RGB emit colour, 0-1 each (default [0.9, 0.9, 1.0])
"""
import bpy
import sys
import json
import os
import math

# --------------------------------------------------------------------------
# Parse args
# --------------------------------------------------------------------------

argv = sys.argv
args_json = argv[argv.index("--") + 1] if "--" in argv else "{}"
args = json.loads(args_json)

SVG_PATH        = args.get("svg_path", "")
OUTPUT_PATH     = args.get("output_path", "/tmp/latex_3d.mp4")
DURATION        = float(args.get("duration", 8.0))
FPS             = int(args.get("fps", 60))
BG_STYLE        = args.get("background_style", "dark")
EXTRUDE_DEPTH   = float(args.get("extrude_depth", 0.05))
EMIT_COLOR      = args.get("emit_color", [0.9, 0.9, 1.0])

TOTAL_FRAMES    = int(DURATION * FPS)
FADE_IN_FRAMES  = int(FPS * 0.8)   # 0.8 s fade-in

# --------------------------------------------------------------------------
# Scene reset
# --------------------------------------------------------------------------

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES

# --------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")

if BG_STYLE == "dark":
    bg_node.inputs["Color"].default_value = (0.02, 0.02, 0.04, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
elif BG_STYLE == "light":
    bg_node.inputs["Color"].default_value = (0.95, 0.95, 0.97, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
else:  # transparent — keep black but enable transparency in render
    bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.render.film_transparent = True

# --------------------------------------------------------------------------
# Import SVG
# --------------------------------------------------------------------------

if not SVG_PATH or not os.path.exists(SVG_PATH):
    raise FileNotFoundError(f"SVG not found: {SVG_PATH}")

bpy.ops.import_curve.svg(filepath=SVG_PATH)

# All newly-imported objects are selected
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == "CURVE"]

if not imported_objects:
    raise RuntimeError("SVG import produced no Curve objects — check the SVG file")

# --------------------------------------------------------------------------
# Convert curves → mesh and join into one object
# --------------------------------------------------------------------------

# Deselect all
bpy.ops.object.select_all(action="DESELECT")

for obj in imported_objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = imported_objects[0]

# Convert to mesh
bpy.ops.object.convert(target="MESH")

# Extrude along Z via solidify modifier (non-destructive)
for obj in bpy.context.selected_objects:
    mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    mod.thickness = EXTRUDE_DEPTH
    mod.offset = 0.0

# Join into single mesh
bpy.ops.object.join()
eq_obj = bpy.context.active_object
eq_obj.name = "LatexEquation"

# --------------------------------------------------------------------------
# Centre and scale to fit in camera view
# --------------------------------------------------------------------------

bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
eq_obj.location = (0, 0, 0)

# Scale so that the equation fits roughly 80% of frame width (assume camera at z=10, fov~50°)
# Bounding box extent in X
bbox = [eq_obj.matrix_world @ v.co for v in eq_obj.data.vertices]
xs = [v.x for v in bbox]
ys = [v.y for v in bbox]
width = max(xs) - min(xs) if xs else 1.0
height = max(ys) - min(ys) if ys else 1.0

target_width = 8.0  # Blender units (world space)
scale = target_width / max(width, 0.001)
eq_obj.scale = (scale, scale, scale)
bpy.ops.object.transform_apply(scale=True)

# --------------------------------------------------------------------------
# Emission material with fade-in
# --------------------------------------------------------------------------

mat = bpy.data.materials.new(name="LatexEmit")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

nodes.clear()

output_node = nodes.new("ShaderNodeOutputMaterial")
emission = nodes.new("ShaderNodeEmission")
transparent = nodes.new("ShaderNodeBsdfTransparent")
mix = nodes.new("ShaderNodeMixShader")

links.new(transparent.outputs["BSDF"], mix.inputs[1])
links.new(emission.outputs["Emission"], mix.inputs[2])
links.new(mix.outputs["Shader"], output_node.inputs["Surface"])

emission.inputs["Color"].default_value = (*EMIT_COLOR, 1.0)
emission.inputs["Strength"].default_value = 3.0

# Animate mix.inputs["Fac"] from 0 (transparent) to 1 (fully visible)
mix.inputs["Fac"].default_value = 0.0
mix.inputs["Fac"].keyframe_insert(data_path="default_value", frame=1)
mix.inputs["Fac"].default_value = 1.0
mix.inputs["Fac"].keyframe_insert(data_path="default_value", frame=FADE_IN_FRAMES)

if eq_obj.data.materials:
    eq_obj.data.materials[0] = mat
else:
    eq_obj.data.materials.append(mat)

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

cam_data = bpy.data.cameras.new("Camera")
cam_data.type = "PERSP"
cam_data.lens = 50
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -10, 0)
cam_obj.rotation_euler = (math.radians(90), 0, 0)

# --------------------------------------------------------------------------
# Light
# --------------------------------------------------------------------------

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 2.0
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (5, -8, 8)

# --------------------------------------------------------------------------
# Render settings
# --------------------------------------------------------------------------

render = scene.render
render.engine = "CYCLES"
render.resolution_x = 1920
render.resolution_y = 1080
render.image_settings.file_format = "FFMPEG"
render.ffmpeg.format = "MPEG4"
render.ffmpeg.codec = "H264"
render.ffmpeg.constant_rate_factor = "HIGH"

scene.cycles.samples = 64
scene.cycles.use_denoising = True

render.filepath = OUTPUT_PATH
bpy.ops.render.render(animation=True)

print(f"RESULT:{json.dumps({'output_path': OUTPUT_PATH, 'frames': TOTAL_FRAMES, 'fps': FPS})}")
