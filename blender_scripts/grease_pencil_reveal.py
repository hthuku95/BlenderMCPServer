"""
grease_pencil_reveal.py — Grease Pencil draw-on / whiteboard-style animation.

Uses the GP_BUILD modifier to animate strokes appearing sequentially, achieving
a hand-drawn / whiteboard / sketch reveal effect.

Args (JSON after '--'):
    text:        str  — text or word to draw on screen (drawn letter by letter)
    style:       "whiteboard" | "neon" | "sketch" | "chalkboard"
    color:       [R,G,B,A]
    bg_color:    [R,G,B,A]
    stroke_width: int (default: 50, range 10–200)
    duration:    float (seconds, default: 6)
    output_path: str
"""
import sys, json, math
import bpy  # type: ignore

args_json = "{}"
if "--" in sys.argv:
    idx = sys.argv.index("--")
    if idx + 1 < len(sys.argv):
        args_json = sys.argv[idx + 1]

args         = json.loads(args_json)
text         = args.get("text",         "HELLO")
style        = args.get("style",        "whiteboard")
color        = args.get("color",        [0.1, 0.1, 0.7, 1.0])
bg_color     = args.get("bg_color",     [1.0, 1.0, 1.0, 1.0])
stroke_width = int(args.get("stroke_width", 50))
duration     = float(args.get("duration", 6.0))
output_path  = args.get("output_path",  "/tmp/grease_pencil_reveal.mp4")

# Style presets
style_presets = {
    "whiteboard":  {"bg": [0.98, 0.98, 0.98, 1.0], "color": [0.05, 0.05, 0.55, 1.0], "width": 45},
    "chalkboard":  {"bg": [0.06, 0.18, 0.10, 1.0], "color": [0.92, 0.92, 0.92, 1.0], "width": 60},
    "neon":        {"bg": [0.02, 0.02, 0.06, 1.0], "color": [0.0,  0.9,  1.0,  1.0], "width": 35},
    "sketch":      {"bg": [0.95, 0.93, 0.85, 1.0], "color": [0.15, 0.10, 0.08, 1.0], "width": 40},
}
preset = style_presets.get(style, style_presets["whiteboard"])
if not args.get("color"):     color    = preset["color"]
if not args.get("bg_color"):  bg_color = preset["bg"]
if not args.get("stroke_width"): stroke_width = preset["width"]

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
scene.display.shading.light      = "FLAT"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = False
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

# ── Grease Pencil object ───────────────────────────────────────────────────────
bpy.ops.object.gpencil_add(location=(0, 0, 0), type='EMPTY')
gp_obj = bpy.context.active_object
gp_obj.name = "GPReveal"
gpd = gp_obj.data

# GP Material
gp_mat = bpy.data.materials.new("GPMat")
bpy.data.materials.create_gpencil_data(gp_mat)
gp_mat.grease_pencil.color       = tuple(color[:3]) + (1.0,)
gp_mat.grease_pencil.show_stroke = True
gp_mat.grease_pencil.show_fill   = False
gpd.materials.append(gp_mat)

# ── Draw letters as GP strokes on frame 1 ─────────────────────────────────────
layer = gpd.layers.new("Text")
frame = layer.frames.new(1)

text_upper = text.upper()[:12]  # cap length
char_w     = 1.0   # width per character
start_x    = -(len(text_upper) - 1) * char_w / 2.0

# Simple stroke representation for each letter
# Each letter is defined as a list of (x, y) point sequences (sub-strokes)
LETTER_STROKES = {
    'A': [[(0,0),(0.5,1.5),(1,0)], [(0.2,0.75),(0.8,0.75)]],
    'B': [[(0,0),(0,1.5)], [(0,1.5),(0.5,1.5),(0.7,1.25),(0.7,1.0),(0.5,0.85),(0,0.85)],
          [(0,0.85),(0.6,0.85),(0.8,0.65),(0.8,0.3),(0.6,0),(0,0)]],
    'C': [[(0.9,1.3),(0.5,1.5),(0.1,1.2),(0,0.75),(0.1,0.3),(0.5,0),(0.9,0.2)]],
    'D': [[(0,0),(0,1.5)], [(0,1.5),(0.5,1.5),(0.9,1.1),(0.9,0.4),(0.5,0),(0,0)]],
    'E': [[(0.8,0),(0,0),(0,1.5),(0.8,1.5)], [(0,0.75),(0.6,0.75)]],
    'F': [[(0,0),(0,1.5),(0.8,1.5)], [(0,0.75),(0.6,0.75)]],
    'G': [[(0.9,1.2),(0.5,1.5),(0.1,1.2),(0,0.75),(0.1,0.3),(0.5,0),(0.9,0.2),(0.9,0.75),(0.5,0.75)]],
    'H': [[(0,0),(0,1.5)], [(1,0),(1,1.5)], [(0,0.75),(1,0.75)]],
    'I': [[(0.5,0),(0.5,1.5)], [(0.2,0),(0.8,0)], [(0.2,1.5),(0.8,1.5)]],
    'J': [[(0.8,1.5),(0.8,0.3),(0.5,0),(0.2,0.2)]],
    'K': [[(0,0),(0,1.5)], [(0,0.75),(0.9,1.5)], [(0,0.75),(0.9,0)]],
    'L': [[(0,1.5),(0,0),(0.8,0)]],
    'M': [[(0,0),(0,1.5),(0.5,0.7),(1,1.5),(1,0)]],
    'N': [[(0,0),(0,1.5),(1,0),(1,1.5)]],
    'O': [[(0.5,0),(0.1,0.2),(0,0.75),(0.1,1.3),(0.5,1.5),(0.9,1.3),(1,0.75),(0.9,0.2),(0.5,0)]],
    'P': [[(0,0),(0,1.5)], [(0,1.5),(0.6,1.5),(0.9,1.2),(0.9,0.9),(0.6,0.7),(0,0.7)]],
    'Q': [[(0.5,0),(0.1,0.2),(0,0.75),(0.1,1.3),(0.5,1.5),(0.9,1.3),(1,0.75),(0.9,0.2),(0.5,0)],
          [(0.6,0.3),(1.0,0.0)]],
    'R': [[(0,0),(0,1.5)], [(0,1.5),(0.6,1.5),(0.9,1.2),(0.9,0.9),(0.6,0.7),(0,0.7)],
          [(0.4,0.7),(0.9,0)]],
    'S': [[(0.8,1.3),(0.5,1.5),(0.1,1.3),(0.1,0.9),(0.9,0.6),(0.9,0.2),(0.5,0),(0.1,0.2)]],
    'T': [[(0,1.5),(1,1.5)], [(0.5,1.5),(0.5,0)]],
    'U': [[(0,1.5),(0,0.3),(0.5,0),(1,0.3),(1,1.5)]],
    'V': [[(0,1.5),(0.5,0),(1,1.5)]],
    'W': [[(0,1.5),(0.25,0),(0.5,0.8),(0.75,0),(1,1.5)]],
    'X': [[(0,1.5),(1,0)], [(0,0),(1,1.5)]],
    'Y': [[(0,1.5),(0.5,0.75),(1,1.5)], [(0.5,0.75),(0.5,0)]],
    'Z': [[(0,1.5),(1,1.5),(0,0),(1,0)]],
    ' ': [],
    '!': [[(0.5,0.3),(0.5,1.5)], [(0.5,0),(0.5,0.15)]],
    '?': [[(0.1,1.2),(0.5,1.5),(0.9,1.2),(0.9,0.9),(0.5,0.6),(0.5,0.35)],
          [(0.5,0),(0.5,0.15)]],
    '.': [[(0.5,0),(0.5,0.15)]],
}

def add_stroke_to_frame(gp_frame, points_2d, offset_x, scale=0.9, gp_z=-0.001):
    if not points_2d:
        return
    stroke = gp_frame.strokes.new()
    stroke.display_mode = '3DSPACE'
    stroke.line_width   = stroke_width
    stroke.material_index = 0
    stroke.points.add(count=len(points_2d))
    for i, (px, py) in enumerate(points_2d):
        stroke.points[i].co = (offset_x + px * scale, gp_z, (py - 0.75) * scale)
        stroke.points[i].pressure = 1.0
        stroke.points[i].strength = 1.0

stroke_count = 0
for ci, ch in enumerate(text_upper):
    x_off = start_x + ci * char_w - char_w * 0.5
    sub_strokes = LETTER_STROKES.get(ch, [[(0,0),(1,1.5)]])
    for ss in sub_strokes:
        add_stroke_to_frame(frame, ss, x_off, scale=0.7)
        stroke_count += 1

# ── BUILD modifier — strokes appear over time ──────────────────────────────────
build_mod = gp_obj.modifiers.new("Build", type='GP_BUILD')
build_mod.mode        = 'SEQUENTIAL'
build_mod.transition  = 'GROW'
build_mod.start_frame = 1
build_mod.length      = max(1, int(total_frames * 0.75))  # reveal over 75% of duration

# ── Optional cursor line ───────────────────────────────────────────────────────
if style in ("whiteboard", "chalkboard"):
    # Add a thin vertical bar (cursor) that slides right as text appears
    bpy.ops.mesh.primitive_plane_add(size=0.05, location=(0, 0, 0))
    cursor = bpy.context.active_object
    cursor.name = "Cursor"
    cursor.scale.z = 20  # tall thin bar
    cmat = bpy.data.materials.new("CursorMat")
    cmat.diffuse_color = tuple(color[:3]) + (1.0,)
    cursor.data.materials.append(cmat)
    start_x_cur = start_x - char_w * 0.5
    end_x_cur   = start_x + len(text_upper) * char_w - char_w * 0.5

    for f in range(1, total_frames + 1):
        t = (f - 1) / max(1, int(total_frames * 0.75))
        t = min(1.0, t)
        cursor.location.x = start_x_cur + t * (end_x_cur - start_x_cur)
        cursor.keyframe_insert(data_path="location", frame=f)

# ── Camera (orthographic for 2D effect) ─────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -10, 0))
cam = bpy.context.active_object
cam.rotation_euler   = (math.radians(90), 0, 0)
cam.data.type        = 'ORTHO'
cam.data.ortho_scale = 8.0
scene.camera         = cam

# ── Lighting ──────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(0, -5, 5))
sun = bpy.context.active_object.data
sun.energy = 3.0

# ── Render ────────────────────────────────────────────────────────────────────
bpy.ops.render.render(animation=True)

result = {
    "output_path": output_path,
    "duration": duration,
    "resolution": "1280x720",
    "frames": total_frames,
    "text": text,
    "style": style,
    "strokes": stroke_count,
}
print(f"RESULT:{json.dumps(result)}")
