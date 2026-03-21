"""
Device Mockup — renders a screenshot/image inside a 3D device frame.

Supported devices:
  "iphone"   — iPhone 15 Pro (393×852 screen, titanium frame)
  "macbook"  — MacBook Pro 14" (silver frame, open lid)
  "browser"  — Chrome browser window (flat/3D perspective)
  "ipad"     — iPad Pro 12.9" (space grey frame)

Animation modes:
  "static"   — single frame PNG (1920×1080)
  "reveal"   — screen fades in over first 1s
  "scroll"   — screenshot scrolls vertically inside screen
  "tilt"     — device tilts from 30° to 0° (product reveal)

Args (JSON after --):
    output_path: str
    device: str                    — "iphone" | "macbook" | "browser" | "ipad"
    screenshot_path: str           — local path to the screenshot image
    animation: str                 — "static" | "reveal" | "scroll" | "tilt"
    duration: float                — clip length in seconds (ignored for static)
    fps: int                       — frame rate (default 60)
    background_color: list[float]  — RGB 0-1 (default [0.05, 0.05, 0.08])
    accent_color: list[float]      — RGB 0-1 for glow/shadow (default [0.3, 0.5, 1.0])
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

OUTPUT_PATH   = args.get("output_path", "/tmp/device_mockup.mp4")
DEVICE        = args.get("device", "iphone")
SCREENSHOT    = args.get("screenshot_path", "")
ANIMATION     = args.get("animation", "reveal")
DURATION      = float(args.get("duration", 6.0))
FPS           = int(args.get("fps", 60))
BG_COLOR      = args.get("background_color", [0.05, 0.05, 0.08])
ACCENT_COLOR  = args.get("accent_color", [0.3, 0.5, 1.0])
TOTAL_FRAMES  = int(DURATION * FPS)

IS_STATIC = ANIMATION == "static"

# ---------------------------------------------------------------------------
# Scene reset
# ---------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = 1 if IS_STATIC else TOTAL_FRAMES

# Output format
render = scene.render
render.resolution_x = 1920
render.resolution_y = 1080
render.engine = "CYCLES"
scene.cycles.samples = 128
scene.cycles.use_denoising = True

if IS_STATIC:
    render.image_settings.file_format = "PNG"
    render.filepath = OUTPUT_PATH.replace(".mp4", ".png")
else:
    render.image_settings.file_format = "FFMPEG"
    render.ffmpeg.format = "MPEG4"
    render.ffmpeg.codec = "H264"
    render.ffmpeg.constant_rate_factor = "HIGH"
    render.filepath = OUTPUT_PATH

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (*BG_COLOR, 1.0)
bg.inputs["Strength"].default_value = 0.5

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_data.lens = 85  # portrait-friendly focal length


def _set_cam(loc, rot_deg):
    cam_obj.location = loc
    cam_obj.rotation_euler = [math.radians(a) for a in rot_deg]


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
def add_light(name, ltype, energy, loc, rot_deg=None):
    ld = bpy.data.lights.new(name, type=ltype)
    ld.energy = energy
    lo = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo)
    lo.location = loc
    if rot_deg:
        lo.rotation_euler = [math.radians(a) for a in rot_deg]
    return lo

add_light("Key",  "AREA", 500, (4, -6, 8))
add_light("Fill", "AREA", 150, (-4, -4, 4))
# Accent rim light tinted with accent_color
rim = add_light("Rim", "AREA", 300, (0, 5, 5))
rim.data.color = ACCENT_COLOR[:3]

# ---------------------------------------------------------------------------
# Screenshot texture helper
# ---------------------------------------------------------------------------
def _screen_material(name: str, img_path: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    tex  = nodes.new("ShaderNodeTexImage")
    mix  = nodes.new("ShaderNodeMixShader")
    transp = nodes.new("ShaderNodeBsdfTransparent")

    if img_path and os.path.exists(img_path):
        try:
            tex.image = bpy.data.images.load(img_path)
        except Exception:
            pass

    links.new(tex.outputs["Color"],     emit.inputs["Color"])
    links.new(transp.outputs["BSDF"],   mix.inputs[1])
    links.new(emit.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"],    out.inputs["Surface"])

    emit.inputs["Strength"].default_value = 2.0
    mix.inputs["Fac"].default_value = 1.0  # fully visible by default
    return mat, mix


def _frame_material(name: str, color: tuple) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.9
        bsdf.inputs["Roughness"].default_value = 0.1
    return mat


# ---------------------------------------------------------------------------
# Device geometry builders
# ---------------------------------------------------------------------------

def build_iphone():
    """iPhone 15 Pro proportions: 2.16:1 height ratio, rounded corners."""
    W, H, D = 0.8, 1.73, 0.08

    # Body
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    body = bpy.context.active_object
    body.name = "iPhone_Body"
    body.scale = (W / 2, D / 2, H / 2)
    bpy.ops.object.transform_apply(scale=True)

    frame_mat = _frame_material("FrameMat", (0.7, 0.65, 0.6))  # titanium
    body.data.materials.append(frame_mat)

    # Screen (slightly inset)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -(D / 2 + 0.001), 0))
    screen = bpy.context.active_object
    screen.name = "iPhone_Screen"
    screen.scale = (W * 0.88 / 2, H * 0.88 / 2, 1)
    bpy.ops.object.transform_apply(scale=True)
    screen.rotation_euler = (math.radians(90), 0, 0)
    bpy.ops.object.transform_apply(rotation=True)

    screen_mat, mix_node = _screen_material("ScreenMat", SCREENSHOT)
    screen.data.materials.append(screen_mat)
    return body, screen, mix_node


def build_macbook():
    """MacBook Pro 14 — lid (screen) + base."""
    LID_W, LID_H, LID_D = 2.4, 1.55, 0.04
    BASE_W, BASE_H, BASE_D = 2.4, 0.06, 1.6

    # Lid
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, BASE_H / 2 + LID_H / 2))
    lid = bpy.context.active_object
    lid.name = "MacBook_Lid"
    lid.scale = (LID_W / 2, LID_D / 2, LID_H / 2)
    bpy.ops.object.transform_apply(scale=True)
    frame_mat = _frame_material("MacFrameMat", (0.76, 0.76, 0.76))  # silver
    lid.data.materials.append(frame_mat)

    # Base
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    base = bpy.context.active_object
    base.name = "MacBook_Base"
    base.scale = (BASE_W / 2, BASE_D / 2, BASE_H / 2)
    bpy.ops.object.transform_apply(scale=True)
    base.data.materials.append(frame_mat)

    # Screen on lid face
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -(LID_D / 2 + 0.001), BASE_H / 2 + LID_H / 2))
    screen = bpy.context.active_object
    screen.name = "MacBook_Screen"
    screen.scale = (LID_W * 0.88 / 2, LID_H * 0.88 / 2, 1)
    bpy.ops.object.transform_apply(scale=True)
    screen.rotation_euler = (math.radians(90), 0, 0)
    bpy.ops.object.transform_apply(rotation=True)
    screen_mat, mix_node = _screen_material("MacScreenMat", SCREENSHOT)
    screen.data.materials.append(screen_mat)
    return lid, screen, mix_node


def build_browser():
    """Flat browser window — chrome bar + content area."""
    W, H = 3.2, 2.0
    CHROME_H = 0.18

    # Chrome bar
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, H / 2 - CHROME_H / 2))
    chrome = bpy.context.active_object
    chrome.name = "Browser_Chrome"
    chrome.scale = (W / 2, 1, CHROME_H / 2)
    bpy.ops.object.transform_apply(scale=True)
    chrome_mat = bpy.data.materials.new("ChromeMat")
    chrome_mat.use_nodes = True
    bsdf = chrome_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.15, 1.0)
    chrome.data.materials.append(chrome_mat)

    # Content area
    content_h = H - CHROME_H
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, H / 2 - CHROME_H - content_h / 2))
    screen = bpy.context.active_object
    screen.name = "Browser_Content"
    screen.scale = (W / 2, 1, content_h / 2)
    bpy.ops.object.transform_apply(scale=True)
    screen_mat, mix_node = _screen_material("BrowserContentMat", SCREENSHOT)
    screen.data.materials.append(screen_mat)
    return chrome, screen, mix_node


def build_ipad():
    """iPad Pro 12.9 — landscape orientation."""
    W, H, D = 2.4, 1.8, 0.06
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    body = bpy.context.active_object
    body.name = "iPad_Body"
    body.scale = (W / 2, D / 2, H / 2)
    bpy.ops.object.transform_apply(scale=True)
    frame_mat = _frame_material("iPadFrameMat", (0.2, 0.2, 0.2))
    body.data.materials.append(frame_mat)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -(D / 2 + 0.001), 0))
    screen = bpy.context.active_object
    screen.name = "iPad_Screen"
    screen.scale = (W * 0.9 / 2, H * 0.9 / 2, 1)
    bpy.ops.object.transform_apply(scale=True)
    screen.rotation_euler = (math.radians(90), 0, 0)
    bpy.ops.object.transform_apply(rotation=True)
    screen_mat, mix_node = _screen_material("iPadScreenMat", SCREENSHOT)
    screen.data.materials.append(screen_mat)
    return body, screen, mix_node


# ---------------------------------------------------------------------------
# Build device
# ---------------------------------------------------------------------------
mix_node = None
if DEVICE == "iphone":
    primary, screen_obj, mix_node = build_iphone()
    _set_cam((0, -4, 0.3), (90, 0, 0))
elif DEVICE == "macbook":
    primary, screen_obj, mix_node = build_macbook()
    _set_cam((0, -5, 1.0), (80, 0, 0))
elif DEVICE == "browser":
    primary, screen_obj, mix_node = build_browser()
    _set_cam((0, -4, 0), (90, 0, 0))
elif DEVICE == "ipad":
    primary, screen_obj, mix_node = build_ipad()
    _set_cam((0, -5, 0), (90, 0, 0))
else:
    primary, screen_obj, mix_node = build_iphone()
    _set_cam((0, -4, 0.3), (90, 0, 0))

# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------
if not IS_STATIC and mix_node is not None:

    if ANIMATION == "reveal":
        # Screen fades in over first 1s
        mix_node.inputs["Fac"].default_value = 0.0
        mix_node.inputs["Fac"].keyframe_insert("default_value", frame=1)
        mix_node.inputs["Fac"].default_value = 1.0
        mix_node.inputs["Fac"].keyframe_insert("default_value", frame=FPS)

    elif ANIMATION == "scroll":
        # Scroll UV offset on the screen texture
        tex_nodes = [n for n in screen_obj.material_slots[0].material.node_tree.nodes if n.type == "TEX_IMAGE"]
        if tex_nodes:
            tex = tex_nodes[0]
            mapping = screen_obj.material_slots[0].material.node_tree.nodes.new("ShaderNodeMapping")
            coord = screen_obj.material_slots[0].material.node_tree.nodes.new("ShaderNodeTexCoord")
            screen_obj.material_slots[0].material.node_tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
            screen_obj.material_slots[0].material.node_tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            mapping.inputs["Location"].default_value = (0, 0, 0)
            mapping.inputs["Location"].keyframe_insert("default_value", frame=1)
            mapping.inputs["Location"].default_value = (0, -1.5, 0)
            mapping.inputs["Location"].keyframe_insert("default_value", frame=TOTAL_FRAMES)

    elif ANIMATION == "tilt":
        # Device tilts from 30° to 0°
        primary.rotation_euler = (0, math.radians(30), 0)
        primary.keyframe_insert("rotation_euler", frame=1)
        if screen_obj:
            screen_obj.rotation_euler = (math.radians(90), math.radians(30), 0)
            screen_obj.keyframe_insert("rotation_euler", frame=1)
        primary.rotation_euler = (0, 0, 0)
        primary.keyframe_insert("rotation_euler", frame=TOTAL_FRAMES // 2)
        if screen_obj:
            screen_obj.rotation_euler = (math.radians(90), 0, 0)
            screen_obj.keyframe_insert("rotation_euler", frame=TOTAL_FRAMES // 2)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
bpy.ops.render.render(animation=not IS_STATIC, write_still=IS_STATIC)

final_path = OUTPUT_PATH if not IS_STATIC else OUTPUT_PATH.replace(".mp4", ".png")
print(f"RESULT:{json.dumps({'output_path': final_path, 'device': DEVICE, 'animation': ANIMATION})}")
