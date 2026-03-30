"""
threed_math_scene.py — Manim ThreeDScene with 3D axes, surfaces, curves, and vectors.

Args (via CHART_SCENE_ARGS env var, JSON):
    scene_type:  "axes"       - 3D axes with optional function surface
                 "curve"      - Parametric 3D curve animation
                 "vector_field" - 2D vector field on 3D plane
                 "torus"      - Spinning torus (default demo)
    title:       str
    duration:    float
    x_range:     [min, max, step]
    y_range:     [min, max, step]
    z_range:     [min, max, step]
    function:    "sin"  | "cos"  | "saddle" | "paraboloid" | "wave"
    color:       Manim color name for surface
"""
import json
import math
import os
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_FUNCTIONS = {
    "sin":        lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
    "cos":        lambda u, v: np.array([u, v, np.cos(np.sqrt(u**2 + v**2 + 0.001))]),
    "saddle":     lambda u, v: np.array([u, v, u**2 - v**2]),
    "paraboloid": lambda u, v: np.array([u, v, u**2 + v**2]),
    "wave":       lambda u, v: np.array([u, v, 0.4 * np.sin(3 * u) * np.cos(3 * v)]),
    "ripple":     lambda u, v: np.array([u, v, np.sin(np.sqrt(u**2 + v**2 + 0.001)) / (np.sqrt(u**2 + v**2) + 0.5)]),
}

_COLOR_MAP = {
    "BLUE": BLUE, "RED": RED, "GREEN": GREEN, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE, "TEAL": TEAL, "GOLD": GOLD,
    "WHITE": WHITE,
}


class ThreeDMathScene(ThreeDScene):
    def construct(self):
        scene_type = _ARGS.get("scene_type", "surface")
        title_text = _ARGS.get("title", "3D Mathematics")
        duration   = float(_ARGS.get("duration", 12.0))
        func_name  = _ARGS.get("function", "wave")
        col_name   = _ARGS.get("color", "BLUE")
        color      = _COLOR_MAP.get(col_name.upper(), BLUE)

        self.camera.background_color = "#0d1117"

        if scene_type == "curve":
            self._parametric_curve(title_text, duration, color)
        elif scene_type == "vector_field":
            self._vector_field_scene(title_text, duration)
        elif scene_type == "torus":
            self._torus_scene(title_text, duration, color)
        else:
            self._surface_scene(title_text, duration, func_name, color)

    def _surface_scene(self, title_text, duration, func_name, color):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-1.5, 1.5, 0.5],
            x_length=6, y_length=6, z_length=3,
        )
        labels = axes.get_axis_labels(
            Text("x", font_size=24), Text("y", font_size=24), Text("z", font_size=24)
        )

        fn = _FUNCTIONS.get(func_name, _FUNCTIONS["wave"])
        surface = Surface(
            lambda u, v: axes.c2p(*fn(u, v)),
            u_range=[-2.5, 2.5], v_range=[-2.5, 2.5],
            resolution=(30, 30),
            fill_opacity=0.85,
            checkerboard_colors=[color, color_gradient([color, BLACK], 2)[1]],
            stroke_color=color,
            stroke_width=0.5,
        )

        title = Text(title_text, font_size=24, color=WHITE).to_corner(UL)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.85)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.6)
        self.play(Create(axes), Write(labels), run_time=1.5)
        self.play(Create(surface), run_time=2.5)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(max(1.0, duration - 5.5))
        self.stop_ambient_camera_rotation()

    def _parametric_curve(self, title_text, duration, color):
        axes = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-2, 6, 1],
        )

        # Helix
        curve = ParametricFunction(
            lambda t: axes.c2p(
                2 * np.cos(t),
                2 * np.sin(t),
                t / 2,
            ),
            t_range=[0, 4 * PI],
            color=color,
            stroke_width=4,
        )

        dot = Sphere(radius=0.12, color=YELLOW).move_to(axes.c2p(2, 0, 0))

        title = Text(title_text, font_size=24, color=WHITE).to_corner(UL)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, zoom=0.8)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.5)
        self.play(Create(axes), run_time=1.0)
        self.play(Create(curve), run_time=3.0)
        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(max(1.0, duration - 5.0))
        self.stop_ambient_camera_rotation()

    def _vector_field_scene(self, title_text, duration):
        plane = NumberPlane(x_range=[-4, 4, 1], y_range=[-3, 3, 1])
        func  = lambda pos: np.array([-pos[1], pos[0], 0]) * 0.4  # rotation field
        field = ArrowVectorField(func, x_range=[-3.5, 3.5, 0.7], y_range=[-2.5, 2.5, 0.7],
                                  color=BLUE)
        stream = StreamLines(func, stroke_width=2, max_anchors_per_line=30,
                              virtual_time=3, color=GREEN)
        title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)

        self.play(Write(title), Create(plane), run_time=1.0)
        self.play(Create(field), run_time=1.5)
        self.play(stream.create(), run_time=3.0)
        self.wait(max(0.5, duration - 6.0))

    def _torus_scene(self, title_text, duration, color):
        axes = ThreeDAxes(x_range=[-4,4,1], y_range=[-4,4,1], z_range=[-2,2,1])
        torus = Surface(
            lambda u, v: np.array([
                (2 + np.cos(v)) * np.cos(u),
                (2 + np.cos(v)) * np.sin(u),
                np.sin(v),
            ]),
            u_range=[0, TAU], v_range=[0, TAU],
            resolution=(40, 20),
            fill_opacity=0.9,
            checkerboard_colors=[color, DARK_GRAY],
            stroke_width=0.3,
        )
        title = Text(title_text, font_size=24, color=WHITE).to_corner(UL)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.5)
        self.play(Create(axes), Create(torus), run_time=2.5)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(max(1.0, duration - 4.0))
        self.stop_ambient_camera_rotation()
