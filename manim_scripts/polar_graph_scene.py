"""
polar_graph_scene.py — PolarPlane, ComplexPlane, and polar function graphs.

Args (via CHART_SCENE_ARGS env var, JSON):
    plane_type:   "polar"   — polar coordinate system with r=f(θ) graph
                  "complex" — complex plane with operations
                  "number_plane" — standard 2D plane with function plot
    title:        str
    function:     "rose"      — r = cos(k*θ)
                  "lemniscate" — r² = cos(2θ)
                  "spiral"    — r = θ/2π
                  "cardioid"  — r = 1 + cos(θ)
                  "circle"    — r = 1
                  "sin"       — y = sin(x) on number plane
                  "parabola"  — y = x² on number plane
    k_value:      int (for rose function, number of petals)
    duration:     float
    color:        Manim colour name
    show_label:   bool
"""
import json
import os
import numpy as np
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_CMAP = {
    "BLUE": BLUE_B, "RED": RED_B, "GREEN": GREEN_B, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE_B, "TEAL": TEAL_B, "GOLD": GOLD,
}


class PolarGraphScene(Scene):
    def construct(self):
        plane_type = _ARGS.get("plane_type", "polar")
        title_text = _ARGS.get("title", "Polar Graph")
        function   = _ARGS.get("function", "rose")
        k_val      = int(_ARGS.get("k_value", 4))
        duration   = float(_ARGS.get("duration", 12.0))
        col_name   = _ARGS.get("color", "BLUE")
        show_label = bool(_ARGS.get("show_label", True))

        color = _CMAP.get(col_name.upper(), BLUE_B)
        self.camera.background_color = "#0d1117"

        title = Text(title_text, font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title), run_time=0.5)

        if plane_type == "complex":
            self._complex_plane(duration, color, function)
        elif plane_type == "number_plane":
            self._number_plane(function, duration, color)
        else:
            self._polar_plane(function, k_val, duration, color, show_label)

    def _polar_plane(self, function, k, duration, color, show_label):
        plane = PolarPlane(
            radius_max=3.5,
            size=6,
            radius_config={"stroke_color": BLUE_E, "stroke_width": 1},
            azimuth_config={"stroke_color": BLUE_E, "stroke_width": 1},
        ).center()

        self.play(Create(plane), run_time=1.2)

        # Define r(θ)
        if function == "rose":
            r_fn = lambda theta: np.cos(k * theta)
            label = f"r = cos({k}θ)"
        elif function == "lemniscate":
            r_fn = lambda theta: np.sqrt(abs(np.cos(2 * theta)))
            label = "r² = cos(2θ)"
        elif function == "spiral":
            r_fn = lambda theta: theta / (2 * np.pi) * 3.5
            label = "r = θ / 2π"
        elif function == "cardioid":
            r_fn = lambda theta: 1.5 * (1 + np.cos(theta))
            label = "r = 1 + cos(θ)"
        else:  # circle
            r_fn = lambda theta: 2.0
            label = "r = 2"

        def polar_to_cartesian(theta):
            r = r_fn(theta)
            if r < 0:
                return None
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return plane.polar_to_point(r, theta)

        curve = ParametricFunction(
            lambda t: plane.polar_to_point(abs(r_fn(t)), t),
            t_range=[0, 2 * TAU, 0.02],
            color=color,
            stroke_width=3,
        )
        self.play(Create(curve), run_time=min(3.0, duration * 0.4))

        if show_label:
            lbl = MathTex(label, color=color, font_size=28)
            lbl.to_corner(DR, buff=0.4)
            self.play(Write(lbl), run_time=0.5)

        # Animate a dot tracing the curve
        dot = Dot(color=YELLOW, radius=0.1)
        dot.move_to(curve.get_start())
        self.add(dot)
        self.play(MoveAlongPath(dot, curve), run_time=min(3.0, duration * 0.35))
        self.wait(max(0.5, duration - self.renderer.time))

    def _complex_plane(self, duration, color, function):
        plane = ComplexPlane(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
        ).add_coordinates()

        self.play(Create(plane), run_time=1.0)

        # Show multiplication by complex number (rotation + scaling)
        vectors = [
            plane.number_to_point(1 + 0j),
            plane.number_to_point(1 + 1j),
            plane.number_to_point(-1 + 1j),
            plane.number_to_point(2 + 0j),
        ]
        dots  = VGroup(*[Dot(v, color=color, radius=0.1) for v in vectors])
        lines = VGroup(*[Line(ORIGIN, v, color=color, stroke_width=2) for v in vectors])

        self.play(Create(dots), Create(lines), run_time=1.0)

        # Rotate by multiplying by e^(iπ/4)
        theta = PI / 4
        self.play(
            Rotate(VGroup(dots, lines), angle=theta, about_point=ORIGIN),
            run_time=2.0,
        )

        label = MathTex(r"z \cdot e^{i\pi/4}", color=YELLOW, font_size=28)
        label.to_corner(UR, buff=0.4)
        self.play(Write(label), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _number_plane(self, function, duration, color):
        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-3.5, 3.5, 1],
            background_line_style={"stroke_color": BLUE_D, "stroke_opacity": 0.4},
        ).add_coordinates()

        self.play(Create(plane), run_time=1.0)

        if function == "parabola":
            graph = plane.plot(lambda x: x**2 / 3, x_range=[-3, 3], color=color, stroke_width=3)
            label = MathTex("y = x^2", color=color)
        elif function == "sin":
            graph = plane.plot(lambda x: 2 * np.sin(x), x_range=[-4.5, 4.5], color=color, stroke_width=3)
            label = MathTex("y = 2\\sin(x)", color=color)
        else:
            graph = plane.plot(lambda x: x**2 / 4, x_range=[-3.5, 3.5], color=color, stroke_width=3)
            label = MathTex("y = f(x)", color=color)

        label.to_corner(UR, buff=0.4)
        self.play(Create(graph), run_time=2.5)
        self.play(Write(label), run_time=0.5)

        # Trace a dot along the curve
        dot = Dot(color=YELLOW, radius=0.1)
        dot.move_to(graph.get_start())
        self.add(dot)
        self.play(MoveAlongPath(dot, graph), run_time=min(2.5, duration * 0.3))
        self.wait(max(0.5, duration - self.renderer.time))
