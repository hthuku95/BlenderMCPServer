"""
vector_field_scene.py — Vector fields, stream lines, and phase portraits.

Args (via CHART_SCENE_ARGS env var, JSON):
    field_type:  "rotation" | "radial" | "saddle" | "curl" | "gravity" | "custom"
    title:       str
    duration:    float
    show_streams: bool (default true) — also render StreamLines
    particle_trace: bool (default false) — trace moving dot along flow
    color:       Manim colour name for vectors/streams
    style:       "dark" | "grid"
"""
import json
import os
from manim import *
import numpy as np

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_FIELDS = {
    "rotation":  lambda p: np.array([-p[1], p[0], 0]) * 0.5,
    "radial":    lambda p: normalize(np.array([p[0], p[1], 0])) * 0.6,
    "sink":      lambda p: -normalize(np.array([p[0], p[1], 0]) + np.array([0.001, 0.001, 0])) * 0.6,
    "saddle":    lambda p: np.array([p[0], -p[1], 0]) * 0.4,
    "curl":      lambda p: np.array([-p[1] + p[0] * 0.1, p[0] + p[1] * 0.1, 0]) * 0.4,
    "gravity":   lambda p: np.array([0, -1, 0]) * min(1.0, 1.0 / (abs(p[1]) + 0.5)),
}

_CMAP = {
    "BLUE": BLUE_B, "RED": RED_B, "GREEN": GREEN_B, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE_B, "TEAL": TEAL_B, "GOLD": GOLD, "WHITE": WHITE,
}


class VectorFieldScene(Scene):
    def construct(self):
        field_type    = _ARGS.get("field_type", "rotation")
        title_text    = _ARGS.get("title", "Vector Field")
        duration      = float(_ARGS.get("duration", 12.0))
        show_streams  = bool(_ARGS.get("show_streams", True))
        particle_trace = bool(_ARGS.get("particle_trace", False))
        col_name      = _ARGS.get("color", "BLUE")
        style         = _ARGS.get("style", "dark")

        color = _CMAP.get(col_name.upper(), BLUE_B)

        if style == "grid":
            self.camera.background_color = "#0a0a1a"
            plane = NumberPlane(
                x_range=[-4, 4, 1], y_range=[-3.5, 3.5, 1],
                background_line_style={"stroke_color": BLUE_D, "stroke_width": 0.5, "stroke_opacity": 0.3},
            )
            self.play(Create(plane), run_time=0.8)
        else:
            self.camera.background_color = "#0d1117"

        fn = _FIELDS.get(field_type, _FIELDS["rotation"])

        title = Text(title_text, font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.15)
        self.play(Write(title), run_time=0.5)

        # Arrow vector field
        field = ArrowVectorField(
            fn,
            x_range=[-3.5, 3.5, 0.8],
            y_range=[-3.0, 3.0, 0.8],
            length_func=lambda norm: 0.45 * sigmoid(norm),
            color=color,
        )
        self.play(Create(field), run_time=1.5)

        # Stream lines
        if show_streams:
            streams = StreamLines(
                fn,
                stroke_width=2,
                max_anchors_per_line=40,
                virtual_time=3,
                x_range=[-3.5, 3.5],
                y_range=[-3.0, 3.0],
                color=color_gradient([color, WHITE], 3)[1],
            )
            self.play(streams.create(), run_time=3.0)

        # Optional particle tracing
        if particle_trace:
            dot = Dot(color=YELLOW, radius=0.12)
            dot.move_to(RIGHT * 2)
            self.add(dot)

            def update_dot(mob, dt):
                pos = mob.get_center()
                velocity = fn(pos)
                mob.shift(velocity * dt * 0.8)

            dot.add_updater(update_dot)
            self.wait(max(2.0, duration - self.renderer.time - 1.0))
            dot.remove_updater(update_dot)
        else:
            self.wait(max(0.5, duration - self.renderer.time))
