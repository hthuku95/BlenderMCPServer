"""
Data visualisation Manim scene.

Supports: bar_chart | line_chart | pie_chart | counter | scatter

Args (passed via CHART_SCENE_ARGS env var as JSON):
    chart_type:   str   — "bar_chart"|"line_chart"|"pie_chart"|"counter"|"scatter"
    title:        str   — chart title
    data:         list  — data points (numbers for bar/line/scatter; label+value dicts for pie)
    labels:       list  — x-axis or segment labels
    duration:     float — clip length in seconds
    y_range:      list  — [min, max, step]   (bar / line only)
    colors:       list  — optional list of colour names (ManimCE named colours)

Output: MP4 (opaque dark background)
"""
from __future__ import annotations

import json
import os

from manim import *


def _load_args() -> dict:
    raw = os.environ.get("CHART_SCENE_ARGS", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ── helpers ────────────────────────────────────────────────────────────────────

_DEFAULT_COLORS = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED, PURPLE, PINK]


def _resolve_colors(color_names: list[str], count: int) -> list:
    """Map string colour names (e.g. 'BLUE') to Manim colour objects."""
    palette = []
    for name in color_names[:count]:
        obj = globals().get(name.upper())
        palette.append(obj if obj is not None else BLUE)
    # pad with defaults if not enough supplied
    while len(palette) < count:
        palette.append(_DEFAULT_COLORS[len(palette) % len(_DEFAULT_COLORS)])
    return palette


# ── scene ──────────────────────────────────────────────────────────────────────

class DataChartScene(Scene):
    def construct(self) -> None:
        args = _load_args()
        chart_type: str = args.get("chart_type", "bar_chart")
        title: str      = args.get("title", "Data Visualisation")
        data: list      = args.get("data", [3, 7, 5, 9, 4, 6])
        labels: list    = args.get("labels", [str(i + 1) for i in range(len(data))])
        duration: float = float(args.get("duration", 10.0))
        y_range: list   = args.get("y_range", [0, max(data) * 1.2 if data else 10, 2])
        color_names: list = args.get("colors", [])

        title_mob = Text(title, font_size=36, color=WHITE)
        title_mob.to_edge(UP, buff=0.3)
        self.play(Write(title_mob), run_time=0.8)
        self.wait(0.2)

        if chart_type == "bar_chart":
            self._bar_chart(data, labels, y_range, color_names, duration - 1.0)
        elif chart_type == "line_chart":
            self._line_chart(data, labels, y_range, color_names, duration - 1.0)
        elif chart_type == "pie_chart":
            self._pie_chart(data, labels, color_names, duration - 1.0)
        elif chart_type == "counter":
            target = float(data[0]) if data else 100.0
            self._counter(target, duration - 1.0)
        elif chart_type == "scatter":
            self._scatter(data, labels, y_range, duration - 1.0)
        else:
            self._bar_chart(data, labels, y_range, color_names, duration - 1.0)

    # ── bar chart ──────────────────────────────────────────────────────────────

    def _bar_chart(self, data, labels, y_range, color_names, duration):
        colors = _resolve_colors(color_names, len(data))
        chart = BarChart(
            values=data,
            bar_names=labels[:len(data)],
            y_range=y_range,
            y_length=5,
            x_length=min(10, len(data) * 1.4),
            bar_colors=colors,
        )
        chart.center().shift(DOWN * 0.3)
        self.play(Create(chart), run_time=min(duration * 0.5, 2.5))
        self.wait(0.3)
        bar_labels = chart.get_bar_labels(font_size=24)
        self.play(
            LaggedStart(*[FadeIn(l, shift=UP * 0.2) for l in bar_labels],
                        lag_ratio=0.12),
            run_time=min(duration * 0.3, 1.5),
        )
        self.wait(max(duration - 4.5, 1.0))

    # ── line chart ─────────────────────────────────────────────────────────────

    def _line_chart(self, data, labels, y_range, color_names, duration):
        axes = Axes(
            x_range=[0, len(data) + 1, 1],
            y_range=y_range,
            x_length=9,
            y_length=5,
            axis_config={"include_tip": True, "color": WHITE},
        )
        axes.center().shift(DOWN * 0.2)

        colors = _resolve_colors(color_names, 1)
        line_color = colors[0]

        x_vals = list(range(1, len(data) + 1))
        points = [axes.c2p(x, y) for x, y in zip(x_vals, data)]
        dot_group = VGroup(*[Dot(p, color=line_color, radius=0.08) for p in points])
        polyline = VMobject(color=line_color, stroke_width=3)
        polyline.set_points_as_corners(points)

        # x-axis tick labels
        x_labels = VGroup()
        for i, label in enumerate(labels[:len(data)]):
            lbl = Text(str(label), font_size=18, color=GRAY).next_to(
                axes.c2p(i + 1, y_range[0]), DOWN, buff=0.15
            )
            x_labels.add(lbl)

        self.play(Create(axes), run_time=0.8)
        self.play(Write(x_labels), run_time=0.6)
        self.play(Create(polyline), run_time=min(duration * 0.4, 2.0))
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dot_group], lag_ratio=0.1),
            run_time=min(duration * 0.25, 1.2),
        )
        self.wait(max(duration - 5.0, 1.0))

    # ── pie chart ──────────────────────────────────────────────────────────────

    def _pie_chart(self, data, labels, color_names, duration):
        total = sum(data) or 1
        colors = _resolve_colors(color_names, len(data))

        slices = VGroup()
        legend = VGroup()
        angle = 0.0
        radius = 2.2

        for i, (val, label) in enumerate(zip(data, labels)):
            sweep = (val / total) * TAU
            sector = AnnularSector(
                inner_radius=0,
                outer_radius=radius,
                angle=sweep,
                start_angle=angle,
                color=colors[i],
                fill_opacity=0.9,
                stroke_width=1,
                stroke_color=BLACK,
            )
            slices.add(sector)

            # legend entry
            swatch = Square(side_length=0.3, color=colors[i], fill_opacity=1).shift(LEFT * 4.5)
            pct = f"{100 * val / total:.1f}%"
            text = Text(f"{label}  {pct}", font_size=22, color=WHITE)
            row = VGroup(swatch, text).arrange(RIGHT, buff=0.2)
            row.shift(DOWN * (i * 0.5 - len(data) * 0.25))
            legend.add(row)

            angle += sweep

        slices.center().shift(RIGHT * 1.5)

        self.play(
            LaggedStart(*[GrowFromCenter(s) for s in slices], lag_ratio=0.08),
            run_time=min(duration * 0.5, 2.5),
        )
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in legend], lag_ratio=0.1),
            run_time=min(duration * 0.3, 1.5),
        )
        self.wait(max(duration - 5.0, 1.0))

    # ── animated counter ───────────────────────────────────────────────────────

    def _counter(self, target: float, duration: float):
        tracker = ValueTracker(0)
        is_int = float(target) == int(target)
        fmt = "{:.0f}" if is_int else "{:.2f}"

        number = always_redraw(
            lambda: Text(
                fmt.format(tracker.get_value()),
                font_size=120,
                color=YELLOW,
            ).move_to(ORIGIN)
        )
        self.add(number)
        self.play(
            tracker.animate.set_value(target),
            run_time=max(duration - 1.0, 2.0),
            rate_func=rush_into,
        )
        self.wait(1.0)

    # ── scatter plot ───────────────────────────────────────────────────────────

    def _scatter(self, data, labels, y_range, duration):
        """
        data: list of [x, y] pairs  OR  flat list of y values (x = index)
        """
        if data and isinstance(data[0], (list, tuple)):
            xy = [(float(p[0]), float(p[1])) for p in data]
        else:
            xy = [(i + 1, float(v)) for i, v in enumerate(data)]

        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        x_range = [min(xs) - 0.5, max(xs) + 0.5, max(1, (max(xs) - min(xs)) / 5)]

        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=9,
            y_length=5,
            axis_config={"include_tip": True, "color": WHITE},
        )
        axes.center().shift(DOWN * 0.2)

        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.1)
            for x, y in xy
        ])

        self.play(Create(axes), run_time=0.8)
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06),
            run_time=min(duration * 0.5, 2.5),
        )
        self.wait(max(duration - 4.0, 1.0))
