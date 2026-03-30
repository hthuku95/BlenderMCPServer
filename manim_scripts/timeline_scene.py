"""
timeline_scene.py — Manim animated horizontal timeline / project roadmap.

Args (via CHART_SCENE_ARGS env var, JSON):
    events:   list of {"date": str, "label": str, "color": str (optional)}
    title:    str
    duration: float
    style:    "dark" | "light" | "gradient"
    orientation: "horizontal" | "vertical"
"""
import json
import os
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_DEFAULT_EVENTS = [
    {"date": "Jan",  "label": "Project Kickoff",     "color": "BLUE"},
    {"date": "Mar",  "label": "MVP Launch",           "color": "GREEN"},
    {"date": "Jun",  "label": "Beta Release",         "color": "YELLOW"},
    {"date": "Sep",  "label": "Marketing Campaign",   "color": "ORANGE"},
    {"date": "Dec",  "label": "Full Launch",          "color": "RED"},
]

_CMAP = {
    "BLUE": BLUE_B, "RED": RED_B, "GREEN": GREEN_B, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE_B, "TEAL": TEAL_B, "GOLD": GOLD,
    "WHITE": WHITE, "CYAN": PURE_GREEN,
}


class TimelineScene(Scene):
    def construct(self):
        events      = _ARGS.get("events", _DEFAULT_EVENTS)
        title_text  = _ARGS.get("title", "Project Timeline")
        duration    = float(_ARGS.get("duration", 12.0))
        style       = _ARGS.get("style", "dark")
        orientation = _ARGS.get("orientation", "horizontal")

        if style == "light":
            self.camera.background_color = WHITE
            line_col, txt_col = DARK_GRAY, BLACK
        else:
            self.camera.background_color = "#0f0f1a"
            line_col, txt_col = GRAY_B, WHITE

        title = Text(title_text, font_size=30, color=txt_col, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.7)

        n = len(events)
        if orientation == "vertical":
            self._vertical_timeline(events, txt_col, line_col, duration)
        else:
            self._horizontal_timeline(events, txt_col, line_col, duration)

    def _horizontal_timeline(self, events, txt_col, line_col, duration):
        n = len(events)
        # Main line
        line_start = LEFT * 5.5
        line_end   = RIGHT * 5.5
        main_line  = Line(line_start, line_end, color=line_col, stroke_width=3)
        self.play(Create(main_line), run_time=0.8)

        positions = [
            line_start + (line_end - line_start) * (i / max(n - 1, 1))
            for i in range(n)
        ]

        anim_time = max(0.3, (duration - 3.0) / n)

        for i, (ev, pos) in enumerate(zip(events, positions)):
            col = _CMAP.get(ev.get("color", "BLUE"), BLUE_B)

            dot = Dot(pos, radius=0.12, color=col)
            dot.set_fill(col, opacity=1)

            # Alternate above/below
            if i % 2 == 0:
                tick_dir = UP
                date_dir = UP * 1.6
                label_dir = UP * 0.7
            else:
                tick_dir = DOWN
                date_dir = DOWN * 1.6
                label_dir = DOWN * 0.7

            tick  = Line(pos, pos + tick_dir * 0.4, color=col, stroke_width=2)
            date  = Text(ev.get("date", ""), font_size=18, color=col)
            date.next_to(pos + tick_dir * 0.4, tick_dir, buff=0.1)
            label = Text(ev.get("label", ""), font_size=16, color=txt_col)
            label.next_to(pos + tick_dir * 0.4 + tick_dir * 0.3, tick_dir, buff=0.0)
            if label.width > 2.2:
                label.scale(2.2 / label.width)

            self.play(
                GrowFromCenter(dot),
                Create(tick),
                Write(date),
                run_time=anim_time * 0.6,
            )
            self.play(FadeIn(label, shift=tick_dir * 0.2), run_time=anim_time * 0.4)

        self.wait(max(0.5, duration - self.renderer.time))

    def _vertical_timeline(self, events, txt_col, line_col, duration):
        n = len(events)
        top    = UP * 3.2
        bottom = DOWN * 3.2
        main_line = Line(top, bottom, color=line_col, stroke_width=3)
        self.play(Create(main_line), run_time=0.8)

        span = top - bottom
        anim_time = max(0.3, (duration - 3.0) / n)

        for i, ev in enumerate(events):
            pos = top + span * (i / max(n - 1, 1))
            col = _CMAP.get(ev.get("color", "BLUE"), BLUE_B)
            dot = Dot(pos, radius=0.12, color=col).set_fill(col, opacity=1)

            side = RIGHT if i % 2 == 0 else LEFT
            tick = Line(pos, pos + side * 0.4, color=col, stroke_width=2)
            date_mob = Text(ev.get("date", ""), font_size=16, color=col)
            date_mob.next_to(pos + side * 0.4, side, buff=0.1)
            label_mob = Text(ev.get("label", ""), font_size=14, color=txt_col)
            label_mob.next_to(date_mob, side, buff=0.05)
            if label_mob.width > 3.0:
                label_mob.scale(3.0 / label_mob.width)

            self.play(GrowFromCenter(dot), Create(tick), Write(date_mob), run_time=anim_time * 0.6)
            self.play(FadeIn(label_mob, shift=side * 0.15), run_time=anim_time * 0.4)

        self.wait(max(0.5, duration - self.renderer.time))
