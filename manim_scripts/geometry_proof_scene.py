"""
geometry_proof_scene.py — Geometric proof animations with shapes, transformations, and labels.

Args (via CHART_SCENE_ARGS env var, JSON):
    proof_type:  "pythagorean"     — Pythagoras a²+b²=c² visual proof
                 "circle_area"     — π r² area proof (inscribed polygon limit)
                 "triangle_sum"    — sum of angles = 180°
                 "similar_triangles" — similar triangles ratio
                 "boolean_ops"     — Union/Difference/Intersection demo
                 "custom"          — show shapes defined in `shapes` param
    title:       str
    duration:    float
    color_a:     Manim colour for shape A
    color_b:     Manim colour for shape B
    show_labels: bool
"""
import json
import os
import math
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_CMAP = {
    "BLUE": BLUE_B, "RED": RED_B, "GREEN": GREEN_B, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE_B, "TEAL": TEAL_B, "GOLD": GOLD,
}


class GeometryProofScene(Scene):
    def construct(self):
        proof_type  = _ARGS.get("proof_type", "pythagorean")
        title_text  = _ARGS.get("title", "Geometry Proof")
        duration    = float(_ARGS.get("duration", 14.0))
        ca_name     = _ARGS.get("color_a", "BLUE")
        cb_name     = _ARGS.get("color_b", "RED")
        show_labels = bool(_ARGS.get("show_labels", True))

        ca = _CMAP.get(ca_name.upper(), BLUE_B)
        cb = _CMAP.get(cb_name.upper(), RED_B)
        self.camera.background_color = "#0d1117"

        title = Text(title_text, font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title), run_time=0.5)

        if proof_type == "pythagorean":
            self._pythagorean(ca, cb, duration, show_labels)
        elif proof_type == "circle_area":
            self._circle_area(ca, duration, show_labels)
        elif proof_type == "triangle_sum":
            self._triangle_sum(ca, cb, duration, show_labels)
        elif proof_type == "boolean_ops":
            self._boolean_ops(ca, cb, duration)
        else:
            self._pythagorean(ca, cb, duration, show_labels)

    def _pythagorean(self, ca, cb, duration, show_labels):
        a, b = 2.0, 1.5
        c = math.sqrt(a**2 + b**2)

        # Right triangle
        tri = Polygon(
            ORIGIN, RIGHT * a, UP * b,
            fill_color=ca, fill_opacity=0.5,
            stroke_color=WHITE, stroke_width=2,
        ).shift(LEFT * 2 + DOWN * 0.5)

        # Squares on each side
        sq_a = Square(side_length=a, fill_color=BLUE_D, fill_opacity=0.4,
                      stroke_color=BLUE_B, stroke_width=2)
        sq_b = Square(side_length=b, fill_color=RED_D, fill_opacity=0.4,
                      stroke_color=RED_B, stroke_width=2)
        sq_c = Square(side_length=c, fill_color=GREEN_D, fill_opacity=0.4,
                      stroke_color=GREEN_B, stroke_width=2)

        # Position squares on triangle sides
        # sq_a below the base
        sq_a.next_to(tri, DOWN, buff=0)
        sq_a.align_to(tri, LEFT)

        self.play(Create(tri), run_time=1.0)
        self.play(Create(sq_a), run_time=0.8)
        self.play(Create(sq_b), run_time=0.8)
        self.play(Create(sq_c), run_time=0.8)

        if show_labels:
            la = MathTex("a^2", color=BLUE_B, font_size=30).move_to(sq_a)
            lb = MathTex("b^2", color=RED_B, font_size=30).move_to(sq_b)
            lc = MathTex("c^2", color=GREEN_B, font_size=30).move_to(sq_c)
            self.play(Write(la), Write(lb), Write(lc), run_time=0.8)

        formula = MathTex("a^2 + b^2 = c^2", color=WHITE, font_size=40)
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula), run_time=1.0)
        self.wait(max(0.5, duration - self.renderer.time))

    def _circle_area(self, ca, duration, show_labels):
        # Inscribed polygon limit approach to π r²
        circle = Circle(radius=2.5, color=WHITE, stroke_width=2)
        self.play(Create(circle), run_time=0.8)

        prev_poly = None
        for n in [4, 6, 8, 12, 24]:
            poly = RegularPolygon(n=n, radius=2.5,
                                   fill_color=ca, fill_opacity=0.4,
                                   stroke_color=ca, stroke_width=2)
            if prev_poly:
                self.play(ReplacementTransform(prev_poly, poly), run_time=0.6)
            else:
                self.play(Create(poly), run_time=0.6)
            if show_labels:
                lbl = Text(f"n = {n}", font_size=22, color=YELLOW)
                lbl.to_corner(DR, buff=0.4)
                self.add(lbl)
            prev_poly = poly

        formula = MathTex(r"A = \pi r^2", color=WHITE, font_size=44)
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula), run_time=1.0)
        self.wait(max(0.5, duration - self.renderer.time))

    def _triangle_sum(self, ca, cb, duration, show_labels):
        # Triangle with arcs showing 180°
        verts = [LEFT * 2.5 + DOWN * 1.2, RIGHT * 2.5 + DOWN * 1.2, UP * 2.0]
        tri = Polygon(*verts, fill_color=ca, fill_opacity=0.3,
                      stroke_color=WHITE, stroke_width=2)
        self.play(Create(tri), run_time=1.0)

        # Angle arcs
        angles = []
        labels = []
        colors = [RED_B, GREEN_B, BLUE_B]
        for i in range(3):
            A = verts[i]
            B = verts[(i + 1) % 3]
            C = verts[(i + 2) % 3]
            arc = Angle(
                Line(A, B), Line(A, C),
                radius=0.4, color=colors[i], stroke_width=3,
            )
            angles.append(arc)
            if show_labels:
                lbl = MathTex(["\\alpha", "\\beta", "\\gamma"][i],
                               color=colors[i], font_size=26)
                lbl.next_to(arc, arc.get_center() - A, buff=0.1)
                labels.append(lbl)

        self.play(*[Create(a) for a in angles], run_time=1.0)
        if labels:
            self.play(*[Write(l) for l in labels], run_time=0.6)

        formula = MathTex(r"\alpha + \beta + \gamma = 180°", color=WHITE, font_size=38)
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula), run_time=1.0)
        self.wait(max(0.5, duration - self.renderer.time))

    def _boolean_ops(self, ca, cb, duration):
        labels = ["Union", "Difference", "Intersection"]
        results = []

        circle1 = Circle(radius=1.2, fill_color=ca, fill_opacity=0.6,
                          stroke_color=ca, stroke_width=2).shift(LEFT * 0.5)
        circle2 = Circle(radius=1.2, fill_color=cb, fill_opacity=0.6,
                          stroke_color=cb, stroke_width=2).shift(RIGHT * 0.5)

        # Show base shapes
        grp = VGroup(circle1, circle2).center()
        self.play(Create(grp), run_time=1.0)

        for i, (op_name, op_cls) in enumerate(
            [("Union", Union), ("Difference", Difference), ("Intersection", Intersection)]
        ):
            c1 = Circle(radius=1.2).shift(LEFT * 0.5)
            c2 = Circle(radius=1.2).shift(RIGHT * 0.5)
            try:
                result = op_cls(c1, c2, fill_opacity=0.7, stroke_width=2)
                result.set_fill(ca if op_name != "Intersection" else YELLOW)
                result.center().shift(DOWN * 0.2)
                lbl = Text(op_name, font_size=22, color=WHITE)
                lbl.next_to(result, DOWN, buff=0.2)
                self.play(ReplacementTransform(grp.copy(), result), Write(lbl), run_time=1.2)
                self.wait(0.8)
                self.play(FadeOut(VGroup(result, lbl)), run_time=0.4)
            except Exception:
                pass

        self.wait(max(0.5, duration - self.renderer.time))
