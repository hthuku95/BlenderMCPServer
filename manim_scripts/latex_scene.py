"""
latex_scene.py — Manim LaTeX / math equation animation.

Args are injected via the LATEX_SCENE_ARGS environment variable (JSON string).
Set by tools/manim_runner.py before invoking:
    python3 -m manim -qm --media_dir /tmp/work --output_file LatexScene latex_scene.py LatexScene

Supported animation_type values:
    "appear"        — equation writes itself in, holds, fades out
    "morph"         — equation appears, then transforms into a boxed version
    "step_by_step"  — each term highlights in sequence after the full equation is shown
"""

import json
import os

from manim import *  # type: ignore

_args: dict = json.loads(os.getenv("LATEX_SCENE_ARGS", "{}"))

LATEX_EXPRESSION: str = _args.get("latex_expression", r"E = mc^2")
ANIMATION_TYPE: str = _args.get("animation_type", "appear")
DURATION: float = float(_args.get("duration", 8.0))
BACKGROUND_STYLE: str = _args.get("background_style", "dark")


def _bg_color() -> ManimColor:
    if BACKGROUND_STYLE == "light":
        return WHITE
    if BACKGROUND_STYLE == "transparent":
        return BLACK  # Manim transparent mode handled separately
    return ManimColor("#0a0a1a")


def _eq_color() -> ManimColor:
    return BLACK if BACKGROUND_STYLE == "light" else WHITE


class LatexScene(Scene):
    def construct(self):
        self.camera.background_color = _bg_color()
        hold_time = max(1.0, DURATION - 4.0)
        eq_col = _eq_color()

        if ANIMATION_TYPE == "appear":
            eq = MathTex(LATEX_EXPRESSION, font_size=80, color=eq_col)
            eq.set_stroke(BLUE_C, width=1.5)
            self.play(Write(eq), run_time=2.0)
            self.wait(hold_time)
            self.play(FadeOut(eq), run_time=1.0)

        elif ANIMATION_TYPE == "morph":
            eq1 = MathTex(LATEX_EXPRESSION, font_size=72, color=eq_col)
            eq2 = MathTex(r"\boxed{" + LATEX_EXPRESSION + r"}", font_size=72)
            eq2.set_color(BLUE_C if BACKGROUND_STYLE != "light" else DARK_BLUE)

            self.play(Write(eq1), run_time=2.0)
            self.wait(hold_time * 0.35)
            self.play(TransformMatchingTex(eq1, eq2), run_time=1.5)
            self.wait(hold_time * 0.35)
            self.play(FadeOut(eq2), run_time=1.0)

        elif ANIMATION_TYPE == "step_by_step":
            eq = MathTex(LATEX_EXPRESSION, font_size=72, color=eq_col)
            self.play(Write(eq), run_time=2.0)
            self.wait(0.4)

            highlight_col = YELLOW if BACKGROUND_STYLE != "light" else GOLD_D
            reset_col = eq_col

            for part in eq:
                self.play(
                    part.animate.set_color(highlight_col).scale(1.25),
                    run_time=0.35,
                )
                self.play(
                    part.animate.set_color(reset_col).scale(1 / 1.25),
                    run_time=0.25,
                )

            self.wait(max(0.5, hold_time * 0.4))
            self.play(FadeOut(eq), run_time=1.0)

        else:
            # Fallback: same as "appear"
            eq = MathTex(LATEX_EXPRESSION, font_size=80, color=eq_col)
            self.play(Write(eq), run_time=2.0)
            self.wait(hold_time)
            self.play(FadeOut(eq), run_time=1.0)
