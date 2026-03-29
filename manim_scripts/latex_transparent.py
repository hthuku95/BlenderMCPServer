"""
Manim LaTeX scene with a **transparent background** — for Option B compositing.

Unlike latex_scene.py (which renders on dark/light background), this scene renders
the equation as a transparent-background clip so MoviePy can composite it over
a Blender 3D scene.

Usage (via manim CLI):
    manim -qm --transparent latex_transparent.py LatexTransparent

Args (passed via LATEX_SCENE_ARGS env var as JSON):
    latex_expression: str   — LaTeX math (bare or wrapped)
    animation_type: str     — "appear" | "morph" | "step_by_step"
    duration: float         — Total clip length in seconds

Output: transparent-background .mov (ProRes 4444 or PNG frames)
        Manim writes to media/videos/latex_transparent/<quality>/LatexTransparent.mov
"""
from __future__ import annotations

import json
import os

from manim import *



def _load_args() -> dict:
    raw = os.environ.get("LATEX_SCENE_ARGS", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


class LatexTransparent(Scene):
    """
    Transparent-background LaTeX animation for Option B compositing.

    Background colour is intentionally set to TRANSPARENT (default when
    manim is called with --transparent).
    """

    def construct(self) -> None:
        args = _load_args()
        expr: str = args.get("latex_expression", r"E = mc^2")
        anim_type: str = args.get("animation_type", "appear")
        duration: float = float(args.get("duration", 8.0))

        # Normalise expression for MathTex.
        # MathTex wraps its argument in \begin{align*}...\end{align*} internally,
        # so passing \[...\] or $...$ wrappers causes double-nesting and a LaTeX
        # compile error.  Strip any display-math delimiters before handing off.
        expr = expr.strip()
        expr = expr.replace(r"\[", "").replace(r"\]", "").strip()
        if expr.startswith("$$") and expr.endswith("$$"):
            expr = expr[2:-2].strip()
        elif expr.startswith("$") and expr.endswith("$"):
            expr = expr[1:-1].strip()

        # White equation on transparent background
        eq = MathTex(expr, color=WHITE)
        eq.scale_to_fit_width(config.frame_width * 0.85)

        if anim_type == "appear":
            self.play(Write(eq), run_time=duration * 0.4)
            self.wait(duration * 0.6)

        elif anim_type == "morph":
            # Morph between three 'steps' by decomposing the expression
            # into segments if it contains \frac, \int, or \sum
            parts = _split_expression(expr)
            if len(parts) < 2:
                self.play(Write(eq), run_time=duration * 0.4)
                self.wait(duration * 0.6)
            else:
                first = MathTex(parts[0], color=WHITE).scale_to_fit_width(
                    config.frame_width * 0.85
                )
                self.play(Write(first), run_time=duration * 0.3)
                for part in parts[1:]:
                    next_eq = MathTex(part, color=WHITE).scale_to_fit_width(
                        config.frame_width * 0.85
                    )
                    self.play(
                        TransformMatchingTex(first, next_eq),
                        run_time=duration * 0.25,
                    )
                    first = next_eq
                self.wait(duration * 0.2)

        elif anim_type == "step_by_step":
            # Reveal each term one at a time
            terms = _tokenise_terms(expr)
            if len(terms) < 2:
                self.play(Write(eq), run_time=duration * 0.4)
                self.wait(duration * 0.6)
            else:
                step_time = (duration * 0.8) / len(terms)
                revealed = VGroup()
                x_cursor = -config.frame_width / 2 + 0.5

                for term in terms:
                    t_mob = MathTex(term, color=WHITE)
                    t_mob.next_to(revealed, RIGHT, buff=0.15) if len(revealed) > 0 else t_mob.move_to(
                        LEFT * (config.frame_width / 2 - 0.5)
                    )
                    self.play(FadeIn(t_mob, shift=UP * 0.3), run_time=step_time)
                    revealed.add(t_mob)

                # Regroup and centre
                self.play(revealed.animate.move_to(ORIGIN), run_time=0.5)
                self.wait(duration * 0.2)
        else:
            self.play(Write(eq), run_time=duration * 0.4)
            self.wait(duration * 0.6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_expression(expr: str) -> list[str]:
    """
    Very simple splitter: returns [lhs, full_expr] if '=' found,
    otherwise returns [expr] (no morphing).
    """
    # Strip display math wrappers
    inner = expr.replace(r"\[", "").replace(r"\]", "").strip()
    if "=" in inner:
        lhs = inner.split("=")[0].strip()
        return [lhs, inner]
    return [expr]


def _tokenise_terms(expr: str) -> list[str]:
    """
    Split a LaTeX expression into individual terms (split on + / - / = at top level).
    Returns the original expression in a single-element list if splitting is not useful.
    """
    inner = expr.replace(r"\[", "").replace(r"\]", "").strip()
    tokens: list[str] = []
    depth = 0
    current = ""
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch in "{":
            depth += 1
            current += ch
        elif ch in "}":
            depth -= 1
            current += ch
        elif ch in ("+", "-", "=") and depth == 0:
            if current.strip():
                tokens.append(current.strip())
            tokens.append(ch)
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        tokens.append(current.strip())
    return tokens if len(tokens) > 1 else [expr]
