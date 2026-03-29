"""
LLM-driven Manim code generator with sandbox execution and retry loop.

Architecture
------------
1. Build a system prompt containing:
   - ManimCE version constraints and deprecated-API blocklist
   - Two-phase instruction (write plan as comments, then write code)
   - Three complete few-shot examples demonstrating correct patterns
2. Call the active LLM (Gemini/Claude via llm_client.generate_text)
3. Static pre-validation: syntax check + deprecated-API scan
4. Sandbox execution via manim_runner.run_manim_scene()
5. On failure: inject (code + error) back into LLM, up to MAX_RETRIES attempts
6. On final failure: raise RuntimeError (caller decides whether to use a fallback)

Usage
-----
    from tools.manim_codegen import generate_and_run_manim

    output_path = await generate_and_run_manim(
        description="Animate the quadratic formula, step by step, "
                    "with each term highlighted in a different colour",
        duration=10.0,
        background="dark",        # "dark" | "light" | "transparent"
        output_path="/tmp/my_anim.mov",
        transparent=True,
    )
"""
from __future__ import annotations

import ast
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 5
SCENE_CLASS_NAME = "GeneratedScene"

# Deprecated / wrong-version identifiers that indicate the LLM hallucinated
# 3b1b-manim APIs instead of ManimCE APIs.
_BANNED_PATTERNS = [
    "ShowCreation",       # renamed to Create() in v0.6
    "manim_imports_ext",  # 3b1b private import
    "CONFIG = {",         # old class-level config dict
    "self.embed()",       # 3b1b interactive mode
    "InteractiveScene",   # 3b1b only
    "GraphScene",         # deprecated; use Axes/NumberPlane
    "from manimlib",      # 3b1b's package, not ManimCE
    "PiCreature",         # 3b1b mascot, not in ManimCE
    "TextMobject",        # renamed to Tex() in v0.6
    "TexMobject",         # renamed to MathTex() in v0.6
]

# ──────────────────────────────────────────────────────────────────────────────
# System prompt with few-shot examples
# ──────────────────────────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLES = '''
### EXAMPLE 1 — Colour-coded equation reveal

```python
from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Plan:
        # 1. Show E=mc² with each symbol in a different colour
        # 2. Indicate the 'E' term, then the 'mc²' term
        # 3. Hold so viewers can read it

        eq = MathTex(r"E", "=", r"m", r"c^2")
        eq[0].set_color(YELLOW)   # E
        eq[1].set_color(WHITE)    # =
        eq[2].set_color(BLUE)     # m
        eq[3].set_color(RED)      # c²
        eq.scale(2).move_to(ORIGIN)

        self.play(Write(eq), run_time=2)
        self.wait(0.5)
        self.play(Indicate(eq[0], color=YELLOW, scale_factor=1.4), run_time=1)
        self.wait(0.3)
        self.play(Indicate(VGroup(eq[2], eq[3]), color=BLUE, scale_factor=1.3), run_time=1)
        self.wait(2)
```

### EXAMPLE 2 — Step-by-step derivation with TransformMatchingTex

```python
from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Plan:
        # Show the quadratic formula being built step by step.
        # Use TransformMatchingTex to morph between steps.
        # Add a surrounding box on the final result.

        title = Text("Quadratic Formula", font_size=36, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1)
        self.wait(0.3)

        step1 = MathTex(r"ax^2 + bx + c = 0")
        step2 = MathTex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")

        for mob in (step1, step2):
            mob.scale(1.4).move_to(ORIGIN)

        self.play(Write(step1), run_time=1.5)
        self.wait(1)
        self.play(TransformMatchingTex(step1, step2), run_time=2)
        self.wait(0.5)

        box = SurroundingRectangle(step2, color=GOLD, buff=0.2)
        self.play(Create(box), run_time=0.8)
        self.wait(2)
```

### EXAMPLE 3 — Animated bar chart with LaggedStart

```python
from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Plan:
        # Show an animated bar chart of 5 values.
        # Bars grow from bottom with a stagger effect.
        # Add value labels that appear after the bars.

        chart = BarChart(
            values=[3, 7, 5, 9, 4],
            bar_names=["Jan", "Feb", "Mar", "Apr", "May"],
            y_range=[0, 10, 2],
            y_length=5,
            x_length=9,
            bar_colors=[BLUE, TEAL, GREEN, YELLOW, ORANGE],
        )
        chart.move_to(ORIGIN)

        title = Text("Monthly Sales", font_size=32).next_to(chart, UP, buff=0.3)

        self.play(Write(title), run_time=0.8)
        self.play(Create(chart), run_time=2.5)
        self.wait(0.5)

        labels = chart.get_bar_labels(font_size=28)
        self.play(LaggedStart(*[FadeIn(l, shift=UP*0.3) for l in labels],
                               lag_ratio=0.15), run_time=1.5)
        self.wait(2)
```
'''

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert Manim animation programmer using ManimCE v0.20.x.

    ═══ CRITICAL VERSION RULES (violating these causes runtime errors) ═══
    • Use `Create()` — NOT `ShowCreation()` (deprecated in v0.6)
    • Import with `from manim import *` — NOT `from manimlib import *`
    • Do NOT use: ShowCreation, manim_imports_ext, CONFIG={{}}, self.embed(),
      InteractiveScene, GraphScene, PiCreature, TextMobject, TexMobject
    • FadeIn direction: `FadeIn(obj, shift=UP)` — NOT `FadeIn(obj, UP)`
    • TransformMatchingTex: for equations. TransformMatchingShapes: for text.

    ═══ STRUCTURE REQUIREMENTS ═══
    • The scene class MUST be named exactly `{scene_class}` and extend `Scene`
      (or `MovingCameraScene`, `ThreeDScene`, `ZoomedScene`).
    • Every object appearing on screen must be explicitly positioned
      (`.to_edge()`, `.move_to()`, `.next_to()`, `.shift()`) before `self.play()`.
    • Add `self.wait()` pauses between steps — viewers need time to read.
    • Do NOT reference external files (no SVG/PNG/MP3 file paths).
    • Target duration: ~{duration:.1f} seconds total.
    • Background colour: {background} (set `background_color` if overriding).
    • Output ONLY valid Python code. No explanation, no markdown fences.

    ═══ FEW-SHOT EXAMPLES (these patterns are known to work) ═══
    {examples}

    ═══ YOUR TASK ═══
    {description}

    Write the complete Python class. Begin with `from manim import *`.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Static pre-validator
# ──────────────────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    # ```python ... ``` or ``` ... ```
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _static_validate(code: str) -> Optional[str]:
    """
    Check the generated code before running Manim.
    Returns an error description string, or None if the code looks OK.
    """
    # 1. Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Python SyntaxError on line {e.lineno}: {e.msg}"

    # 2. Banned / deprecated API patterns
    for pattern in _BANNED_PATTERNS:
        if pattern in code:
            return (
                f"Deprecated or wrong-version API detected: '{pattern}'. "
                "Use ManimCE v0.20.x APIs only."
            )

    # 3. Scene class present
    if f"class {SCENE_CLASS_NAME}" not in code:
        return (
            f"Scene class '{SCENE_CLASS_NAME}' not found. "
            f"The class must be named exactly '{SCENE_CLASS_NAME}'."
        )

    # 4. construct method present
    if "def construct(self)" not in code:
        return "Missing `def construct(self):` method inside the scene class."

    # 5. Correct import
    if "from manim import" not in code and "import manim" not in code:
        return "Missing Manim import. Add `from manim import *` at the top."

    return None  # all checks passed


# ──────────────────────────────────────────────────────────────────────────────
# LLM code generation
# ──────────────────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> str:
    """Generate text via the active LLM provider (Gemini / Claude)."""
    from tools.llm_client import generate_text
    text, _ = await generate_text(
        prompt,
        temperature=0.3,   # low temp → more consistent code
        max_tokens=4096,
    )
    return text


async def _generate_code(description: str, duration: float, background: str) -> str:
    """Ask the LLM to produce a Manim Python class for the given description."""
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        scene_class=SCENE_CLASS_NAME,
        duration=duration,
        background=background,
        examples=_FEW_SHOT_EXAMPLES,
        description=description,
    )
    raw = await _call_llm(prompt)
    return _extract_code(raw)


async def _fix_code(code: str, error: str, description: str, duration: float, background: str) -> str:
    """Ask the LLM to fix failing code given the execution error."""
    prompt = textwrap.dedent(f"""\
        The following ManimCE v0.20.x Python code failed to execute.

        ═══ ORIGINAL TASK ═══
        {description}

        ═══ FAILING CODE ═══
        ```python
        {code}
        ```

        ═══ ERROR ═══
        {error}

        ═══ INSTRUCTIONS ═══
        • Fix the error shown above.
        • Keep the overall animation intent the same.
        • The class must still be named `{SCENE_CLASS_NAME}`.
        • Do NOT use deprecated APIs (ShowCreation, TextMobject, etc.).
        • If the error mentions a missing attribute or wrong argument, simplify
          rather than guess — use a simpler animation that is guaranteed to work.
        • Target duration: ~{duration:.1f} seconds.
        • Output ONLY the corrected Python code. No explanation.
    """)
    raw = await _call_llm(prompt)
    return _extract_code(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

async def generate_and_run_manim(
    description: str,
    duration: float = 10.0,
    background: str = "dark",
    output_path: Optional[str] = None,
    transparent: bool = False,
    quality: str = "m",
) -> str:
    """
    Generate a Manim animation from a natural language description and render it.

    Args:
        description:  Natural language description of the desired animation.
        duration:     Target clip duration in seconds.
        background:   Background style hint for the LLM ("dark"|"light"|"transparent").
        output_path:  Destination file path.  Extension is auto-adjusted for transparent.
        transparent:  If True render with alpha channel (ProRes .mov).
        quality:      Manim quality flag (l=480p, m=720p, h=1080p).

    Returns:
        Absolute path to the rendered file.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    from tools.manim_runner import run_manim_scene

    if output_path is None:
        ext = ".mov" if transparent else ".mp4"
        output_path = f"/tmp/manim_gen_{os.getpid()}{ext}"

    code: str = ""
    last_error: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        # ── generate / fix ────────────────────────────────────────────────────
        if attempt == 1:
            code = await _generate_code(description, duration, background)
        else:
            code = await _fix_code(code, last_error, description, duration, background)

        # ── static validate ───────────────────────────────────────────────────
        static_err = _static_validate(code)
        if static_err:
            last_error = f"Static validation failed: {static_err}"
            continue  # count this as an attempt, loop to fix

        # ── write to temp file ────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            prefix=f"manim_gen_{attempt}_",
            delete=False,
            mode="w",
        ) as f:
            f.write(code)
            scene_file = f.name

        # ── run manim ─────────────────────────────────────────────────────────
        try:
            result = await run_manim_scene(
                scene_file=scene_file,
                scene_class=SCENE_CLASS_NAME,
                args={},          # args are baked into the generated code
                quality=quality,
                output_path=output_path,
                transparent=transparent,
                timeout=300,
            )
            # success
            try:
                os.unlink(scene_file)
            except OSError:
                pass
            return result

        except RuntimeError as e:
            last_error = str(e)[-2000:]  # keep last 2000 chars to avoid context overflow
            try:
                os.unlink(scene_file)
            except OSError:
                pass

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"LLM-generated Manim failed after {MAX_RETRIES} attempts. "
                    f"Last error:\n{last_error}"
                ) from e
            # else: loop to fix

    # Should not reach here, but satisfy type checker
    raise RuntimeError("generate_and_run_manim: unexpected exit from retry loop")
