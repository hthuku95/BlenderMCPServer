"""
code_animation_scene.py — Manim Code object with line-by-line reveal and highlighting.

Args (via CHART_SCENE_ARGS env var, JSON):
    code:           str  — the code string to display
    language:       str  — "python" | "javascript" | "rust" | "cpp" | "java" | "bash" | "sql"
    title:          str
    highlight_lines: list[int] — 1-indexed lines to highlight after reveal
    reveal_mode:    "all_at_once" | "line_by_line" | "block"
    duration:       float
    style:          "monokai" | "dracula" | "solarized-dark"
"""
import json
import os
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_DEFAULT_CODE = '''\
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Generate first 10 Fibonacci numbers
result = [fibonacci(i) for i in range(10)]
print(result)
'''


class CodeAnimationScene(Scene):
    def construct(self):
        code_str       = _ARGS.get("code", _DEFAULT_CODE)
        language       = _ARGS.get("language", "python")
        title_text     = _ARGS.get("title", "Code Walkthrough")
        highlight_lines = _ARGS.get("highlight_lines", [])
        reveal_mode    = _ARGS.get("reveal_mode", "line_by_line")
        duration       = float(_ARGS.get("duration", 12.0))
        theme          = _ARGS.get("style", "monokai")

        self.camera.background_color = "#1e1e2e"

        title = Text(title_text, font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.3)

        code_mob = Code(
            code=code_str,
            tab_width=4,
            background="window",
            language=language,
            font="Monospace",
            style=theme,
            font_size=22,
        )

        # Scale to fit screen width
        max_w = 11.5
        if code_mob.width > max_w:
            code_mob.scale(max_w / code_mob.width)

        code_mob.center()
        if title_text:
            code_mob.shift(DOWN * 0.4)

        self.play(Write(title), run_time=0.6)

        if reveal_mode == "all_at_once":
            self.play(FadeIn(code_mob), run_time=1.5)
        elif reveal_mode == "line_by_line":
            # Reveal background/window frame first
            self.play(FadeIn(code_mob.background_mobject), run_time=0.5)
            lines = code_mob.code  # VGroup of code lines
            n_lines = len(lines)
            time_per_line = max(0.15, min(0.5, (duration - 3.0) / max(n_lines, 1)))
            for line in lines:
                self.play(FadeIn(line, shift=RIGHT * 0.1), run_time=time_per_line)
        else:
            # block mode: animate whole code in chunks
            self.play(Create(code_mob), run_time=2.5)

        # Highlight specific lines
        if highlight_lines:
            lines = code_mob.code
            rects = []
            for ln in highlight_lines:
                idx = ln - 1
                if 0 <= idx < len(lines):
                    rect = BackgroundRectangle(
                        lines[idx], color=YELLOW, fill_opacity=0.35, buff=0.05
                    )
                    rects.append(rect)
            if rects:
                self.play(*[FadeIn(r) for r in rects], run_time=0.8)

        self.wait(max(0.5, duration - self.renderer.time))
