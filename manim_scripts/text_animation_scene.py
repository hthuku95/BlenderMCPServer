"""
text_animation_scene.py — Kinetic typography / text animation.

Args (via CHART_SCENE_ARGS env var, JSON):
    text:        str  — main text / title
    subtitle:    str  — optional second line
    mode:        "letter_by_letter" | "word_by_word" | "typewriter" | "wave" | "zoom_burst"
                 | "spin_in" | "color_cycle" | "highlight_words"
    color:       Manim colour name (main text)
    bg_color:    hex string or "dark" | "light"
    duration:    float
    font_size:   int (default 72)
    words_to_highlight: list[str] — for highlight_words mode, words to colour differently
"""
import json
import os
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_CMAP = {
    "WHITE": WHITE, "BLACK": BLACK, "BLUE": BLUE_B, "RED": RED_B,
    "GREEN": GREEN_B, "YELLOW": YELLOW, "ORANGE": ORANGE, "PURPLE": PURPLE_B,
    "TEAL": TEAL_B, "GOLD": GOLD, "PINK": PINK,
}


class TextAnimationScene(Scene):
    def construct(self):
        text_str    = _ARGS.get("text", "Make it Count")
        subtitle    = _ARGS.get("subtitle", "")
        mode        = _ARGS.get("mode", "letter_by_letter")
        col_name    = _ARGS.get("color", "WHITE")
        bg_style    = _ARGS.get("bg_color", "dark")
        duration    = float(_ARGS.get("duration", 8.0))
        font_size   = int(_ARGS.get("font_size", 72))
        highlights  = _ARGS.get("words_to_highlight", [])

        color = _CMAP.get(col_name.upper(), WHITE)

        if bg_style == "light":
            self.camera.background_color = WHITE
            color = _CMAP.get(col_name.upper(), BLACK)
        else:
            self.camera.background_color = "#0d1117"

        if mode == "letter_by_letter":
            self._letter_by_letter(text_str, subtitle, color, font_size, duration)
        elif mode == "word_by_word":
            self._word_by_word(text_str, subtitle, color, font_size, duration)
        elif mode == "typewriter":
            self._typewriter(text_str, subtitle, color, font_size, duration)
        elif mode == "wave":
            self._wave(text_str, subtitle, color, font_size, duration)
        elif mode == "zoom_burst":
            self._zoom_burst(text_str, subtitle, color, font_size, duration)
        elif mode == "spin_in":
            self._spin_in(text_str, subtitle, color, font_size, duration)
        elif mode == "color_cycle":
            self._color_cycle(text_str, subtitle, color, font_size, duration)
        elif mode == "highlight_words":
            self._highlight_words(text_str, subtitle, color, font_size, duration, highlights)
        else:
            self._letter_by_letter(text_str, subtitle, color, font_size, duration)

    def _letter_by_letter(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()

        # Split into individual letter mobjects via LaggedStart
        self.play(
            AddTextLetterByLetter(mob, time_per_char=max(0.05, 1.5 / max(len(text_str), 1))),
            run_time=min(3.0, duration * 0.4),
        )
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.wait(max(0.5, duration - self.renderer.time))

    def _word_by_word(self, text_str, subtitle, color, fs, duration):
        words = text_str.split()
        if not words:
            words = [text_str]

        # Build full text invisible, then reveal word by word
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()

        # Use ShowSubmobjectsOneByOne on words, keeping previous
        # Approximate by writing word vgroups
        time_per_word = max(0.2, (duration * 0.6) / len(words))
        y_pos = 0.5 if subtitle else 0
        x = -mob.width / 2
        shown = VGroup()

        for i, word in enumerate(words):
            w_mob = Text(word, font_size=fs, color=color, weight=BOLD)
            if i == 0:
                w_mob.move_to(LEFT * (len(text_str) / 2 * 0.22) + UP * y_pos)
            # Just use the full text but reveal with LaggedStart instead
            break

        # Simpler: LaggedStart on characters grouped by word
        self.play(
            LaggedStart(
                *[FadeIn(Text(w, font_size=fs, color=color, weight=BOLD).move_to(
                    mob.get_center() + LEFT * (mob.width/2 - mob.width * (sum(len(words[j])+1 for j in range(i)) / max(len(text_str),1)))
                ), shift=UP * 0.3)
                  for i, w in enumerate(words)],
                lag_ratio=0.3,
            ),
            run_time=min(3.5, duration * 0.5),
        )
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(ORIGIN + DOWN * 0.8, DOWN, buff=0.0)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _typewriter(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        cursor = Rectangle(width=0.06, height=mob.height * 1.1, color=color, fill_opacity=1)
        cursor.next_to(mob[0] if len(mob) > 0 else mob, LEFT, buff=0.0)

        self.play(
            AddTextLetterByLetter(mob, time_per_char=max(0.04, 1.8 / max(len(text_str), 1))),
            run_time=min(3.5, duration * 0.45),
        )
        self.play(Blink(cursor, n_times=3), run_time=0.6)
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_C)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub, shift=UP * 0.15), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _wave(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        self.play(
            LaggedStart(
                *[GrowFromCenter(char) for char in mob],
                lag_ratio=0.08,
            ),
            run_time=min(2.5, duration * 0.4),
        )
        # Wave distortion
        self.play(ApplyWave(mob, amplitude=0.3, wave_func=smooth), run_time=1.0)
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _zoom_burst(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        mob_big = mob.copy().scale(3).set_opacity(0)
        self.add(mob_big)
        self.play(
            mob_big.animate.scale(1/3).set_opacity(1),
            run_time=0.8,
        )
        self.play(Flash(mob.get_center(), color=color, line_length=0.5, num_lines=12), run_time=0.6)
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob_big, DOWN, buff=0.3)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _spin_in(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        self.play(
            LaggedStart(
                *[SpinInFromNothing(char) for char in mob],
                lag_ratio=0.06,
            ),
            run_time=min(3.0, duration * 0.45),
        )
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _color_cycle(self, text_str, subtitle, color, fs, duration):
        mob = Text(text_str, font_size=fs, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        colors = [RED_B, ORANGE, YELLOW, GREEN_B, BLUE_B, PURPLE_B]
        for i, char in enumerate(mob):
            char.set_color(colors[i % len(colors)])

        self.play(Write(mob), run_time=min(2.5, duration * 0.35))
        # Cycle colours
        n_cycles = max(1, int((duration - 3) / 1.5))
        for _ in range(n_cycles):
            new_colors = [colors[(i + 1) % len(colors)] for i in range(len(mob))]
            self.play(
                *[mob[i].animate.set_color(new_colors[i]) for i in range(len(mob))],
                run_time=0.8,
            )
        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))

    def _highlight_words(self, text_str, subtitle, color, fs, duration, highlights):
        mob = Text(text_str, font_size=fs, color=color, weight=BOLD)
        if mob.width > 12:
            mob.scale(12 / mob.width)
        mob.center()
        self.play(Write(mob), run_time=min(2.0, duration * 0.3))

        # Highlight specific words with Indicate
        highlight_colors = [YELLOW, RED_B, GREEN_B, ORANGE, PURPLE_B]
        for i, word in enumerate(highlights[:5]):
            # Find the word in the text mob using substring search
            try:
                word_mob = mob.get_part_by_text(word)
                self.play(
                    Indicate(word_mob, color=highlight_colors[i % len(highlight_colors)]),
                    run_time=0.7,
                )
            except Exception:
                pass  # word not found

        if subtitle:
            sub = Text(subtitle, font_size=int(fs * 0.45), color=GRAY_B)
            sub.next_to(mob, DOWN, buff=0.3)
            self.play(FadeIn(sub), run_time=0.5)
        self.wait(max(0.5, duration - self.renderer.time))
