"""
manim_codegen.py — LLM-driven Manim code generator with VIGA features.

VIGA additions:
- Multiple candidates with VLM tournament selection (VIGA_NUM_CANDIDATES env var)
- Parallel candidate generation and rendering
- Tournament bracket via Vision Language Model comparison
- Verifier quality gate (VIGA_ENABLE_VERIFIER env var)
- Docker sandbox pre-validation (VIGA_ENABLE_DOCKER_SANDBOX env var)
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from tools.rag_client import store_success, query_similar
from tools.verifier_loop import run_verifier_loop as _run_verifier_loop


_WEB_SEARCH_RE = re.compile(r"WEB_SEARCH:\s*(.+?)(?:\n|$)", re.IGNORECASE)

MAX_RETRIES = 5
SCENE_CLASS_NAME = "GeneratedScene"
NUM_CANDIDATES = int(os.getenv("VIGA_NUM_CANDIDATES", "1"))
USE_VLM_TOURNAMENT = NUM_CANDIDATES > 1
_VIGA_ENABLE_DOCKER = os.getenv("VIGA_ENABLE_DOCKER_SANDBOX", "").lower() in ("true", "1", "yes")


async def _web_search(query: str, num_results: int = 5) -> str:
    from tools.browserbase_client import browserbase_search
    return await browserbase_search(query, num_results)


async def _execute_web_search(text: str) -> str:
    results = []
    for match in _WEB_SEARCH_RE.finditer(text):
        query = match.group(1).strip()
        result = await _web_search(query)
        results.append(f"Search query: {query}\nResults:\n{result}")
    return "\n\n".join(results)


def _strip_web_search_markers(text: str) -> str:
    return _WEB_SEARCH_RE.sub("", text).strip()


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
    • The scene class MUST be named exactly `{scene_class}` and extend `Scene`.
    • Every object appearing on screen must be explicitly positioned.
    • Add `self.wait()` pauses between steps.
    • Do NOT reference external files (no SVG/PNG/MP3 file paths).
    • Target duration: ~{duration:.1f} seconds total.
    • Background colour: {background}.
    • Output ONLY valid Python code. No explanation, no markdown fences.

    ═══ WHITEBOARD STYLE (when background="light" or prompt says "whiteboard") ═══
    • Set `self.camera.background_color = "#F5F0E8"`.
    • Use `Write()` for text reveals, `Create()` for shapes.
    • Use dark marker colours.

    ═══ ERROR FIXING ═══
    If unsure about the correct Manim API, include:
        WEB_SEARCH: <natural language query about Manim API>

    ═══ YOUR TASK ═══
    {description}

    Write the complete Python class. Begin with `from manim import *`.
""")


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


async def _call_llm(prompt: str) -> str:
    from tools.llm_client import generate_text
    text, _ = await generate_text(prompt, temperature=0.3, max_tokens=4096)
    return text


async def _generate_code(description: str, duration: float, background: str) -> str:
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        scene_class=SCENE_CLASS_NAME, duration=duration,
        background=background, description=description,
    )
    raw = await _call_llm(prompt)
    return _extract_code(raw)


async def _fix_code(code: str, error: str, description: str, duration: float, background: str, search_results: str = "", rag_context: str = "") -> str:
    search_section = ""
    if search_results:
        search_section = f"\n═══ WEB SEARCH RESULTS ═══\n{search_results}\n"
    rag_section = rag_context if rag_context else ""

    fix_prompt = textwrap.dedent(f"""\
        The following ManimCE v0.20.x Python code failed to execute.

        ═══ ORIGINAL TASK ═══
        {description}
        {search_section}
        {rag_section}
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
        • If unsure about the correct Manim API, include:
            WEB_SEARCH: <query about the correct API>
        • Simplify rather than guess — use a simpler animation.
        • Target duration: ~{duration:.1f} seconds.
        • Output ONLY the corrected Python code. No explanation.
    """)
    raw = await _call_llm(fix_prompt)
    return _extract_code(raw)


async def _docker_validate(code: str, script_type: str = "manim") -> str | None:
    from tools.docker_sandbox import validate_in_docker
    result = await validate_in_docker(code, script_type, timeout=15)
    if not result["passed"]:
        logs = result.get("logs", "")
        error = result.get("error", "Docker sandbox validation failed")
        if len(logs) > 3000:
            logs = logs[-3000:]
        return f"[Docker sandbox pre-check] {error}\nLogs:\n{logs}"
    return None


async def _render_manim_single(description: str, duration: float, background: str, output_path: str, transparent: bool, quality: str, attempt: int) -> dict:
    from tools.manim_runner import run_manim_scene

    code = await _generate_code(description, duration, background)
    search_results = await _execute_web_search(code)
    if search_results:
        code = _strip_web_search_markers(code)
    code = _extract_code(code)

    if _VIGA_ENABLE_DOCKER:
        dock_err = await _docker_validate(code)
        if dock_err:
            return {"error": dock_err, "code": code}

    with tempfile.NamedTemporaryFile(suffix=".py", prefix=f"manim_candidate_{attempt}_", delete=False, mode="w") as f:
        f.write(code)
        scene_file = f.name

    try:
        result = await run_manim_scene(
            scene_file=scene_file, scene_class=SCENE_CLASS_NAME,
            args={}, quality=quality, output_path=output_path,
            transparent=transparent, timeout=300,
        )
        if os.path.exists(result):
            return {"path": result, "code": code, "score": None}
        return {"error": "no output file", "code": code}
    except RuntimeError as e:
        return {"error": str(e)[-2000:], "code": code}
    finally:
        try:
            os.unlink(scene_file)
        except OSError:
            pass


async def _run_vlm_tournament(results: list[dict], description: str) -> int:
    from tools.vision_tools import compare_render_to_reference

    valid = [(i, r) for i, r in enumerate(results) if "path" in r and os.path.exists(r["path"])]
    if len(valid) <= 1:
        return valid[0][0] if valid else 0

    candidates = [(idx, r["path"]) for idx, r in valid]
    while len(candidates) > 1:
        next_round = []
        for i in range(0, len(candidates), 2):
            if i + 1 < len(candidates):
                idx_a, path_a = candidates[i]
                idx_b, path_b = candidates[i + 1]
                try:
                    comp = compare_render_to_reference(
                        render_path=path_b, reference_path_or_url=path_a, prompt_context=description,
                    )
                    score_a = comp.get("match_score", 0.5)
                    comp_rev = compare_render_to_reference(
                        render_path=path_a, reference_path_or_url=path_b, prompt_context=description,
                    )
                    score_b = comp_rev.get("match_score", 0.5)
                    winner = idx_a if score_a >= score_b else idx_b
                except Exception:
                    winner = idx_a
                next_round.append((winner, path_a if winner == idx_a else path_b))
            else:
                next_round.append(candidates[i])
        candidates = next_round
    return candidates[0][0] if candidates else 0


async def _retry_render_single(description: str, duration: float, background: str, output_path: str, transparent: bool, quality: str) -> tuple[str, str]:
    from tools.manim_runner import run_manim_scene

    code = ""
    last_error = ""
    search_results = ""

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            code = await _generate_code(description, duration, background)
        else:
            rag_context = await query_similar("manim", last_error, description)
            code = await _fix_code(code, last_error, description, duration, background, search_results, rag_context)

        search_results = await _execute_web_search(code)
        if search_results:
            code = _strip_web_search_markers(code)

        if _VIGA_ENABLE_DOCKER:
            dock_err = await _docker_validate(code)
            if dock_err:
                last_error = dock_err
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Manim code failed Docker sandbox validation after {MAX_RETRIES} attempts.\n"
                        f"Last error:\n{last_error}"
                    )
                continue

        with tempfile.NamedTemporaryFile(suffix=".py", prefix=f"manim_gen_{attempt}_", delete=False, mode="w") as f:
            f.write(code)
            scene_file = f.name

        try:
            result = await run_manim_scene(
                scene_file=scene_file, scene_class=SCENE_CLASS_NAME,
                args={}, quality=quality, output_path=output_path,
                transparent=transparent, timeout=300,
            )
            await store_success("manim", code, description)
            try:
                os.unlink(scene_file)
            except OSError:
                pass
            return (result, code)
        except RuntimeError as e:
            last_error = str(e)[-2000:]
            try:
                os.unlink(scene_file)
            except OSError:
                pass
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"LLM-generated Manim failed after {MAX_RETRIES} attempts. "
                    f"Last error:\n{last_error}"
                ) from e

    raise RuntimeError("_retry_render_single: unexpected exit")


async def generate_and_run_manim(
    description: str,
    duration: float = 10.0,
    background: str = "dark",
    output_path: Optional[str] = None,
    transparent: bool = False,
    quality: str = "m",
) -> str:
    if output_path is None:
        ext = ".mov" if transparent else ".mp4"
        output_path = f"/tmp/manim_gen_{os.getpid()}{ext}"

    async def _render_manim(description: str, **kwargs) -> tuple[str, str]:
        if not USE_VLM_TOURNAMENT:
            return await _retry_render_single(description, duration, background, output_path, transparent, quality)

        candidate_dir = tempfile.mkdtemp(prefix="manim_candidates_")
        try:
            candidate_paths = [
                os.path.join(candidate_dir, f"candidate_{i}.mp4")
                for i in range(NUM_CANDIDATES)
            ]
            tasks = [
                _render_manim_single(description, duration, background, cp, transparent, quality, i + 1)
                for i, cp in enumerate(candidate_paths)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    processed.append({"error": str(r), "code": ""})
                else:
                    processed.append(r)
            winner_idx = await _run_vlm_tournament(processed, description)
            winner = processed[winner_idx]
            if "path" in winner and os.path.exists(winner["path"]):
                shutil.copy2(winner["path"], output_path)
                return (output_path, winner.get("code", ""))
            return await _retry_render_single(description, duration, background, output_path, transparent, quality)
        finally:
            shutil.rmtree(candidate_dir, ignore_errors=True)

    final_path = await _run_verifier_loop(
        render_fn=_render_manim,
        prompt=description,
        code="",
        render_kwargs={
            "duration": duration,
            "background": background,
            "quality": quality,
            "transparent": transparent,
        },
    )
    return final_path
