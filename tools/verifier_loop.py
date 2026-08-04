"""
VIGA verifier loop: wraps generate_and_run_bpy / generate_and_run_manim
with an optional quality gate.  After every successful render the verifier
VLM reviews the output.  If it scores < 0.7 the render is rejected and a
fix prompt is fed back to the generator for re-render (up to N iterations).

This module is imported by bpy_codegen and manim_codegen — it is the
shared quality-assurance layer for all DFY renders.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_VIGA_ENABLE_VERIFIER = os.getenv("VIGA_ENABLE_VERIFIER", "").lower() in ("true", "1", "yes")
_VIGA_MAX_ITERATIONS = int(os.getenv("VIGA_VERIFIER_MAX_ITERATIONS", "3"))


async def run_verifier_loop(
    *,
    render_fn,
    prompt: str,
    code: str = "",
    render_kwargs: dict | None = None,
) -> str:
    """Wrap a render function with VIGA verifier quality gating.

    Args:
        render_fn:     Async callable that returns (output_path: str, code: str).
                       Signature: render_fn(prompt=..., **render_kwargs) -> (str, str)
                       The caller must set all render-relevant kwargs in render_kwargs
                       (e.g. style, duration, reference_image_url for Blender;
                        duration, background, quality, transparent for Manim).
        prompt:        Original creative brief (or the latest fix_prompt).
        code:          Generated code from the most recent attempt (empty on first call).
        render_kwargs: Extra kwargs passed to render_fn (e.g. duration, style, etc.).

    Returns:
        Local path to the accepted render.
    """
    from tools.storage import upload_render
    from agents.verifier import verify_and_suggest_fixes as _vsf

    if not _VIGA_ENABLE_VERIFIER:
        path, _ = await render_fn(prompt=prompt, **(render_kwargs or {}))
        return path

    current_prompt = prompt
    previous_feedback = []
    final_path = None
    final_code = code

    for iteration in range(1, _VIGA_MAX_ITERATIONS + 1):
        output_path, generated_code = await render_fn(
            prompt=current_prompt,
            **(render_kwargs or {}),
        )

        if not output_path or not os.path.exists(output_path):
            logger.warning("VIGA verifier: render_fn returned no output (iteration %d)", iteration)
            if iteration == _VIGA_MAX_ITERATIONS:
                raise RuntimeError(
                    f"VIGA verifier: render failed after {_VIGA_MAX_ITERATIONS} iterations"
                )
            continue

        final_path = output_path
        final_code = generated_code or final_code

        r2_url = upload_render(output_path, prefix="vigav_renders")

        feedback = await _vsf(
            prompt=current_prompt,
            video_url=r2_url,
            code=final_code,
            previous_feedback=previous_feedback,
        )
        previous_feedback.append({
            "iteration": iteration,
            "score": feedback.get("quality_score", 0.0),
            "summary": feedback.get("summary", ""),
        })

        score = feedback.get("quality_score", 0.0)
        pass_fail = feedback.get("pass_fail", "fail")

        logger.info(
            "VIGA verifier iteration=%d score=%.2f pass=%s prompt_len=%d code_len=%d",
            iteration, score, pass_fail, len(current_prompt), len(final_code),
        )

        if pass_fail == "pass" and score >= 0.7:
            logger.info("VIGA verifier: accepted after %d iteration(s)", iteration)
            return output_path

        # Prepare fix prompt for next iteration
        fix_prompt = feedback.get("fix_prompt", "")
        if fix_prompt:
            current_prompt = fix_prompt
        else:
            issues = "; ".join(feedback.get("issues", []))
            current_prompt = (
                f"Improve the following render. Issues: {issues}\n\n"
                f"Original brief: {prompt}"
            )

    logger.warning(
        "VIGA verifier: all %d iterations exhausted, returning last render (score=%.2f)",
        _VIGA_MAX_ITERATIONS,
        previous_feedback[-1].get("score", 0.0) if previous_feedback else 0.0,
    )
    if final_path and os.path.exists(final_path):
        return final_path
    raise RuntimeError(
        f"VIGA verifier: all {_VIGA_MAX_ITERATIONS} iterations exhausted with no output"
    )
