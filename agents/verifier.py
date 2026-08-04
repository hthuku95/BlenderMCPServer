"""
agents/verifier.py — VIGA-style Verifier Agent

This LangGraph agent reviews rendered video output against the original prompt
and provides structured feedback. It has access to investigator tools that let
it examine 3D scenes (viewpoint control, object visibility, scene analysis).

Workflow:
1. Receives: rendered video URL, original prompt, generated bpy code, .blend state file
2. Analyzes video using VLM (Gemini/Claude) — watches full video
3. Investigates 3D scene using investigator tools (camera navigation, object inspection, render settings check)
4. Produces structured feedback report for the Generator
5. Generator uses feedback to produce improved code (up to VIGA_VERIFIER_MAX_ITERATIONS, default 3)

The Verifier's structured output includes:
- quality_score: 0.0–1.0
- pass_fail: pass | fail
- issues: list of specific problems found
- strengths: list of what worked well
- fix_suggestions: actionable improvement instructions for the Generator
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

VERIFIER_MAX_ITERATIONS = int(os.getenv("VIGA_VERIFIER_MAX_ITERATIONS", "3"))


def _build_verifier_prompt(
    prompt: str,
    video_url: str,
    code: str,
    blender_file_path: str = "",
    previous_feedback: Optional[list[dict]] = None,
) -> str:
    """Build the prompt for the Verifier VLM."""
    feedback_section = ""
    if previous_feedback:
        feedback_section = "\n".join(
            f"- Iteration {fb.get('iteration', '?')}: {fb.get('summary', '')}"
            for fb in previous_feedback
        )

    return f"""\
You are a VIGA Verifier Agent. Your job is to evaluate a rendered 3D animation
video against its original creative brief and judge quality.

═══ ORIGINAL PROMPT ═══
{prompt}

═══ GENERATED BLENDER CODE ═══
```python
{code[:8000]}
```

═══ OUTPUT VIDEO URL ═══
{video_url}

{f"═══ BLENDER FILE ═══\n{blender_file_path}" if blender_file_path else ""}
{f"═══ PREVIOUS FEEDBACK HISTORY ═══\n{feedback_section}" if feedback_section else ""}

═══ EVALUATION CRITERIA ═══
Rate the rendered video on these dimensions (1-10):
1. **Prompt alignment**: Does the video match the creative brief? Does it
   include the requested elements, mood, and style?
2. **Visual quality**: Lighting, materials, composition, camera work. Is it
   polished and professional-looking?
3. **Animation quality**: Smooth motion, appropriate pacing, good keyframing.
   Does it look natural and intentional?
4. **Technical correctness**: No visible artifacts, proper rendering, correct
   duration and resolution. No missing objects, flickering, or broken geometry.
5. **Creative execution**: Does it go beyond the minimum? Does it show creative
   interpretation and thoughtful design?

═══ OUTPUT FORMAT ═══
You MUST return a JSON object with these fields:
{{
  "quality_score": <float 0.0-1.0>,
  "pass_fail": "pass" or "fail",
  "dimension_scores": {{
    "prompt_alignment": <int 1-10>,
    "visual_quality": <int 1-10>,
    "animation_quality": <int 1-10>,
    "technical_correctness": <int 1-10>,
    "creative_execution": <int 1-10>
  }},
  "summary": "<1-2 sentence summary>",
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "strengths": ["<strength 1>", "<strength 2>"],
  "fix_suggestions": [
    "<actionable instruction for Generator - what to change in the bpy code>"
  ],
  "investigation_findings": {{
    "scene_state": "<what the scene investigation found>",
    "render_settings": "<render config notes>",
    "objects_found": "<notable objects or missing elements>"
  }}
}}

Be critical but fair. A score of 0.7+ is "pass". Below 0.7 is "fail".
For fails, provide SPECIFIC fix suggestions the Generator can follow.
"""


async def review_render(
    prompt: str,
    video_url: str,
    code: str,
    blender_file_path: str = "",
    previous_feedback: Optional[list[dict]] = None,
) -> dict:
    """Call the Verifier VLM to review a rendered video and return structured feedback."""
    from tools.llm_client import generate_text

    sys_prompt = _build_verifier_prompt(prompt, video_url, code, blender_file_path, previous_feedback)
    text, _ = await generate_text(sys_prompt, temperature=0.2, max_tokens=4096)

    # Parse JSON from response
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text else text
    if text.endswith("```"):
        text = text[:-3]

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Verifier LLM returned non-JSON, attempting to extract")
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {
                    "quality_score": 0.5,
                    "pass_fail": "fail",
                    "summary": "Could not parse Verifier output",
                    "issues": ["LLM output parsing failed"],
                    "strengths": [],
                    "fix_suggestions": ["Retry with different parameters"],
                }
        else:
            result = {
                "quality_score": 0.5,
                "pass_fail": "fail",
                "summary": text[:500],
                "issues": [],
                "strengths": [],
                "fix_suggestions": [],
            }

    result["video_url"] = video_url
    return result


async def run_investigation(blender_file_path: str, prompt: str) -> dict:
    """Run scene investigation tools on the .blend file and return findings.

    This provides the Verifier with ground-truth scene data to inform its review.
    """
    if not blender_file_path or not os.path.exists(blender_file_path):
        return {
            "scene_state": "No .blend file available for investigation",
            "render_settings": "N/A",
            "objects_found": "N/A",
        }

    os.environ["BLENDER_CURRENT_FILE"] = blender_file_path
    from tools import blender_investigator as bi

    # Gather scene info
    scene_info = await bi.get_scene_info()
    render_info = await bi.investigate_render()

    # If there's a camera, get its viewpoint for reference
    viewpoint_info = await bi.initialize_viewpoint()

    findings = {
        "scene_state": f"{scene_info.get('total_objects', 0)} objects, "
                       f"{scene_info.get('total_materials', 0)} materials, "
                       f"{scene_info.get('object_types', {})}",
        "render_settings": f"{render_info.get('render_engine', 'N/A')} at "
                           f"{render_info.get('resolution_x', '?')}x{render_info.get('resolution_y', '?')}, "
                           f"{render_info.get('fps', '?')}fps, "
                           f"frames {render_info.get('frame_start')}-{render_info.get('frame_end')}",
        "objects_found": f"Camera: {viewpoint_info.get('has_camera', False)}, "
                         f"cam pos: {viewpoint_info.get('camera_location')}",
        "raw_scene_info": scene_info,
        "raw_render_info": render_info,
    }
    return findings


async def generate_fix_prompt(
    prompt: str,
    code: str,
    feedback: dict,
    investigation_results: dict,
) -> str:
    """Generate a fix prompt for the Generator based on Verifier feedback.

    Returns a natural language prompt that the Generator can use to
    produce improved Blender code.
    """
    issues = "\n".join(f"- {i}" for i in feedback.get("issues", []))
    suggestions = "\n".join(f"- {s}" for s in feedback.get("fix_suggestions", []))

    investigation_text = ""
    if investigation_results:
        investigation_text = f"""
═══ SCENE INVESTIGATION FINDINGS ═══
Scene state: {investigation_results.get('scene_state', 'N/A')}
Render settings: {investigation_results.get('render_settings', 'N/A')}
Objects: {investigation_results.get('objects_found', 'N/A')}
"""

    return f"""\
IMPROVED RENDER REQUEST (based on Verifier review)

═══ ORIGINAL PROMPT ═══
{prompt}

═══ PREVIOUS CODE ═══
```python
{code}
```
{investigation_text}
═══ VERIFIER FEEDBACK ═══
Score: {feedback.get('quality_score', 0.5)} | {"PASS" if feedback.get('pass_fail') == 'pass' else 'FAIL'}

Issues found:
{issues}

Suggested fixes:
{suggestions}

Strengths to maintain:
{chr(10).join(f"- {s}" for s in feedback.get("strengths", []))}

Generate an improved Blender Python script that addresses ALL issues listed above.
Keep what worked well. Make specific, targeted changes.
"""


async def verify_and_suggest_fixes(
    prompt: str,
    video_url: str,
    code: str,
    blender_file_path: str = "",
    previous_feedback: Optional[list[dict]] = None,
) -> dict:
    """Run the full Verifier pipeline: investigate scene + review video + return structured feedback.

    This is the main entry point called by the Generator after each render iteration.
    """
    # Step 1: Investigate the scene (if .blend file available)
    investigation_results = await run_investigation(blender_file_path, prompt)

    # Step 2: Have the VLM review the rendered video
    feedback = await review_render(
        prompt, video_url, code, blender_file_path, previous_feedback
    )

    # Step 3: Attach investigation findings to the feedback
    feedback["investigation_findings"] = investigation_results

    # Step 4: Generate a fix prompt for the Generator
    fix_prompt = await generate_fix_prompt(prompt, code, feedback, investigation_results)
    feedback["fix_prompt"] = fix_prompt

    return feedback
