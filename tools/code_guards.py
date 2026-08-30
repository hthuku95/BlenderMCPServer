"""Deterministic pre-render guards applied to every generated script,
regardless of which codegen path produced it (standard / ReAct / fix-loop)."""

import re


def guard_orphan_root(code: str) -> str:
    """Neutralize references to `root` when the Sketchfab helper was never
    called. Models copy the taught usage lines without the assignment ->
    NameError at render time. Orphan lines are commented out so the rest of
    the script renders."""
    calls_helper = bool(re.search(r"\bload_model_from_url\s*\(", code))
    assigns_root = bool(re.search(r"^\s*root\s*=", code, flags=re.M)) or \
        bool(re.search(r"\broot\s*=\s*load_model_from_url", code))
    if calls_helper and assigns_root:
        return code
    if "root" not in code:
        return code
    out = []
    for line in code.splitlines():
        if re.search(r"\broot\b", line) and not line.lstrip().startswith("#"):
            out.append("# [orphan-root removed] " + line)
        else:
            out.append(line)
    return "\n".join(out)


def guard_empty_objects(code: str) -> str:
    """Fix the recurring invalid pattern `bpy.data.objects.new(name, type='EMPTY')`.

    bpy.data.objects.new's second parameter is object_data (a mesh/curve/camera
    datablock); empties take None. `type='EMPTY'` raises
    `TypeError: BlendDataObjects.new(): required parameter "object_data" not
    specified` at render time (observed repeatedly in failed jobs, Aug 2026)."""
    return re.sub(
        r"bpy\.data\.objects\.new\(([^()]*?),\s*type\s*=\s*['\"]EMPTY['\"]\s*\)",
        r"bpy.data.objects.new(\1, None)",
        code,
    )


def apply_all(script_content: str) -> str:
    script_content = guard_orphan_root(script_content)
    script_content = guard_empty_objects(script_content)
    return script_content
