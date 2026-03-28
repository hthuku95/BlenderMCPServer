"""
Standalone LaTeX → SVG compiler.

Uses the system TeX Live distribution (latex + dvisvgm).
Does NOT require Manim — this is the entry point for Option A.

The SVG output is then passed to Blender via bpy.ops.import_curve.svg().
"""
import asyncio
import os
from pathlib import Path

_TEMPLATE = r"""\documentclass[preview]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\begin{document}
%s
\end{document}
"""


async def latex_to_svg(latex_expression: str, output_path: str | None = None) -> str:
    """
    Compile a LaTeX expression to a clean vector SVG.

    Args:
        latex_expression: Raw LaTeX string. Bare expressions (no $ delimiters) are
                          wrapped in display math automatically.
        output_path:      Optional destination path for the SVG file.

    Returns: Absolute path to the generated SVG file.
    Raises:  RuntimeError on compilation failure.
    """
    import tempfile
    import shutil

    # Wrap bare expressions in display math
    expr = latex_expression.strip()
    if not expr.startswith(r"\begin") and "$" not in expr and r"\[" not in expr:
        body = rf"\[ {expr} \]"
    else:
        body = expr

    tex_source = _TEMPLATE % body

    with tempfile.TemporaryDirectory(prefix="latex_compile_") as work_dir:
        tex_file = Path(work_dir) / "eq.tex"
        tex_file.write_text(tex_source)

        # Step 1: latex → DVI
        latex_proc = await asyncio.create_subprocess_exec(
            "latex",
            "-interaction=nonstopmode",
            "-output-directory", work_dir,
            str(tex_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        stdout_b, stderr_b = await asyncio.wait_for(latex_proc.communicate(), timeout=30)
        if latex_proc.returncode != 0:
            # LaTeX writes errors to stdout (the log), not stderr
            log = (stdout_b.decode()[-2000:] or stderr_b.decode()[-1000:]).strip()
            raise RuntimeError(f"latex failed:\n{log}")

        dvi_file = Path(work_dir) / "eq.dvi"
        if not dvi_file.exists():
            raise RuntimeError("latex ran but produced no .dvi file")

        # Step 2: dvisvgm → SVG
        stable_svg = output_path or f"/tmp/eq_{os.getpid()}_{id(latex_expression)}.svg"
        svg_proc = await asyncio.create_subprocess_exec(
            "dvisvgm",
            "--no-fonts",        # embed glyphs as paths (no font deps)
            "--output", stable_svg,
            str(dvi_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await asyncio.wait_for(svg_proc.communicate(), timeout=30)
        if svg_proc.returncode != 0:
            raise RuntimeError(f"dvisvgm failed:\n{stderr_b.decode()[-500:]}")

        if not os.path.exists(stable_svg):
            raise RuntimeError("dvisvgm ran but produced no SVG file")

        return stable_svg


async def validate_latex(expression: str) -> tuple[bool, str]:
    """
    Quick syntax-check a LaTeX expression.

    Returns: (is_valid: bool, error_message: str)
    """
    try:
        path = await latex_to_svg(expression)
        os.unlink(path)
        return True, ""
    except RuntimeError as e:
        return False, str(e)
