"""
matrix_transform_scene.py — Linear algebra: matrix transformations with NumberPlane.

Args (via CHART_SCENE_ARGS env var, JSON):
    matrix:       [[a,b],[c,d]] — 2x2 transformation matrix (default: rotation 45°)
    title:        str
    duration:     float
    show_vectors: bool  — show sample vectors being transformed
    show_det:     bool  — show determinant annotation
    show_eigen:   bool  — show eigenvectors (if applicable)
    style:        "dark" | "grid"
    mode:         "transform" | "compare" | "step_by_step"
"""
import json
import math
import os
from manim import *
import numpy as np

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))


class MatrixTransformScene(LinearTransformationScene):
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            show_basis_vectors=True,
            **kwargs
        )

    def construct(self):
        matrix_raw   = _ARGS.get("matrix", [[0, -1], [1, 0]])  # 90° rotation
        title_text   = _ARGS.get("title", "Linear Transformation")
        duration     = float(_ARGS.get("duration", 12.0))
        show_vectors = bool(_ARGS.get("show_vectors", True))
        show_det     = bool(_ARGS.get("show_det", True))
        mode         = _ARGS.get("mode", "transform")

        # Ensure 2×2
        mat = np.array(matrix_raw, dtype=float)
        if mat.shape != (2, 2):
            mat = np.array([[0, -1], [1, 0]])

        title = Text(title_text, font_size=26, color=WHITE, weight=BOLD)
        title.to_corner(UL, buff=0.2)
        self.add_foreground_mobject(title)

        # Show the matrix
        mat_mob = Matrix(matrix_raw, h_buff=0.8)
        mat_mob.scale(0.7)
        mat_mob.to_corner(UR, buff=0.3)
        self.add_foreground_mobject(mat_mob)

        # Determinant annotation
        if show_det:
            det = mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]
            det_text = Text(f"det = {det:.2f}", font_size=20, color=YELLOW)
            det_text.next_to(mat_mob, DOWN, buff=0.1)
            self.add_foreground_mobject(det_text)

        # Additional sample vectors
        vectors = []
        if show_vectors:
            sample_vecs = [[1, 1], [2, 0], [0, 2], [-1, 1]]
            for v in sample_vecs:
                vec = self.add_vector(v, color=ORANGE, animate=False)
                vectors.append((v, vec))

        # Apply transformation
        self.apply_matrix(mat.tolist(), run_time=min(3.0, duration * 0.5))

        self.wait(max(0.5, duration - self.renderer.time))
