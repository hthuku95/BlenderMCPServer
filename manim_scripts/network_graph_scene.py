"""
network_graph_scene.py — Manim animated network / knowledge graph.

Args (via CHART_SCENE_ARGS env var, JSON):
    nodes:   list of {"id": str, "label": str, "color": str (optional), "size": float (optional)}
    edges:   list of {"from": str, "to": str, "label": str (optional), "directed": bool (optional)}
    title:   str
    layout:  "circular" | "spring" | "radial"
    duration: float
    style:   "dark" | "neon"
"""
import json
import math
import os
import random
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))

_DEFAULT_NODES = [
    {"id": "A", "label": "Machine Learning", "color": "BLUE"},
    {"id": "B", "label": "Deep Learning",    "color": "GREEN"},
    {"id": "C", "label": "NLP",              "color": "ORANGE"},
    {"id": "D", "label": "Computer Vision",  "color": "RED"},
    {"id": "E", "label": "Reinforcement",    "color": "PURPLE"},
    {"id": "F", "label": "Data Science",     "color": "YELLOW"},
]
_DEFAULT_EDGES = [
    {"from": "A", "to": "B"}, {"from": "A", "to": "C"},
    {"from": "A", "to": "D"}, {"from": "A", "to": "E"},
    {"from": "A", "to": "F"}, {"from": "B", "to": "C"},
    {"from": "B", "to": "D"},
]

_CMAP = {
    "BLUE": BLUE_B, "RED": RED_B, "GREEN": GREEN_B, "YELLOW": YELLOW,
    "ORANGE": ORANGE, "PURPLE": PURPLE_B, "TEAL": TEAL_B, "GOLD": GOLD,
    "WHITE": WHITE, "PINK": PINK,
}


def _circular_layout(nodes, radius=3.0):
    n = len(nodes)
    positions = {}
    for i, nd in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        positions[nd["id"]] = np.array([radius * math.cos(angle),
                                         radius * math.sin(angle), 0])
    return positions


def _radial_layout(nodes, edges):
    """Hub-and-spoke: most-connected node at center."""
    from collections import Counter
    degree = Counter()
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]]   += 1

    if not nodes:
        return {}
    hub_id = max(nodes, key=lambda n: degree.get(n["id"], 0))["id"]
    spokes = [n for n in nodes if n["id"] != hub_id]

    positions = {hub_id: ORIGIN}
    r = 3.0
    for i, nd in enumerate(spokes):
        angle = 2 * math.pi * i / max(len(spokes), 1)
        positions[nd["id"]] = np.array([r * math.cos(angle), r * math.sin(angle), 0])
    return positions


class NetworkGraphScene(Scene):
    def construct(self):
        nodes_raw  = _ARGS.get("nodes", _DEFAULT_NODES)
        edges_raw  = _ARGS.get("edges", _DEFAULT_EDGES)
        title_text = _ARGS.get("title", "Network Graph")
        layout     = _ARGS.get("layout", "radial")
        duration   = float(_ARGS.get("duration", 12.0))
        style      = _ARGS.get("style", "dark")

        if style == "neon":
            self.camera.background_color = "#000011"
            edge_col = "#333366"
        else:
            self.camera.background_color = "#111122"
            edge_col = DARK_GRAY

        # Compute positions
        if layout == "circular" or layout == "spring":
            positions = _circular_layout(nodes_raw, radius=2.8)
        else:
            positions = _radial_layout(nodes_raw, edges_raw)

        # Scale to fit
        if positions:
            max_r = max(np.linalg.norm(p) for p in positions.values()) or 1
            if max_r > 3.2:
                scale = 3.2 / max_r
                positions = {k: v * scale for k, v in positions.items()}

        # Build node mobs
        node_mobs = {}
        for nd in nodes_raw:
            nid   = nd["id"]
            label = nd.get("label", nid)
            col   = _CMAP.get(nd.get("color", "BLUE"), BLUE_B)
            size  = float(nd.get("size", 0.35))
            pos   = positions.get(nid, ORIGIN)

            circle = Circle(radius=size, fill_color=col, fill_opacity=0.9,
                            stroke_color=WHITE, stroke_width=1.5)
            circle.move_to(pos)

            txt = Text(label, font_size=15, color=WHITE)
            if txt.width > size * 2.2:
                txt.scale(size * 2.2 / txt.width)

            # Label below the node if it doesn't fit inside
            if txt.width > size * 1.6:
                txt.next_to(circle, DOWN, buff=0.08)
            else:
                txt.move_to(pos)

            grp = VGroup(circle, txt)
            node_mobs[nid] = grp

        # Build edge mobs
        edge_mobs = []
        edge_label_mobs = []
        for e in edges_raw:
            src, dst = e.get("from"), e.get("to")
            if src not in positions or dst not in positions:
                continue
            directed = e.get("directed", False)
            start_p  = positions[src]
            end_p    = positions[dst]

            if directed:
                line = Arrow(start_p, end_p, buff=0.38, color=edge_col,
                             stroke_width=2, max_tip_length_to_length_ratio=0.1)
            else:
                line = Line(start_p, end_p, color=edge_col, stroke_width=2)

            edge_mobs.append(line)
            lbl = e.get("label", "")
            if lbl:
                mid = (start_p + end_p) / 2 + UP * 0.15
                lt  = Text(lbl, font_size=12, color=GRAY_B).move_to(mid)
                edge_label_mobs.append(lt)

        # Title
        title_mob = Text(title_text, font_size=28, color=WHITE, weight=BOLD)
        title_mob.to_edge(UP, buff=0.2)

        # --- Animate ---
        self.play(Write(title_mob), run_time=0.7)

        # Draw edges first (behind nodes)
        if edge_mobs:
            self.play(*[Create(e) for e in edge_mobs], run_time=1.5)
        if edge_label_mobs:
            self.play(*[FadeIn(l) for l in edge_label_mobs], run_time=0.5)

        # Reveal nodes with a wave/stagger
        node_list = list(node_mobs.values())
        anim_per_node = max(0.2, (duration - 4.0) / max(len(node_list), 1))
        for grp in node_list:
            self.play(GrowFromCenter(grp), run_time=anim_per_node)

        # Pulse animation on hub (most connected node)
        from collections import Counter
        degree = Counter()
        for e in edges_raw:
            degree[e.get("from")] += 1
            degree[e.get("to")]   += 1
        if degree and nodes_raw:
            hub = max(nodes_raw, key=lambda n: degree.get(n["id"], 0))["id"]
            if hub in node_mobs:
                hub_mob = node_mobs[hub]
                self.play(hub_mob.animate.scale(1.2), run_time=0.4)
                self.play(hub_mob.animate.scale(1 / 1.2), run_time=0.4)

        self.wait(max(0.5, duration - self.renderer.time))
