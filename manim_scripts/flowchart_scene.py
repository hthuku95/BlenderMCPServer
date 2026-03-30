"""
flowchart_scene.py — Manim animated flowchart.

Args (via CHART_SCENE_ARGS env var, JSON):
    nodes:    list of {"id": str, "label": str, "type": "process"|"decision"|"start"|"end"}
    edges:    list of {"from": str, "to": str, "label": str (optional)}
    title:    str
    duration: float
    style:    "dark" | "light" | "blue"
"""
import json
import os
from manim import *

_ARGS = json.loads(os.environ.get("CHART_SCENE_ARGS", "{}"))


def _parse_color(name: str, default) -> color.ManimColor:
    mapping = {
        "BLUE": BLUE, "RED": RED, "GREEN": GREEN, "YELLOW": YELLOW,
        "WHITE": WHITE, "GRAY": GRAY, "ORANGE": ORANGE, "PURPLE": PURPLE,
        "TEAL": TEAL, "GOLD": GOLD, "PINK": PINK, "MAROON": MAROON,
    }
    return mapping.get(name.upper(), default)


class FlowchartScene(Scene):
    def construct(self):
        nodes_raw = _ARGS.get("nodes", [
            {"id": "start",   "label": "Start",          "type": "start"},
            {"id": "step1",   "label": "Gather Data",    "type": "process"},
            {"id": "decide",  "label": "Data Valid?",    "type": "decision"},
            {"id": "step2",   "label": "Process Data",   "type": "process"},
            {"id": "error",   "label": "Handle Error",   "type": "process"},
            {"id": "end",     "label": "End",            "type": "end"},
        ])
        edges_raw = _ARGS.get("edges", [
            {"from": "start",  "to": "step1"},
            {"from": "step1",  "to": "decide"},
            {"from": "decide", "to": "step2",  "label": "Yes"},
            {"from": "decide", "to": "error",  "label": "No"},
            {"from": "step2",  "to": "end"},
            {"from": "error",  "to": "end"},
        ])
        title_text = _ARGS.get("title", "Process Flowchart")
        duration   = float(_ARGS.get("duration", 12.0))
        style      = _ARGS.get("style", "dark")

        # Style
        if style == "light":
            bg_col, box_col, text_col, arrow_col = WHITE, BLUE_D, BLACK, DARK_GRAY
        elif style == "blue":
            bg_col, box_col, text_col, arrow_col = "#0a1628", BLUE_D, WHITE, BLUE_B
        else:  # dark
            bg_col, box_col, text_col, arrow_col = "#1a1a2e", BLUE_D, WHITE, BLUE_B

        self.camera.background_color = bg_col

        # Build index
        node_index = {n["id"]: n for n in nodes_raw}

        # Auto-layout: BFS from start node, assign row depths
        from collections import deque, defaultdict
        adj = defaultdict(list)
        for e in edges_raw:
            adj[e["from"]].append(e["to"])

        depth = {}
        start_id = nodes_raw[0]["id"]
        q = deque([(start_id, 0)])
        while q:
            nid, d = q.popleft()
            if nid in depth:
                continue
            depth[nid] = d
            for child in adj[nid]:
                if child not in depth:
                    q.append((child, d + 1))
        # Any unvisited nodes
        for n in nodes_raw:
            if n["id"] not in depth:
                depth[n["id"]] = max(depth.values(), default=0) + 1

        # Group by depth
        by_depth = defaultdict(list)
        for nid, d in depth.items():
            by_depth[d].append(nid)
        max_depth = max(by_depth.keys(), default=0)

        # Positions: Y = -depth * 1.4, X = spread siblings
        NODE_W, NODE_H = 2.4, 0.65
        V_SPACING, H_SPACING = 1.5, 2.8

        positions = {}
        for d, nids in by_depth.items():
            count = len(nids)
            for i, nid in enumerate(nids):
                x = (i - (count - 1) / 2) * H_SPACING
                y = -d * V_SPACING + (max_depth * V_SPACING / 2)
                positions[nid] = np.array([x, y, 0])

        # Build Manim mobjects for each node
        COLORS = {
            "start":    GREEN_D,
            "end":      RED_D,
            "process":  BLUE_D,
            "decision": GOLD_D,
        }
        STROKE = {
            "start": GREEN_B, "end": RED_B, "process": BLUE_B, "decision": GOLD_B,
        }

        node_mobs = {}
        node_centers = {}
        all_node_vg = VGroup()

        for n in nodes_raw:
            nid  = n["id"]
            ntype = n.get("type", "process")
            label = n.get("label", nid)
            pos   = positions.get(nid, ORIGIN)
            fill  = COLORS.get(ntype, BLUE_D)
            stroke = STROKE.get(ntype, BLUE_B)

            if ntype == "decision":
                # Diamond
                pts = [UP * 0.5, RIGHT * 1.2, DOWN * 0.5, LEFT * 1.2]
                shape = Polygon(*pts, fill_color=fill, fill_opacity=0.85,
                                stroke_color=stroke, stroke_width=2)
            elif ntype in ("start", "end"):
                shape = RoundedRectangle(corner_radius=0.32, width=NODE_W * 0.9,
                                         height=NODE_H, fill_color=fill,
                                         fill_opacity=0.9, stroke_color=stroke,
                                         stroke_width=2)
            else:
                shape = Rectangle(width=NODE_W, height=NODE_H, fill_color=fill,
                                  fill_opacity=0.85, stroke_color=stroke, stroke_width=2)

            shape.move_to(pos)
            txt = Text(label, font_size=18, color=WHITE if style != "light" else BLACK)
            txt.move_to(pos)
            if txt.width > shape.width * 0.85:
                txt.scale(shape.width * 0.85 / txt.width)

            grp = VGroup(shape, txt)
            node_mobs[nid] = grp
            node_centers[nid] = pos
            all_node_vg.add(grp)

        # Scale to fit screen
        if all_node_vg.width > 11:
            all_node_vg.scale(11 / all_node_vg.width)
        if all_node_vg.height > 6:
            all_node_vg.scale(6 / all_node_vg.height)
        all_node_vg.center()

        # Recompute centers after scaling/centering
        for nid in node_mobs:
            node_centers[nid] = node_mobs[nid].get_center()

        # Build arrows
        arrow_mobs = []
        edge_labels = []
        for e in edges_raw:
            src, dst = e.get("from"), e.get("to")
            if src not in node_centers or dst not in node_centers:
                continue
            start_pt = node_centers[src] + DOWN * 0.32
            end_pt   = node_centers[dst] + UP   * 0.32
            arr = Arrow(start_pt, end_pt, buff=0.1, color=arrow_col,
                        stroke_width=2.5, max_tip_length_to_length_ratio=0.12)
            arrow_mobs.append(arr)
            lbl = e.get("label", "")
            if lbl:
                mid = (start_pt + end_pt) / 2 + RIGHT * 0.25
                ltxt = Text(lbl, font_size=14, color=YELLOW)
                ltxt.move_to(mid)
                edge_labels.append(ltxt)

        # Title
        title_mob = Text(title_text, font_size=26, color=WHITE if style != "light" else BLACK,
                         weight=BOLD)
        title_mob.to_edge(UP, buff=0.15)

        # --- Animate ---
        anim_time_per_step = max(0.3, (duration - 2.0) / (len(nodes_raw) + len(arrow_mobs) + 1))

        self.play(Write(title_mob), run_time=0.8)
        self.wait(0.2)

        # Reveal nodes in BFS order
        for d in sorted(by_depth.keys()):
            row_mobs = [node_mobs[nid] for nid in by_depth[d] if nid in node_mobs]
            if row_mobs:
                self.play(*[GrowFromCenter(m) for m in row_mobs],
                          run_time=anim_time_per_step * 1.2)

        # Draw arrows one by one
        self.play(*[GrowArrow(a) for a in arrow_mobs], run_time=anim_time_per_step * 2)
        if edge_labels:
            self.play(*[FadeIn(l) for l in edge_labels], run_time=0.5)

        self.wait(max(0.5, duration - self.renderer.time))
