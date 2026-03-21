"""
SVG Export — generates a simple SVG screenshot mockup from a design spec.

Used upstream of device_mockup.py to produce a screenshot image when the caller
doesn't supply one directly.  The output PNG is a flat render of the SVG via
cairosvg (or Inkscape headless as fallback), suitable as the screenshot_path
argument for device_mockup.py.

Functions:
    build_browser_svg(url_text, bg_color, content_blocks) -> str (SVG text)
    build_app_svg(app_name, bg_color, content_blocks) -> str (SVG text)
    svg_to_png(svg_text, output_path, width, height) -> str (output_path)
    screenshot_from_spec(spec: dict, output_path: str) -> str (png path)

spec dict schema:
    {
      "type": "browser" | "app",
      "url": str,                           # shown in browser address bar
      "app_name": str,                      # shown in app header
      "bg_color": "#hexcolor",
      "accent_color": "#hexcolor",
      "title": str,
      "body": str,                          # plain text body paragraph
      "width": int,                         # default 1170
      "height": int,                        # default 2532  (portrait)
    }
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# SVG builder helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb_float(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r / 255, g / 255, b / 255


def _is_light(hex_color: str) -> bool:
    r, g, b = _hex_to_rgb_float(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 0.5


def build_browser_svg(
    url_text: str = "https://example.com",
    bg_color: str = "#ffffff",
    accent_color: str = "#0070f3",
    title: str = "My App",
    body: str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    width: int = 1170,
    height: int = 2532,
) -> str:
    text_color = "#111111" if _is_light(bg_color) else "#f0f0f0"
    chrome_bg = "#1e1e1e"
    bar_height = 80

    # Wrap body text manually at ~60 chars for SVG
    words = body.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > 60:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)

    body_svg = "\n".join(
        f'      <tspan x="60" dy="{45 if i else 0}">{ln}</tspan>'
        for i, ln in enumerate(lines)
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{bg_color}" rx="20"/>

  <!-- Browser chrome bar -->
  <rect width="{width}" height="{bar_height}" fill="{chrome_bg}" rx="0"/>
  <!-- Traffic lights -->
  <circle cx="40" cy="{bar_height//2}" r="12" fill="#ff5f57"/>
  <circle cx="80" cy="{bar_height//2}" r="12" fill="#febc2e"/>
  <circle cx="120" cy="{bar_height//2}" r="12" fill="#28c840"/>
  <!-- Address bar -->
  <rect x="160" y="18" width="{width - 220}" height="44" fill="#2a2a2a" rx="8"/>
  <text x="{width // 2}" y="47" text-anchor="middle" font-family="sans-serif"
        font-size="22" fill="#aaaaaa">{url_text}</text>

  <!-- Page title -->
  <text x="60" y="{bar_height + 80}" font-family="sans-serif" font-size="52"
        font-weight="bold" fill="{text_color}">{title}</text>

  <!-- Accent divider -->
  <rect x="60" y="{bar_height + 100}" width="100" height="6" fill="{accent_color}" rx="3"/>

  <!-- Body text -->
  <text x="60" y="{bar_height + 170}" font-family="sans-serif" font-size="34"
        fill="{text_color}" opacity="0.75">
{body_svg}
  </text>

  <!-- Card placeholder -->
  <rect x="60" y="{bar_height + 400}" width="{width - 120}" height="320"
        fill="{accent_color}" opacity="0.12" rx="16"/>
  <rect x="80" y="{bar_height + 430}" width="200" height="30"
        fill="{accent_color}" opacity="0.4" rx="4"/>
  <rect x="80" y="{bar_height + 480}" width="{width - 200}" height="20"
        fill="{text_color}" opacity="0.12" rx="4"/>
  <rect x="80" y="{bar_height + 515}" width="{width - 300}" height="20"
        fill="{text_color}" opacity="0.12" rx="4"/>
  <rect x="80" y="{bar_height + 550}" width="{width - 250}" height="20"
        fill="{text_color}" opacity="0.12" rx="4"/>

  <!-- CTA button -->
  <rect x="60" y="{bar_height + 780}" width="280" height="80"
        fill="{accent_color}" rx="12"/>
  <text x="200" y="{bar_height + 830}" text-anchor="middle"
        font-family="sans-serif" font-size="32" font-weight="bold" fill="#ffffff">Get Started</text>
</svg>
"""


def build_app_svg(
    app_name: str = "MyApp",
    bg_color: str = "#0a0a0a",
    accent_color: str = "#6c63ff",
    title: str = "Dashboard",
    body: str = "Your analytics and reports at a glance.",
    width: int = 1170,
    height: int = 2532,
) -> str:
    text_color = "#111111" if _is_light(bg_color) else "#f0f0f0"
    status_bar_h = 54
    nav_h = 120

    words = body.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > 52:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)

    body_svg = "\n".join(
        f'      <tspan x="60" dy="{42 if i else 0}">{ln}</tspan>'
        for i, ln in enumerate(lines)
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{bg_color}" rx="40"/>

  <!-- Status bar -->
  <rect width="{width}" height="{status_bar_h}" fill="{bg_color}" rx="0"/>
  <text x="{width // 2}" y="38" text-anchor="middle" font-family="sans-serif"
        font-size="26" font-weight="600" fill="{text_color}" opacity="0.6">9:41</text>

  <!-- Navigation bar -->
  <rect y="{status_bar_h}" width="{width}" height="{nav_h}" fill="{bg_color}"/>
  <text x="60" y="{status_bar_h + 76}" font-family="sans-serif" font-size="44"
        font-weight="bold" fill="{text_color}">{app_name}</text>
  <!-- Hamburger icon area -->
  <rect x="{width - 120}" y="{status_bar_h + 34}" width="60" height="8"
        fill="{text_color}" opacity="0.6" rx="4"/>
  <rect x="{width - 120}" y="{status_bar_h + 52}" width="40" height="8"
        fill="{text_color}" opacity="0.4" rx="4"/>
  <rect x="{width - 120}" y="{status_bar_h + 70}" width="60" height="8"
        fill="{text_color}" opacity="0.6" rx="4"/>

  <!-- Section title -->
  <text x="60" y="{status_bar_h + nav_h + 70}" font-family="sans-serif" font-size="48"
        font-weight="bold" fill="{text_color}">{title}</text>

  <!-- Accent pill -->
  <rect x="60" y="{status_bar_h + nav_h + 88}" width="80" height="8"
        fill="{accent_color}" rx="4"/>

  <!-- Body -->
  <text x="60" y="{status_bar_h + nav_h + 150}" font-family="sans-serif"
        font-size="32" fill="{text_color}" opacity="0.65">
{body_svg}
  </text>

  <!-- Stats row -->
  {_stat_card(60,  status_bar_h + nav_h + 340, 320, 200, accent_color, "1.2K", "Users",    text_color)}
  {_stat_card(420, status_bar_h + nav_h + 340, 320, 200, accent_color, "98%", "Uptime",   text_color)}
  {_stat_card(780, status_bar_h + nav_h + 340, 320, 200, accent_color, "$4.1K", "Revenue", text_color)}

  <!-- Activity list -->
  {_list_item(60, status_bar_h + nav_h + 620, width - 120, accent_color, "New sign-up",   "2 min ago",  text_color)}
  {_list_item(60, status_bar_h + nav_h + 720, width - 120, accent_color, "Payment received", "15 min ago", text_color)}
  {_list_item(60, status_bar_h + nav_h + 820, width - 120, accent_color, "Report generated", "1 hour ago", text_color)}

  <!-- Bottom tab bar -->
  <rect y="{height - 140}" width="{width}" height="140" fill="{bg_color}" opacity="0.95"/>
  <rect y="{height - 142}" width="{width}" height="2" fill="{text_color}" opacity="0.08"/>
  {_tab_icon(width//5 * 1, height - 80, "⊞", "Home",     accent_color, text_color, True)}
  {_tab_icon(width//5 * 2, height - 80, "📊", "Stats",    text_color,   text_color, False)}
  {_tab_icon(width//5 * 3, height - 80, "⊕", "Add",      text_color,   text_color, False)}
  {_tab_icon(width//5 * 4, height - 80, "✉", "Messages", text_color,   text_color, False)}
</svg>
"""


def _stat_card(x, y, w, h, accent, value, label, text_color):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{accent}" opacity="0.15" rx="16"/>'
        f'<text x="{x + w//2}" y="{y + h//2 - 10}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="52" font-weight="bold" fill="{text_color}">{value}</text>'
        f'<text x="{x + w//2}" y="{y + h//2 + 40}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="26" fill="{text_color}" opacity="0.55">{label}</text>'
    )


def _list_item(x, y, w, accent, title, meta, text_color):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="80" fill="{text_color}" opacity="0.04" rx="12"/>'
        f'<rect x="{x + 20}" y="{y + 25}" width="30" height="30" fill="{accent}" rx="15"/>'
        f'<text x="{x + 70}" y="{y + 46}" font-family="sans-serif" font-size="30" '
        f'fill="{text_color}">{title}</text>'
        f'<text x="{x + w - 20}" y="{y + 46}" text-anchor="end" font-family="sans-serif" '
        f'font-size="24" fill="{text_color}" opacity="0.4">{meta}</text>'
    )


def _tab_icon(x, y, icon, label, color, text_color, active):
    opacity = "1.0" if active else "0.45"
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-size="36" fill="{color}" '
        f'opacity="{opacity}">{icon}</text>'
        f'<text x="{x}" y="{y + 32}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="22" fill="{color}" opacity="{opacity}">{label}</text>'
    )


# ---------------------------------------------------------------------------
# Rasteriser — SVG → PNG
# ---------------------------------------------------------------------------

def svg_to_png(
    svg_text: str,
    output_path: str,
    width: int = 1170,
    height: int = 2532,
) -> str:
    """
    Convert SVG text to a PNG file.

    Tries cairosvg first (pip install cairosvg), then falls back to
    Inkscape headless (inkscape --export-png).  If neither is available
    writes a 1×1 placeholder and logs a warning.
    """
    # Try cairosvg
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(
            bytestring=svg_text.encode(),
            write_to=output_path,
            output_width=width,
            output_height=height,
        )
        return output_path
    except ImportError:
        pass

    # Try Inkscape
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
        f.write(svg_text)
        svg_tmp = f.name

    try:
        result = subprocess.run(
            [
                "inkscape",
                svg_tmp,
                f"--export-filename={output_path}",
                f"--export-width={width}",
                f"--export-height={height}",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.unlink(svg_tmp)
        except OSError:
            pass

    # Last resort — write a minimal 1×1 white PNG
    import struct, zlib
    def _png1x1(path):
        def chunk(t, d):
            c = zlib.crc32(t + d) & 0xFFFFFFFF
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", c)
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\xFF\xFF\xFF")
        data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(data)

    import warnings
    warnings.warn(
        f"svg_to_png: neither cairosvg nor Inkscape available — writing 1×1 placeholder to {output_path}"
    )
    _png1x1(output_path)
    return output_path


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def screenshot_from_spec(spec: dict, output_path: str) -> str:
    """
    Build a PNG screenshot from a design spec dict.

    Returns the path to the written PNG file (same as output_path).
    """
    svg_type  = spec.get("type", "browser")
    bg        = spec.get("bg_color", "#ffffff")
    accent    = spec.get("accent_color", "#0070f3")
    title     = spec.get("title", "My App")
    body      = spec.get("body", "")
    url       = spec.get("url", "https://example.com")
    app_name  = spec.get("app_name", "App")
    width     = int(spec.get("width", 1170))
    height    = int(spec.get("height", 2532))

    if svg_type == "app":
        svg = build_app_svg(app_name, bg, accent, title, body, width, height)
    else:
        svg = build_browser_svg(url, bg, accent, title, body, width, height)

    return svg_to_png(svg, output_path, width, height)
