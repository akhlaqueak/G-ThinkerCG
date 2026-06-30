from math import hypot
from pathlib import Path

import networkx as nx
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/akahmad/Documents/G-ThinkerCG")
OUT_DIR = ROOT / "output/pdf"
TMP_DIR = ROOT / "tmp/pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "gmatch_queries_q1_to_q6_2_5_6_7_8_9.pdf"
PREVIEW_PATH = TMP_DIR / "gmatch_queries_q1_to_q6_2_5_6_7_8_9_preview.png"


QUERIES = [
    (
        "Q1",
        "pattern 2 - chordal square",
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)},
    ),
    (
        "Q2",
        "pattern 5 - chordal house",
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)],
        {0: (0.55, 1.55), 1: (0.0, 0.8), 2: (1.0, 0.8), 3: (0.35, 0.0), 4: (1.15, 0.0)},
    ),
    (
        "Q3",
        "pattern 6 - chordal roof",
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)],
        {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.2, 0.0), 3: (0.8, 0.0), 4: (0.5, -1.0)},
    ),
    (
        "Q4",
        "pattern 7 - three triangles",
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4)],
        {0: (0.8, 0.5), 1: (0.0, 1.0), 2: (0.0, 0.0), 3: (1.8, 1.0), 4: (1.8, 0.0)},
    ),
    (
        "Q5",
        "pattern 8 - solar square",
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)],
        {0: (0.5, 0.5), 1: (0.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 0.0), 4: (0.0, 0.0)},
    ),
    (
        "Q6",
        "pattern 9 - near 5 clique",
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)],
        {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.0, 0.0), 3: (1.0, 0.0), 4: (0.5, -0.9)},
    ),
]


PAGE_W, PAGE_H = 1650, 1275
MARGIN_X, MARGIN_Y = 80, 120
TITLE_Y = 42
FOOTER_Y = PAGE_H - 55
COLS, ROWS = 3, 2
CELL_W = (PAGE_W - 2 * MARGIN_X) // COLS
CELL_H = (PAGE_H - 2 * MARGIN_Y) // ROWS
NODE_R = 24


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
LABEL_FONT = load_font(24, bold=True)
SUB_FONT = load_font(18, bold=False)
NODE_FONT = load_font(22, bold=True)
FOOTER_FONT = load_font(17, bold=False)


def transform_positions(pos, box):
    left, top, width, height = box
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    usable_w = width * 0.72
    usable_h = height * 0.52
    scale = min(usable_w / span_x, usable_h / span_y)
    cx = left + width / 2
    cy = top + height / 2 + 25

    mapped = {}
    for node, (x, y) in pos.items():
        px = cx + (x - (min_x + max_x) / 2) * scale
        py = cy - (y - (min_y + max_y) / 2) * scale
        mapped[node] = (px, py)
    return mapped


def draw_centered_text(draw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def draw_graph(draw, qname, subtitle, edges, pos, cell_box):
    left, top, width, height = cell_box
    draw.rounded_rectangle(
        [left + 10, top + 10, left + width - 10, top + height - 10],
        radius=20,
        fill="#f8fafc",
        outline="#cbd5e1",
        width=2,
    )
    draw_centered_text(draw, (left + width / 2, top + 32), qname, LABEL_FONT, "#0f172a")
    draw_centered_text(draw, (left + width / 2, top + 60), subtitle, SUB_FONT, "#475569")

    mapped = transform_positions(pos, cell_box)
    g = nx.Graph()
    g.add_edges_from(edges)

    for u, v in g.edges():
        x1, y1 = mapped[u]
        x2, y2 = mapped[v]
        dist = max(hypot(x2 - x1, y2 - y1), 1e-6)
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        p1 = (x1 + dx * NODE_R, y1 + dy * NODE_R)
        p2 = (x2 - dx * NODE_R, y2 - dy * NODE_R)
        draw.line([p1, p2], fill="#475569", width=7)

    for node, (x, y) in mapped.items():
        draw.ellipse([x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R], fill="#fffef5", outline="#0f172a", width=3)
        draw_centered_text(draw, (x, y - 1), str(node), NODE_FONT, "#0f172a")


def main():
    image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(image)

    draw_centered_text(draw, (PAGE_W / 2, TITLE_Y), "Query Graphs: Q1 to Q6", TITLE_FONT, "#0f172a")
    draw_centered_text(draw, (PAGE_W / 2, FOOTER_Y), "Based on preset patterns 2, 5, 6, 7, 8, 9 from common/graph.h", FOOTER_FONT, "#475569")

    for idx, (qname, subtitle, edges, pos) in enumerate(QUERIES):
        row = idx // COLS
        col = idx % COLS
        cell_left = MARGIN_X + col * CELL_W
        cell_top = MARGIN_Y + row * CELL_H
        draw_graph(draw, qname, subtitle, edges, pos, (cell_left, cell_top, CELL_W, CELL_H))

    image.save(PREVIEW_PATH, "PNG")
    image.save(PDF_PATH, "PDF", resolution=150.0)
    print(PDF_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
