from math import hypot
from pathlib import Path

import networkx as nx
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/akahmad/Documents/G-ThinkerCG")
OUT_DIR = ROOT / "output/pdf"
TMP_DIR = ROOT / "tmp/pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "gmatch_presets_all.pdf"
PREVIEW_PATH = TMP_DIR / "gmatch_presets_all_preview_page1.png"


PRESETS = [
    ("P0 - triangle", [(0, 1), (1, 2), (2, 0)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.5, 0.0)}),
    ("P1 - square", [(0, 1), (1, 2), (2, 3), (3, 0)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)}),
    ("P2 - chordal square", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)}),
    ("P3 - 2 tails triangle", [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)], {0: (0.0, 0.9), 1: (1.0, 0.9), 2: (0.5, 0.0), 3: (1.25, -0.8), 4: (2.0, -1.5)}),
    ("P4 - house", [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)], {0: (0.5, 1.6), 1: (0.0, 0.8), 2: (1.0, 0.8), 3: (0.0, 0.0), 4: (1.0, 0.0)}),
    ("P5 - chordal house", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)], {0: (0.55, 1.55), 1: (0.0, 0.8), 2: (1.0, 0.8), 3: (0.35, 0.0), 4: (1.15, 0.0)}),
    ("P6 - chordal roof", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.2, 0.0), 3: (0.8, 0.0), 4: (0.5, -1.0)}),
    ("P7 - three triangles", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4)], {0: (0.8, 0.5), 1: (0.0, 1.0), 2: (0.0, 0.0), 3: (1.8, 1.0), 4: (1.8, 0.0)}),
    ("P8 - solar square", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)], {0: (0.5, 0.5), 1: (0.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 0.0), 4: (0.0, 0.0)}),
    ("P9 - near 5 clique", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.0, 0.0), 3: (1.0, 0.0), 4: (0.5, -0.9)}),
    ("P10 - four triangles", [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)], {0: (1.0, 0.8), 1: (0.0, 1.3), 2: (0.3, 0.2), 3: (1.0, -0.2), 4: (1.7, 0.2), 5: (2.0, 1.3)}),
    ("P11 - one in three triangles", [(0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5)], {0: (0.0, 0.8), 1: (1.0, 0.8), 2: (0.5, 0.0), 3: (-0.3, 1.7), 4: (1.3, 1.7), 5: (0.5, -1.0)}),
    ("P12 - near 6 clique", [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)], {0: (0.0, 1.1), 1: (1.0, 1.1), 2: (0.4, 0.3), 3: (1.6, 0.3), 4: (0.8, -0.5), 5: (1.8, -0.5)}),
    ("P13 - square on top", [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)], {0: (0.0, 1.3), 1: (1.0, 1.3), 2: (0.0, 0.3), 3: (1.0, 0.3), 4: (0.2, -0.8), 5: (0.8, -0.8)}),
    ("P14 - near 7 clique", [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (3, 6), (4, 5), (5, 6)], {0: (0.0, 1.4), 1: (1.0, 1.4), 2: (-0.2, 0.5), 3: (0.8, 0.5), 4: (1.8, 0.5), 5: (0.8, -0.5), 6: (1.8, -0.5)}),
    ("P15 - 5 clique on top", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6)], {0: (0.2, 1.4), 1: (1.0, 1.4), 2: (-0.2, 0.6), 3: (0.6, 0.6), 4: (1.4, 0.6), 5: (0.6, -0.4), 6: (1.4, -0.4)}),
    ("P16 - 5 cycle", [(0, 1), (1, 3), (3, 4), (4, 2), (2, 0)], {0: (0.0, 0.6), 1: (0.6, 1.2), 3: (1.4, 0.8), 4: (1.1, -0.1), 2: (0.1, -0.2)}),
    ("P17 - 6 cycle", [(0, 1), (1, 3), (3, 5), (5, 4), (4, 2), (2, 0)], {0: (0.0, 0.6), 1: (0.5, 1.2), 3: (1.4, 1.2), 5: (1.9, 0.6), 4: (1.4, 0.0), 2: (0.5, 0.0)}),
    ("P18 - hourglass", [(0, 1), (0, 2), (0, 4), (1, 2), (1, 5), (2, 3), (3, 4), (3, 5), (4, 5)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.5, 0.2), 3: (0.5, -0.6), 4: (0.0, -1.4), 5: (1.0, -1.4)}),
    ("P23 - 3 clique", [(0, 1), (1, 2), (2, 0)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.5, 0.0)}),
    ("P24 - 4 clique", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (0.0, 0.0), 3: (1.0, 0.0)}),
    ("P25 - 5 clique", [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (-0.1, 0.0), 3: (1.1, 0.0), 4: (0.5, -0.9)}),
    ("P26 - 6 clique", [(i, j) for i in range(6) for j in range(i + 1, 6)], {0: (0.5, 1.4), 1: (1.3, 1.0), 2: (1.3, 0.0), 3: (0.5, -0.4), 4: (-0.3, 0.0), 5: (-0.3, 1.0)}),
    ("P27 - 7 clique", [(i, j) for i in range(7) for j in range(i + 1, 7)], {0: (0.6, 1.5), 1: (1.4, 1.1), 2: (1.6, 0.3), 3: (1.0, -0.3), 4: (0.2, -0.3), 5: (-0.4, 0.3), 6: (-0.2, 1.1)}),
]


PAGE_W, PAGE_H = 1650, 1275
MARGIN_X, MARGIN_Y = 80, 120
TITLE_Y = 40
FOOTER_Y = PAGE_H - 50
COLS, ROWS = 3, 3
CELL_W = (PAGE_W - 2 * MARGIN_X) // COLS
CELL_H = (PAGE_H - 2 * MARGIN_Y) // ROWS
NODE_R = 22
PER_PAGE = COLS * ROWS


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
LABEL_FONT = load_font(20, bold=True)
NODE_FONT = load_font(20, bold=True)
FOOTER_FONT = load_font(16, bold=False)


def transform_positions(pos, box):
    left, top, width, height = box
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    usable_w = width * 0.68
    usable_h = height * 0.55
    scale = min(usable_w / span_x, usable_h / span_y)
    cx = left + width / 2
    cy = top + height / 2 + 14

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


def draw_graph(draw, title, edges, pos, cell_box):
    left, top, width, height = cell_box
    draw.rounded_rectangle(
        [left + 8, top + 8, left + width - 8, top + height - 8],
        radius=18,
        fill="#f8fafc",
        outline="#cbd5e1",
        width=2,
    )
    draw_centered_text(draw, (left + width / 2, top + 30), title, LABEL_FONT, "#0f172a")

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
        draw.line([p1, p2], fill="#475569", width=6)

    for node, (x, y) in mapped.items():
        draw.ellipse([x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R], fill="#fffef5", outline="#0f172a", width=3)
        draw_centered_text(draw, (x, y - 1), str(node), NODE_FONT, "#0f172a")


def render_page(page_index, presets):
    image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    draw_centered_text(draw, (PAGE_W / 2, TITLE_Y), f"Preset Query Graphs (page {page_index})", TITLE_FONT, "#0f172a")
    draw_centered_text(draw, (PAGE_W / 2, FOOTER_Y), "Implemented presets from common/graph.h; P19-P22 are placeholders", FOOTER_FONT, "#475569")

    for idx, (title, edges, pos) in enumerate(presets):
        row = idx // COLS
        col = idx % COLS
        cell_left = MARGIN_X + col * CELL_W
        cell_top = MARGIN_Y + row * CELL_H
        draw_graph(draw, title, edges, pos, (cell_left, cell_top, CELL_W, CELL_H))
    return image


def main():
    pages = []
    for start in range(0, len(PRESETS), PER_PAGE):
        page_num = start // PER_PAGE + 1
        page = render_page(page_num, PRESETS[start:start + PER_PAGE])
        pages.append(page)

    pages[0].save(PDF_PATH, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    pages[0].save(PREVIEW_PATH, "PNG")
    print(PDF_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
