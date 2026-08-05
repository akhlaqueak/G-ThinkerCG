#!/usr/bin/env python3
"""Plot a log-log degree-frequency histogram from a G-Thinker CSR .bin graph."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a G-Thinker CSR .bin graph and plot the number of vertices "
            "having each nonzero degree. Only the CSR row offsets are read; "
            "the adjacency array is not loaded."
        )
    )
    parser.add_argument("graph", type=Path, help="input graph in G-Thinker .bin format")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output image (default: <graph-stem>-degree-histogram.png)",
    )
    parser.add_argument("--title", help="plot title (default: graph filename stem)")
    parser.add_argument("--dpi", type=int, default=300, help="output resolution (default: 300)")
    parser.add_argument(
        "--marker-size", type=float, default=18.0,
        help="scatter marker area in points squared (default: 18)",
    )
    parser.add_argument("--show", action="store_true", help="also display the plot")
    return parser.parse_args()


def read_csr_degrees(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Return degrees, vertex count, edge count, and vertex-ID byte width."""
    size_t_bytes = struct.calcsize("P")
    header_bytes = 4 * size_t_bytes

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot access {path}: {exc}") from exc

    if file_size < header_bytes:
        raise ValueError(f"file is too small to contain a {header_bytes}-byte header")

    with path.open("rb") as graph_file:
        raw_header = graph_file.read(header_bytes)

    fields = [
        int.from_bytes(raw_header[i : i + size_t_bytes], byteorder=sys.byteorder)
        for i in range(0, header_bytes, size_t_bytes)
    ]
    vertex_id_bytes, edge_id_bytes, vertex_count, edge_count = fields

    if vertex_id_bytes not in (1, 2, 4, 8):
        raise ValueError(f"unsupported vertex-ID width in header: {vertex_id_bytes} bytes")
    if edge_id_bytes not in (1, 2, 4, 8):
        raise ValueError(f"unsupported edge-offset width in header: {edge_id_bytes} bytes")

    expected_size = (
        header_bytes
        + (vertex_count + 1) * edge_id_bytes
        + edge_count * vertex_id_bytes
    )
    if file_size != expected_size:
        raise ValueError(
            "file size does not match its CSR header: "
            f"expected {expected_size:,} bytes, found {file_size:,}. "
            "The file may use a different machine's size_t width or be truncated."
        )

    offset_dtype = np.dtype(f"=u{edge_id_bytes}")
    row_offsets = np.memmap(
        path,
        dtype=offset_dtype,
        mode="r",
        offset=header_bytes,
        shape=(vertex_count + 1,),
    )

    if row_offsets[0] != 0:
        raise ValueError(f"invalid CSR: first row offset is {int(row_offsets[0])}, not 0")
    if row_offsets[-1] != edge_count:
        raise ValueError(
            f"invalid CSR: final row offset is {int(row_offsets[-1]):,}, "
            f"but header edge count is {edge_count:,}"
        )
    if np.any(row_offsets[1:] < row_offsets[:-1]):
        raise ValueError("invalid CSR: row offsets are not monotonically increasing")

    degrees = np.diff(row_offsets)
    return degrees, vertex_count, edge_count, vertex_id_bytes


def plot_degree_histogram(
    degrees: np.ndarray,
    title: str,
    output: Path,
    dpi: int,
    marker_size: float,
    show: bool,
) -> None:
    nonzero = degrees[degrees > 0]
    if nonzero.size == 0:
        raise ValueError("the graph has no vertices with nonzero degree")

    degree_values, counts = np.unique(nonzero, return_counts=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(
        degree_values,
        counts,
        s=marker_size,
        color="#1010a0",
        edgecolors="none",
        alpha=0.88,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log scale)", fontsize=18)
    ax.set_ylabel("Count (log scale)", fontsize=18)
    ax.set_title(title, fontsize=22, pad=18)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, color="0.85")
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=13)
    ax.tick_params(which="major", length=8, width=1.5)
    ax.tick_params(which="minor", length=4, width=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    global np, plt
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        print(
            f"error: missing Python package {exc.name!r}; "
            "install plotting dependencies with: python3 -m pip install numpy matplotlib",
            file=sys.stderr,
        )
        return 1

    output = args.output or args.graph.with_name(
        f"{args.graph.stem}-degree-histogram.png"
    )

    try:
        degrees, vertex_count, edge_count, vertex_id_bytes = read_csr_degrees(args.graph)
        plot_degree_histogram(
            degrees=degrees,
            title=args.title or args.graph.stem,
            output=output,
            dpi=args.dpi,
            marker_size=args.marker_size,
            show=args.show,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    zero_degree_count = int(np.count_nonzero(degrees == 0))
    print(f"vertices: {vertex_count:,}")
    print(f"directed CSR entries: {edge_count:,}")
    print(f"vertex-ID width: {vertex_id_bytes} bytes")
    print(f"zero-degree vertices omitted from log plot: {zero_degree_count:,}")
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
