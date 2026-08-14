#!/usr/bin/env python3
"""Report basic statistics from an app_qc expanded binary graph (.sbin)."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import BinaryIO, NamedTuple


HEADER = struct.Struct("=iQQ")
UINT64 = struct.Struct("=Q")
OFFSET_BLOCK_SIZE = 65_536


class GraphStats(NamedTuple):
    vertices: int
    edges: int
    max_degree: int


def read_exact(graph_file: BinaryIO, byte_count: int, description: str) -> bytes:
    data = graph_file.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"truncated .sbin file while reading {description}")
    return data


def read_sbin_stats(graph_file: BinaryIO) -> GraphStats:
    vertices, edges, two_hop_entries = HEADER.unpack(
        read_exact(graph_file, HEADER.size, "header")
    )
    if vertices <= 0:
        raise ValueError(f"invalid vertex count in .sbin header: {vertices}")

    expected_size = (
        HEADER.size
        + edges * 4
        + (vertices + 1) * UINT64.size
        + two_hop_entries * 4
        + (vertices + 1) * UINT64.size
    )
    graph_file.seek(0, 2)
    actual_size = graph_file.tell()
    if actual_size != expected_size:
        raise ValueError(
            f"invalid .sbin size: expected {expected_size} bytes from the header, "
            f"found {actual_size}"
        )

    # The one-hop offset array follows the one-hop neighbor array.
    graph_file.seek(HEADER.size + edges * 4)
    previous_offset = UINT64.unpack(
        read_exact(graph_file, UINT64.size, "first one-hop offset")
    )[0]
    if previous_offset != 0:
        raise ValueError(f"invalid first one-hop offset: {previous_offset} (expected 0)")

    max_degree = 0
    offsets_remaining = vertices
    while offsets_remaining:
        block_count = min(offsets_remaining, OFFSET_BLOCK_SIZE)
        block = read_exact(
            graph_file, block_count * UINT64.size, "one-hop offsets"
        )
        for (current_offset,) in struct.iter_unpack("=Q", block):
            if current_offset < previous_offset:
                raise ValueError("one-hop offsets are not nondecreasing")
            max_degree = max(max_degree, current_offset - previous_offset)
            previous_offset = current_offset
        offsets_remaining -= block_count

    if previous_offset != edges:
        raise ValueError(
            f"invalid final one-hop offset: {previous_offset} (expected E={edges})"
        )

    return GraphStats(vertices, edges, max_degree)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report V, E, and maximum one-hop degree from an .sbin file."
    )
    parser.add_argument("sbin_file", type=Path, help="expanded binary graph file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.sbin_file.open("rb") as graph_file:
            stats = read_sbin_stats(graph_file)
    except (OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"V = {stats.vertices}")
    print(f"E = {stats.edges}")
    print(f"max_degree = {stats.max_degree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
