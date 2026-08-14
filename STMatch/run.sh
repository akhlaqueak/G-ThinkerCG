#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="${STMATCH_EXE:-bin/table_edge_ulb.exe}"
ARCH="${STMATCH_ARCH:-sm_80}"
GRID_DIM="${STMATCH_GRID_DIM:-16}"
BLOCK_DIM="${STMATCH_BLOCK_DIM:-128}"
GRAPH_DEGREE="${STMATCH_GRAPH_DEGREE:-16384}"

usage() {
    cat <<EOF
Usage:
  $0 --rebuild -dg <graph.bin> -q <query_id>
  $0 -dg <graph.bin> -q <query_id> [stmatch args]
  $0 <stmatch_graph_prefix> <pattern.g>

Environment:
  STMATCH_EXE=<path>   Executable to build/run. Default: bin/table_edge_ulb.exe
  STMATCH_ARCH=<arch>  CUDA architecture for rebuild. Default: sm_80
  STMATCH_GRID_DIM=<n> CUDA grid dimension for rebuild. Default: 16
  STMATCH_BLOCK_DIM=<n> CUDA block dimension for rebuild. Default: 128
  STMATCH_GRAPH_DEGREE=<n> Per-set candidate capacity for rebuild. Default: 16384

Examples:
  $0 -dg ../app_gmatch/ds/edit-shwiki.bin -q 2
  STMATCH_GRID_DIM=32 STMATCH_BLOCK_DIM=256 STMATCH_GRAPH_DEGREE=32768 $0 --rebuild -dg /path/to/data.bin -q 6
  STMATCH_ARCH=sm_86 $0 --rebuild -dg /path/to/data.bin -q 6
  STMATCH_EXE=bin/fig_local.exe $0 -dg /path/to/data.bin -q 6
EOF
}

if [ "$#" -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

cd "$SCRIPT_DIR" || exit 1

if [ "${1:-}" = "--rebuild" ]; then
    shift
    rm -f "$EXE" "${EXE%.exe}.o"
fi

if [ ! -x "$EXE" ]; then
    echo "Building STMatch target: $EXE ARCH=$ARCH GRID_DIM=$GRID_DIM BLOCK_DIM=$BLOCK_DIM GRAPH_DEGREE=$GRAPH_DEGREE"
    make ARCH="$ARCH" GRID_DIM="$GRID_DIM" BLOCK_DIM="$BLOCK_DIM" GRAPH_DEGREE="$GRAPH_DEGREE" "$EXE"
fi

"./$EXE" "$@"
