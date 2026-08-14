#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="${GAMMA_EXE:-sm}"
ARCH="${GAMMA_ARCH:-sm_80}"

usage() {
    cat <<EOF
Usage:
  $0 --rebuild -dg <graph.bin> -q <query_id>
  $0 -dg <graph.bin> -q <query_id> [gamma args]
  $0 <gamma_graph_prefix> <query_file> <graph_mt> [debug]

Environment:
  GAMMA_EXE=<path>   Executable to build/run. Default: sm
  GAMMA_ARCH=<arch>  CUDA architecture for rebuild. Default: sm_80

Examples:
  $0 -dg ../app_gmatch/ds/edit-shwiki.bin -q 2
  GAMMA_ARCH=sm_86 $0 --rebuild -dg /path/to/data.bin -q 6
EOF
}

if [ "$#" -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

cd "$SCRIPT_DIR" || exit 1

if [ "${1:-}" = "--rebuild" ]; then
    shift
    rm -f "$EXE" log.o
fi

if [ ! -x "$EXE" ]; then
    echo "Building GAMMA target: $EXE ARCH=$ARCH"
    make ARCH="$ARCH" "$EXE"
fi

"./$EXE" "$@"
