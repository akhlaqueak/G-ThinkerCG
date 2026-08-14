#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${EGSM_BUILD_DIR:-build}"
CUDA_ARCH="${EGSM_CUDA_ARCH:-80}"
EXE="$SCRIPT_DIR/$BUILD_DIR/EGSM"

usage() {
    cat <<EOF
Usage:
  $0 --rebuild -dg <graph.bin> -q <query_id> [EGSM options]
  $0 -dg <graph.bin> -q <query_id> [EGSM options]
  $0 -d <data.graph> -q <query.graph> [EGSM options]

Environment:
  EGSM_BUILD_DIR=<dir>    Build directory. Default: build
  EGSM_CUDA_ARCH=<arch>   CUDA architecture. Default: 80

Examples:
  $0 -dg ../app_gmatch/ds/edit-shwiki.bin -q 2
  EGSM_CUDA_ARCH=80 $0 --rebuild -dg /path/to/data.bin -q 6
EOF
}

if [ "$#" -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

cd "$SCRIPT_DIR" || exit 1

if [ "${1:-}" = "--rebuild" ]; then
    shift
    rm -rf "$BUILD_DIR"
fi

if [ ! -x "$EXE" ]; then
    cmake -S . -B "$BUILD_DIR" -DEGSM_CUDA_ARCHITECTURES="$CUDA_ARCH"
    cmake --build "$BUILD_DIR" --target EGSM -j 1
fi

"$EXE" "$@"
