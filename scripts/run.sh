#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="../build"
BIN_DIR="../bin"
EXECUTABLE="$BIN_DIR/skew_correction"
PROJECT_ROOT="$(dirname "$(realpath "$0")")/.."


build_project() {
    mkdir -p "$BUILD_DIR"

    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd - >/dev/null
}

build_project

if [[ -x "$EXECUTABLE" ]]; then
    "$EXECUTABLE"
else
    exit 1
fi
