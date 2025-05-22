BUILD_DIR="../build"
BIN_DIR="../bin"
EXECUTABLE="$BIN_DIR/skew_corrector"
PROJECT_ROOT="$(dirname "$(realpath "$0")")/.."


OUTPUT_DIR=$(dirname "$OUTPUT_IMAGE")
mkdir -p "$OUTPUT_DIR"

build_project() {
    mkdir -p "$BUILD_DIR"

    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd - >/dev/null
}

build_project

"$EXECUTABLE"