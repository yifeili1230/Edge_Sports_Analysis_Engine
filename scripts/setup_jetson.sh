#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_JETPACK=false
SKIP_MODEL=false
BUILD_TENSORRT=false

usage() {
    cat <<'EOF'
Usage: scripts/setup_jetson.sh [options]

Installs the C++ toolchain and CPU-capable OpenCV, downloads both pose models,
builds the project, runs tests, and reports whether CUDA FP16 DNN is available.

Options:
  --install-jetpack  Also install NVIDIA's large nvidia-jetpack metapackage.
  --build-tensorrt   Build the device-specific YOLO11n-pose FP16 engine.
  --skip-model       Do not download either pose model.
  -h, --help         Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --install-jetpack) INSTALL_JETPACK=true ;;
        --build-tensorrt) BUILD_TENSORRT=true ;;
        --skip-model) SKIP_MODEL=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "$(uname -m)" != "aarch64" ]] || [[ ! -r /etc/nv_tegra_release ]]; then
    echo "This setup script must be run on an NVIDIA Jetson (aarch64)." >&2
    exit 1
fi

if ((EUID == 0)); then
    SUDO=()
else
    SUDO=(sudo)
fi

echo "Jetson release:"
head -n 1 /etc/nv_tegra_release

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libopencv-dev \
    ninja-build \
    pkg-config \
    v4l-utils

if [[ "${INSTALL_JETPACK}" == true ]]; then
    "${SUDO[@]}" apt-get install -y nvidia-jetpack
fi

if [[ "${SKIP_MODEL}" == false ]]; then
    "${ROOT_DIR}/scripts/download_pose_model.sh"
fi
if [[ "${BUILD_TENSORRT}" == true ]]; then
    "${ROOT_DIR}/scripts/build_yolo11_tensorrt.sh"
fi

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/build" --parallel 2
ctest --test-dir "${ROOT_DIR}/build" --output-on-failure
"${ROOT_DIR}/build/video_engine_diagnostics"

cat <<EOF

Setup completed.

CPU smoke test (safe with every OpenCV build):
  cd ${ROOT_DIR}
  ./build/video_engine --source video_source/squat.mov \\
    --config configs/pose.yaml --inference-platform cpu --no-display --no-save

If diagnostics reported "DNN CUDA FP16 target available: yes", use:
  ./build/video_engine --config configs/pose_jetson.yaml

If you selected --build-tensorrt, use YOLO on the GPU regardless of OpenCV CUDA:
  ./build/video_engine --config configs/yolo11_pose_jetson.yaml

Otherwise, CPU mode is ready. See docs/JETSON_SETUP.md for both GPU paths.
EOF
