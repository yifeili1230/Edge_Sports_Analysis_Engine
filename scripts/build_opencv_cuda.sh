#!/usr/bin/env bash
set -Eeuo pipefail

readonly OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
readonly INSTALL_PREFIX="${OPENCV_INSTALL_PREFIX:-/opt/opencv-cuda}"
readonly JOBS="${JOBS:-2}"
readonly WORK_DIR="${OPENCV_BUILD_ROOT:-$HOME/opencv-cuda-build}"
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -m)" != "aarch64" ]] || [[ ! -r /etc/nv_tegra_release ]]; then
    echo "This script must be run on an NVIDIA Jetson (aarch64)." >&2
    exit 1
fi
if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    echo "CUDA toolkit not found. First run: sudo apt install nvidia-jetpack" >&2
    exit 1
fi

if ((EUID == 0)); then
    SUDO=()
else
    SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
    build-essential ca-certificates cmake curl git ninja-build pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libgtk-3-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libjpeg-dev libpng-dev libswscale-dev libtiff-dev libv4l-dev

mkdir -p "${WORK_DIR}"
if [[ ! -d "${WORK_DIR}/opencv/.git" ]]; then
    git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv.git "${WORK_DIR}/opencv"
fi
if [[ ! -d "${WORK_DIR}/opencv_contrib/.git" ]]; then
    git clone --branch "${OPENCV_VERSION}" --depth 1 \
        https://github.com/opencv/opencv_contrib.git "${WORK_DIR}/opencv_contrib"
fi

cmake -S "${WORK_DIR}/opencv" -B "${WORK_DIR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DOPENCV_EXTRA_MODULES_PATH="${WORK_DIR}/opencv_contrib/modules" \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCUDA_ARCH_BIN=8.7 \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_CUBLAS=ON \
    -DOPENCV_DNN_CUDA=ON \
    -DWITH_GSTREAMER=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF

cmake --build "${WORK_DIR}/build" --parallel "${JOBS}"
"${SUDO[@]}" cmake --install "${WORK_DIR}/build"

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build-cuda" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR="${INSTALL_PREFIX}/lib/cmake/opencv4"
cmake --build "${ROOT_DIR}/build-cuda" --parallel "${JOBS}"
"${ROOT_DIR}/build-cuda/video_engine_diagnostics" --require-cuda

echo "CUDA build is ready: ${ROOT_DIR}/build-cuda/video_engine"
