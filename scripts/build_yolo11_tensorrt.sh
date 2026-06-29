#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ONNX_MODEL="${ROOT_DIR}/models/yolo11n-pose.onnx"
readonly ENGINE="${ROOT_DIR}/models/yolo11n-pose-fp16.engine"
readonly TEMPORARY="${ENGINE}.partial"

if [[ ! -f "${ONNX_MODEL}" ]]; then
    "${ROOT_DIR}/scripts/download_pose_model.sh" --yolo
fi

TRTEXEC="${TRTEXEC:-}"
if [[ -z "${TRTEXEC}" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
        TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    else
        echo "trtexec was not found. Install TensorRT from nvidia-jetpack." >&2
        exit 1
    fi
fi

if [[ -f "${ENGINE}" && "${ENGINE}" -nt "${ONNX_MODEL}" ]]; then
    echo "TensorRT engine is already current: ${ENGINE}"
    exit 0
fi

trap 'rm -f "${TEMPORARY}"' EXIT
"${TRTEXEC}" \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${TEMPORARY}" \
    --fp16 \
    --memPoolSize=workspace:1024 \
    --avgTiming=1 \
    --builderOptimizationLevel=0 \
    --skipInference
mv "${TEMPORARY}" "${ENGINE}"
trap - EXIT
echo "TensorRT FP16 engine built for this Jetson: ${ENGINE}"
