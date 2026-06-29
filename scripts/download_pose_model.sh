#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OPENPOSE_DEST="${ROOT_DIR}/models/pose_iter_440000.caffemodel"
readonly OPENPOSE_SHA256="b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549"
# The original CMU host no longer resolves reliably. This mirror URL is pinned to an
# immutable revision, and the downloaded bytes are verified against the model checksum.
readonly OPENPOSE_DEFAULT_URL="https://huggingface.co/camenduru/openpose/resolve/7dfccd4d04829843e92f7702eb100544357a4f72/pose_iter_440000.caffemodel?download=true"
readonly OPENPOSE_URL="${OPENPOSE_MODEL_URL:-${OPENPOSE_DEFAULT_URL}}"

readonly YOLO_DEST="${ROOT_DIR}/models/yolo11n-pose.onnx"
readonly YOLO_SHA256="3f170e2b091fb37e4b0c4d36e90cf9d201592387b6210a1f10ad1fa9d1557199"
readonly YOLO_DEFAULT_URL="https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/0d17b24163fbc33dec51d811d7be8db15b8df274/yolo11n-pose.onnx?download=true"
readonly YOLO_URL="${YOLO_MODEL_URL:-${YOLO_DEFAULT_URL}}"

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required. Install it with: sudo apt install curl" >&2
    exit 1
fi

selection="${1:---all}"
if [[ "${selection}" != "--all" && "${selection}" != "--openpose" &&
      "${selection}" != "--yolo" ]]; then
    echo "Usage: scripts/download_pose_model.sh [--all|--openpose|--yolo]" >&2
    exit 2
fi

mkdir -p "${ROOT_DIR}/models"
trap 'rm -f "${OPENPOSE_DEST}.partial" "${YOLO_DEST}.partial"' EXIT

download_model() {
    local name="$1"
    local destination="$2"
    local checksum="$3"
    local url="$4"
    local temporary_file="${destination}.partial"

    if [[ -f "${destination}" ]] &&
       echo "${checksum}  ${destination}" | sha256sum --check --status; then
        echo "${name} is already present and verified: ${destination}"
        return
    fi

    echo "Downloading ${name}..."
    curl --fail --location --retry 3 --retry-delay 2 \
        --output "${temporary_file}" "${url}"
    echo "${checksum}  ${temporary_file}" | sha256sum --check
    mv "${temporary_file}" "${destination}"
    echo "${name} installed at: ${destination}"
}

if [[ "${selection}" == "--all" || "${selection}" == "--openpose" ]]; then
    download_model "OpenPose COCO Caffe model (209 MB)" \
        "${OPENPOSE_DEST}" "${OPENPOSE_SHA256}" "${OPENPOSE_URL}"
fi
if [[ "${selection}" == "--all" || "${selection}" == "--yolo" ]]; then
    download_model "YOLO11n-pose ONNX model (12 MB)" \
        "${YOLO_DEST}" "${YOLO_SHA256}" "${YOLO_URL}"
fi

trap - EXIT
