# Jetson Orin Nano: First-Time Setup

This is the shortest reproducible path from a newly flashed Orin Nano to a running
project. Run every project command from the repository root.

## Supported baseline

- NVIDIA Jetson Orin Nano (`aarch64`)
- JetPack 6.2.x / Jetson Linux 36.x is the project's preferred compatibility target
- JetPack 7 can run the CPU path, but the optional OpenCV CUDA source build may require
  selecting a newer OpenCV release with `OPENCV_VERSION`
- C++17, CMake 3.16+, and OpenCV 4

Check the board before changing packages:

```bash
head -n 1 /etc/nv_tegra_release
uname -m
df -h /
```

NVIDIA's current JetPack installation instructions should be used if the board has not
yet been flashed or its firmware is too old. Do not change the L4T apt repository by
hand just to run this project.

## 1. One-command CPU-ready setup

```bash
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

The script:

1. Confirms it is running on a Jetson.
2. Installs the compiler, CMake, Ninja, OpenCV development files, curl, and camera tools.
3. Downloads the 209 MB OpenPose COCO weights and 12 MB YOLO11n-pose ONNX model,
   verifying both SHA-256 checksums.
4. Makes a Release build with at most two parallel compile jobs to avoid memory pressure.
5. Runs all tests.
6. Reports whether the linked OpenCV supports CUDA FP16 DNN.

To also build the YOLO11n-pose TensorRT engine:

```bash
./scripts/setup_jetson.sh --build-tensorrt
```

Use `--install-jetpack` only if CUDA, cuDNN, and TensorRT are not already installed and
the board has enough free storage:

```bash
./scripts/setup_jetson.sh --install-jetpack
```

## 2. Verify with the bundled video

This path works even when Ubuntu's OpenCV has no CUDA support:

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose.yaml \
  --inference-platform cpu \
  --no-display \
  --no-save
```

Stop with `Ctrl+C`. The current Caffe network is large, so CPU mode is a functional
smoke test rather than the production performance configuration.

For the squat UI and JSON report on a locally attached display:

```bash
./build/video_engine --config configs/squat.yaml
```

## 3. Check GPU readiness

```bash
./build/video_engine_diagnostics
```

Only use `configs/pose_jetson.yaml` when the result includes:

```text
DNN CUDA FP16 target available: yes
```

Then run:

```bash
./build/video_engine --config configs/pose_jetson.yaml
```

YOLO11n-pose does not require CUDA-enabled OpenCV. It uses the native TensorRT runner
compiled automatically when TensorRT development files are installed:

```bash
./scripts/build_yolo11_tensorrt.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/video_engine --config configs/yolo11_pose_jetson.yaml
```

For YOLO, capture, resize, model input creation, output decoding, analytics, and drawing
remain on the CPU. Only the neural network forward pass runs in the TensorRT FP16 engine
on the GPU/Tensor Cores. TensorRT engines are tied to the target TensorRT/GPU stack, so
build the engine on the Jetson rather than committing or copying it from another host.

The legacy `configs/pose_jetson.yaml` profile instead sends its Caffe network through
OpenCV DNN's CUDA FP16 target.

If the result is `no`, CUDA may be installed correctly while the linked OpenCV itself
was compiled without CUDA DNN. The application cannot repair that at runtime.

## 4. Optional CUDA-enabled OpenCV

This is a long source build and needs several gigabytes of free space. The default is
OpenCV 4.10.0, an install prefix of `/opt/opencv-cuda`, and two build jobs:

```bash
./scripts/build_opencv_cuda.sh
```

Overrides are available for newer JetPack toolchains or machines with more memory:

```bash
OPENCV_VERSION=4.12.0 JOBS=4 ./scripts/build_opencv_cuda.sh
```

The script does not replace Ubuntu's OpenCV. It builds the application separately in
`build-cuda/`, explicitly linked to `/opt/opencv-cuda`:

```bash
./build-cuda/video_engine_diagnostics --require-cuda
./build-cuda/video_engine --config configs/pose_jetson.yaml
```

Inspect the CMake configuration summary before allowing a long compile. It must report
CUDA, cuDNN, and `OPENCV_DNN_CUDA` as enabled.

## 5. Cameras and headless use

The current `webcam` source opens V4L2 camera index 0. It is suitable for a USB UVC
camera:

```bash
v4l2-ctl --list-devices
./build/video_engine --config configs/pose.yaml --source webcam
```

A CSI ribbon camera using `nvarguscamerasrc` is not yet implemented by this source
class. Validate deployment with the bundled video or a USB camera. CSI support requires
a dedicated GStreamer source rather than pretending the Argus pipeline is a file path.

Over SSH, always disable the window:

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose_jetson.yaml \
  --no-display \
  --no-save
```

Monitor the board in a second shell:

```bash
tegrastats
```

## 6. Common failures

- `Pose model weights not found`: run `scripts/download_pose_model.sh`.
- `DNN module was not built with CUDA backend`: run diagnostics and use CPU mode, or
  build CUDA-enabled OpenCV.
- Process killed during OpenCV compilation: retry with `JOBS=1`.
- `Failed to open source: webcam`: check `v4l2-ctl --list-devices`, permissions, and
  whether the camera is CSI rather than USB.
- No window over SSH: add `--no-display`.
- MP4 writer failure: add `--no-save`; encoding support depends on the OpenCV/FFmpeg or
  GStreamer build.

The legacy OpenPose weights are licensed by their provider for non-commercial use.
Confirm model licensing before using them in a commercial product.

Ultralytics states that YOLO11 models are available under AGPL-3.0 and Enterprise
licenses. The downloaded ONNX file contains those model weights; a third-party mirror
does not change their upstream license.
