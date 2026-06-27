# macOS and Jetson Orin Nano Run Guide

This project can run on both macOS and Jetson Orin Nano, but each platform uses its own
OpenCV build and pose-inference configuration.

> A binary built on macOS cannot be copied directly to Orin Nano. Build the project
> separately on each machine. The source code, configuration files, and model files can
> be shared.

## 1. Platform Configuration Matrix

| Platform | Configuration | DNN backend | DNN target | Input size | Inference cadence |
| --- | --- | --- | --- | --- | --- |
| macOS | `configs/pose.yaml` | OpenCV | CPU | 256×256 | Every frame |
| Jetson Orin Nano | `configs/pose_jetson.yaml` | CUDA | CUDA FP16 | 192×192 | Every frame |

`configs/pose.yaml` prioritizes compatibility. `configs/pose_jetson.yaml` prioritizes
Orin Nano performance and requires an OpenCV build with CUDA DNN support.

## 2. Model Files Required on Both Platforms

Run commands from the project root so these relative paths resolve correctly:

```text
models/pose_deploy_linevec.prototxt
models/pose_iter_440000.caffemodel
```

## 3. Build and Run on macOS

### 3.1 Build

After installing CMake and OpenCV:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### 3.2 Webcam with CPU pose inference

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose.yaml
```

The configuration selects:

```yaml
pose_backend: opencv
pose_target: cpu
```

The CPU profile can also be selected explicitly:

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose.yaml \
  --inference-platform cpu
```

### 3.3 Video file with CPU pose inference

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose.yaml
```

### 3.4 Headless benchmark

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose.yaml \
  --no-display \
  --no-save
```

Do not select the Jetson profile with a normal macOS OpenCV build. macOS does not provide
the NVIDIA CUDA backend.

## 4. Build and Run on Jetson Orin Nano

### 4.1 Software requirements

JetPack 6 is recommended. OpenCV must be built with CUDA DNN support, including:

```text
WITH_CUDA=ON
WITH_CUDNN=ON
WITH_CUBLAS=ON
OPENCV_DNN_CUDA=ON
CUDA_ARCH_BIN=8.7
```

Installing CUDA alone does not make OpenCV DNN CUDA-enabled. The application uses the
OpenCV library selected by CMake.

Inspect the installed OpenCV build with:

```bash
opencv_version --verbose
```

Confirm that CUDA and cuDNN are enabled.

### 4.2 Build

From the project root on Orin Nano:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

If multiple OpenCV installations exist, point CMake to the CUDA-enabled build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/cuda-opencv/lib/cmake/opencv4
```

### 4.3 Webcam with CUDA FP16 pose inference

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose_jetson.yaml
```

The profile in `pose_jetson.yaml` resolves to:

```text
pose_backend = cuda
pose_target  = cuda_fp16
```

The following command selects the same inference device but retains the input size from
`pose.yaml`:

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose.yaml \
  --inference-platform jetson-orin-nano
```

Prefer `configs/pose_jetson.yaml` for the existing Caffe model because it also selects a
smaller 192×192 input. Both configurations run fresh inference on every frame.

### 4.4 Video file with CUDA FP16 pose inference

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose_jetson.yaml
```

### 4.5 SSH or headless benchmark

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose_jetson.yaml \
  --no-display \
  --no-save
```

In another terminal, monitor the hardware:

```bash
tegrastats
```

Check GPU load, CPU load, memory use, temperature, and power.

## 5. Command-Line Options

| Option | Effect |
| --- | --- |
| `--config <path>` | Select a configuration file |
| `--source webcam` | Use the default camera |
| `--source <video>` | Use a video file |
| `--pipeline pose` | Run pose estimation |
| `--pipeline motion` | Run motion detection |
| `--pipeline tracking` | Run motion detection and tracking |
| `--inference-platform cpu` | Force OpenCV CPU inference |
| `--inference-platform jetson` | Force CUDA FP16 inference |
| `--inference-platform jetson-orin-nano` | Explicit alias for the Jetson profile |
| `--display` | Enable the OpenCV window |
| `--no-display` | Disable the window, useful over SSH |
| `--save <prefix>` | Enable output video and set its path prefix |
| `--no-save` | Disable output video |

The configuration file is loaded first and command-line overrides are applied afterward.
The supported overrides include source, inference platform, display, and saving behavior.

## 6. Reading Latency Logs

Pose mode reports processor and inference-stage latency:

```text
[Frame 3] FPS: 3.4
| resize: 0.40 ms
| pose_preprocess: 0.41 ms
| pose_inference: 261.56 ms
| pose_postprocess: 0.01 ms
| pose_estimator: 261.98 ms
| skeleton_renderer: 0.10 ms
| pipeline: 262.47 ms
```

This sample was measured on the current macOS CPU setup; it is not an Orin Nano
performance claim.

- `pose_inference` is the wall-clock time around `net.forward()` and is the primary value
  for comparing CPU and CUDA execution.
- `pose_estimator` includes preprocessing, inference, and postprocessing.
- `pipeline` includes all processors but excludes capture, display, and video writing.
For a fair platform comparison, use the same video and model input size, with display and
saving disabled.

## 7. Common Problems

### CUDA backend is unavailable

Check that:

1. OpenCV was built with `OPENCV_DNN_CUDA=ON`.
2. CMake linked this application to that OpenCV installation.
3. cuDNN is available.
4. The Orin CUDA architecture is set to `8.7` when building CUDA code.

### No window appears over SSH

SSH normally has no graphical display. Use `--no-display`, run from the local Jetson
desktop, or configure VNC/X11 forwarding.

### Pose inference is slow on macOS

This is expected for the current OpenPose Caffe model on CPU. macOS is useful for
development and correctness checks; Orin Nano is the CUDA performance target.

For the complete inference data flow and tuning strategy, see
[Pose Inference Guide](POSE_INFERENCE_GUIDE.md).
