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
  --config configs/pose.yaml \
  --inference-platform cpu
```

### 3.3 Video file with CPU pose inference

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose.yaml
```

### 3.4 Headless benchmark

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose.yaml \
  --no-display \
  --no-save
```

Do not select the Jetson profile with a normal macOS OpenCV build. macOS does not provide
the NVIDIA CUDA backend.

### 3.5 Squat analysis

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/squat.yaml
```

The live window adds a metrics panel beside the video. The final session summary is
written to `output/squat.json`. See
[Squat Analytics Guide](SQUAT_ANALYTICS_GUIDE.md).

## 4. Build and Run on Jetson Orin Nano

For a new board, start with [Jetson First-Time Setup](JETSON_SETUP.md). It provides the
maintained bootstrap script, verified model download, CUDA capability diagnostic, and
optional source build. The manual commands below are useful once the base environment
already exists.

### 4.1 Software requirements

JetPack 6.2.x is the project's preferred baseline. CPU mode works with the distribution
OpenCV. The Jetson GPU profile requires OpenCV itself to be built with CUDA DNN support,
including:

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
  --config configs/pose.yaml \
  --inference-platform jetson-orin-nano
```

Prefer `configs/pose_jetson.yaml` for the existing Caffe model because it also selects a
smaller 192×192 input. Both configurations run fresh inference on every frame.

For YOLO11n-pose, use native TensorRT; it does not require CUDA support in OpenCV:

```bash
./scripts/download_pose_model.sh --yolo
./scripts/build_yolo11_tensorrt.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/video_engine --config configs/yolo11_pose_jetson.yaml
```

The engine file is device/runtime-specific and intentionally ignored by Git.

### 4.4 Video file with CUDA FP16 pose inference

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose_jetson.yaml
```

### 4.5 SSH or headless benchmark

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose_jetson.yaml \
  --no-display \
  --no-save
```

In another terminal, monitor the hardware:

```bash
tegrastats
```

Check GPU load, CPU load, memory use, temperature, and power.

### 4.6 Squat analysis with CUDA FP16

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose_jetson.yaml \
  --exercise squat
```

Exercise analysis remains on the CPU and consumes the canonical pose produced by CUDA
inference. Its measured cost is small compared with the current neural network.

## 5. Command-Line Options

| Option | Effect |
| --- | --- |
| `--config <path>` | Select a configuration file |
| `--source webcam` | Use the default camera |
| `--source <video>` | Use a video file |
| `--inference-platform cpu` | Force OpenCV CPU inference |
| `--inference-platform jetson` | Force CUDA FP16 inference |
| `--inference-platform jetson-orin-nano` | Explicit alias for the Jetson profile |
| `--exercise squat` | Enable mathematical squat analysis and its live side panel |
| `--film-side <view>` | Select one of eight camera views around the subject |
| `--analysis-output <dir>` | Select the JSON summary directory |
| `--display` | Enable the OpenCV window |
| `--no-display` | Disable the window, useful over SSH |
| `--save <prefix>` | Enable output video and set its path prefix |
| `--no-save` | Disable output video |

The configuration file is loaded first and command-line overrides are applied afterward.
The supported overrides include source, inference platform, exercise analysis, display,
saving behavior, and analytics output location.

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

After an analytics result is successfully written, the terminal prints one
`[Average Benchmarks]` line associated with its JSON path. Its FPS is end-to-end
throughput across the completed run, and each stage latency is the arithmetic mean
of that stage's samples across all processed frames.

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
