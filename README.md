# Real-Time Pose and Exercise Analytics Engine

A modular C++17/OpenCV engine for real-time video processing, human-pose inference, and
exercise analytics on macOS and NVIDIA Jetson Orin Nano.

The project turns webcam or recorded video into connected pose joints. Its squat and
overhead-press modes derive joint angles, movement speed,
phases, and repetitions from reliable per-frame timestamps while preserving structured
results for future subject tracking and learned analytics.

> Current state: the video engine, pose processing flow, canonical pose schema, timing contract,
> mathematical squat/OHP analytics, platform profiles, and tests are implemented. Persistent
> primary-person tracking and a faster TensorRT pose model are on the roadmap.

## What Has Been Built

- Webcam and recorded-video input
- OpenPose COCO 18-joint inference through OpenCV DNN
- Fresh pose inference for every decoded frame, with no stale-result reuse
- Canonical named-joint pose data independent of the current model
- Monotonic frame IDs and source-relative timestamps for motion calculations
- Optional depth and 3D fields reserved in the pose interface
- Per-stage latency for preprocessing, inference, postprocessing, rendering, and the
  complete processor pipeline
- Real-time squat phase, joint-angle, speed, and repetition analysis in a side panel
- Input-named JSON squat summaries with per-rep metrics
- Rear-view overhead-press phase, lockout, velocity, rep, and JSON analysis
- CPU configuration for macOS and CUDA FP16 configuration for Jetson Orin Nano
- Automated tests for rendering, timeline behavior, pose mapping, and
  squat state semantics

The current Caffe model produces 2D keypoints. Those results are available as structured
data for the squat analyzer and future learned analytics instead of being limited to the
skeleton overlay.

## Demo

The executable always performs pose estimation and draws joints connected by skeleton
lines. It can display a live OpenCV window or save an annotated video. A typical
runtime report looks like:

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

Run the included squat video:

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/squat.yaml
```

The live result places the video on the left and squat phase, angles, speed, and rep count
on the right. The final summary is written to `output/squat.json`. Use
`--save output/pose_demo` to write an annotated video.

Run the included overhead-press video with the current CPU analytics stack:

```bash
./build/video_engine --config configs/ohp.yaml
```

On Jetson, use `configs/yolo11_ohp_jetson.yaml` for native TensorRT inference. Analytics
profiles save the same composite frame shown in the window, including the side panel.

Equivalent Jetson CUDA FP16 profiles for the OpenPose Caffe model are
`configs/caffe_pose_jetson.yaml`, `configs/caffe_squat_jetson.yaml`, and
`configs/caffe_ohp_jetson.yaml`.

## Architecture

```text
Webcam / Video File
        |
        v
   VideoSource
        |
        v
  FrameTimeline
  frame ID + source time + fresh validity state
        |
        v
 ResizeProcessor
        |
        v
 PoseEstimator
 preprocess -> DNN -> decode
        |
        v
 Canonical Pose Schema
        |
        v
 Optional PoseAnalyticsProcessor
        |
        v
 SkeletonRenderer
        |
        v
 Optional PoseAnalyticsRenderer
        |
        v
 Display / Video Output
```

All frame processors implement a shared `IFrameProcessor` interface and exchange data
through `FrameContext`. Pose-model channel indices are translated by
`OpenPoseCocoAdapter` into named joints before rendering or analytics. This keeps future
ONNX/TensorRT models and exercise-analysis modules separate from the current Caffe
implementation.

The CPU currently handles capture, resize, orchestration, pose decoding, analytics-ready
data, rendering, and output. Neural-network inference runs on the configured OpenCV DNN
target:

- macOS profile: CPU
- Jetson Orin Nano profile: CUDA FP16
- Planned Orin production path: lightweight ONNX model with TensorRT FP16

For the complete frame lifecycle and CPU/GPU boundary, see the
[Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md) and
[Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md).

## How to Run

### Requirements

- CMake 3.16 or newer
- A C++17 compiler
- OpenCV with `core`, `imgproc`, `highgui`, `videoio`, and `dnn`
- OpenPose COCO model files:
  - `models/pose_deploy_linevec.prototxt`
  - `models/pose_iter_440000.caffemodel`

The repository includes the model definition but not the large Caffe weights. Place the
weights at the path above before running pose inference.

Download both supported pose models with verified checksums:

```bash
./scripts/download_pose_model.sh --all
```

The second model is `models/yolo11n-pose.onnx`. Build its device-specific TensorRT FP16
engine once on the Jetson:

```bash
./scripts/build_yolo11_tensorrt.sh
./build/video_engine --config configs/yolo11_pose_jetson.yaml
```

Run the portrait-safe squat profile with:

```bash
./build/video_engine --config configs/yolo11_squat_jetson.yaml
```

OpenCV letterboxing, keypoint decoding, NMS, analytics, and rendering stay on the CPU;
only the network runs through native TensorRT on the GPU/Tensor Cores. YOLO11n-pose
produces multiple COCO-17 poses. The adapter maps those joints into the engine's
canonical schema and synthesizes the missing neck point from the shoulders.
Ultralytics YOLO11 models are offered under AGPL-3.0 or an Enterprise license; confirm
that choice for the intended deployment.

Frame resizing uses `resize_mode: fit` by default. This preserves the source aspect
ratio instead of stretching portrait video into 4:3 and corrupting joint geometry.
Changing `width`/`height` only changes CPU frame processing; YOLO inference remains
640×640 because the TensorRT engine has a fixed input shape.

### Build and test

Use a Release build for meaningful performance measurements:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

### macOS

Run pose inference with the CPU profile:

```bash
./build/video_engine \
  --config configs/pose.yaml
```

### Jetson Orin Nano

For a newly flashed board with no compiler or model installed:

```bash
chmod +x scripts/*.sh
./scripts/setup_jetson.sh --build-tensorrt
```

This installs the native C++ build tools and CPU-capable OpenCV, downloads and verifies
both models, builds the TensorRT FP16 engine, builds the project, and runs its tests.
Use YOLO11n-pose without requiring a CUDA-enabled OpenCV build:

```bash
./build/video_engine --config configs/yolo11_pose_jetson.yaml
```

The legacy Caffe model still uses OpenCV DNN. If diagnostics reports CUDA FP16 as
available, run:

```bash
./build/video_engine \
  --config configs/pose_jetson.yaml
```

Ubuntu's `libopencv-dev` may not include CUDA DNN even when JetPack and CUDA are
installed. Native TensorRT YOLO inference and the CPU paths still work; the optional
`scripts/build_opencv_cuda.sh` creates a separate CUDA-enabled OpenCV and `build-cuda/`
application. See the [first-time Jetson setup](docs/JETSON_SETUP.md) for exact commands,
storage cautions, headless use, and camera support.

Common overrides:

| Option | Purpose |
| --- | --- |
| `--source webcam` | Use the default camera |
| `--source <path>` | Read a video file |
| `--inference-platform cpu` | Force OpenCV DNN CPU inference |
| `--inference-platform jetson` | Select CUDA FP16 inference |
| `--exercise squat` | Enable mathematical squat analysis |
| `--exercise ohp` | Enable mathematical overhead-press analysis |
| `--film-side <view>` | Select one of eight camera views around the subject |
| `--analysis-output <dir>` | Select the JSON summary directory |
| `--display` / `--no-display` | Enable or disable the GUI window |
| `--save <prefix>` / `--no-save` | Enable or disable annotated video output |

## Current Benchmark

Representative frame from the completed 117-frame macOS CPU squat validation:

| Stage | Example latency |
| --- | ---: |
| Resize | 0.28 ms |
| Pose preprocessing | 0.26 ms |
| Neural-network inference | 262.30 ms |
| Pose postprocessing | 0.01 ms |
| Complete pose estimator | 262.57 ms |
| Squat analytics | <0.01 ms |
| Skeleton rendering | 0.09 ms |
| Analytics side panel | 0.60 ms |
| Processor pipeline | 263.55 ms |
| Observed throughput | 3.8 FPS |

Inference is the dominant cost; resizing and drawing are currently negligible by
comparison. This points the next optimization effort toward a smaller model and an
accelerated inference backend.

These numbers are a development baseline, not a formal cross-platform benchmark. The
Jetson Orin Nano profile is prepared but has not yet been measured on the physical
device. Its project target is at least **20 FPS** and less than **100 ms** end-to-end
latency.

For a reproducible benchmark procedure and an explanation of every timing field, see the
[Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md).

## Roadmap

### Next

- Add persistent primary-person tracking
  - retain the selected athlete when other people enter the frame
  - use position, bounding-box overlap, keypoint similarity, and reacquisition rules

### After that

- Replace the heavy Caffe network with a faster, accurate multi-person ONNX pose model
- Use one canonical keypoint contract across macOS and Jetson
- Integrate TensorRT FP16, then evaluate eligible INT8 execution on Orin Nano
- Benchmark power mode, temperature, Tensor Core use, and CPU/GPU transfers
- Evaluate hardware decoding, asynchronous stages, and zero-copy only where profiling
  demonstrates a meaningful gain
- Connect stereo/depth input through the reserved depth and 3D pose interface
- Keep the mathematical analytics API open for future learned exercise-analysis models

Detailed decisions, dependencies, and implementation tickets are maintained in the
[Pose Analytics Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md).

## More Documentation

| Page | Contents |
| --- | --- |
| [Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md) | Full runtime architecture, processors, configuration, and extension points |
| [Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md) | Pose data flow, timing, settings, CPU/GPU boundaries, and benchmarking |
| [Squat Analytics Guide](docs/SQUAT_ANALYTICS_GUIDE.md) | Live metrics, rep semantics, settings, JSON output, and limitations |
| [Platform Run Guide](docs/PLATFORM_RUN_GUIDE.md) | macOS and Jetson Orin Nano setup and commands |
| [Jetson First-Time Setup](docs/JETSON_SETUP.md) | New-board bootstrap, model download, diagnostics, and optional CUDA OpenCV build |
| [Pose Analytics Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md) | Resolved design choices, roadmap tickets, and acceptance criteria |
| [Resources](RESOURCES.md) | NVIDIA, OpenCV, TensorRT, tracking, and calibration references |

Quick references:

- [Jetson Pipeline Cheat Sheet](reference/jetson-pipeline-cheatsheet.html)
- [Pose Analytics Math Cheat Sheet](reference/pose-analytics-cheatsheet.html)
