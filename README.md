Demo link: 
https://drive.google.com/file/d/1pitZi41Tm_PuvF9850QGWtEyzKAKXDTN/view?usp=drive_link
https://drive.google.com/file/d/14UmqQF9ooPby_5HyPL2SiKm8YxDRWhNX/view?usp=sharing

# Real-Time Pose and Exercise Analytics Engine

A modular C++17/OpenCV engine for real-time video processing, human-pose inference, and
exercise analytics on macOS and NVIDIA Jetson Orin Nano.

The project turns webcam or recorded video into connected pose joints. Its squat and
overhead-press modes derive joint angles, movement speed,
phases, and repetitions from reliable per-frame timestamps while preserving structured
results for future subject tracking and learned analytics.

> Current state: the engine is deployed on a Jetson Orin Nano Developer Kit “Super” with
> YOLO11n-pose and native TensorRT FP16 inference. Squat and overhead-press analytics,
> canonical pose conversion, per-stage benchmarking, platform profiles, and eight automated
> tests are implemented. Persistent primary-person tracking remains on the roadmap.

## What Has Been Built

- Webcam and recorded-video input
- OpenPose COCO 18-joint inference through OpenCV DNN
- Multi-person YOLO11n-pose inference through OpenCV DNN or native TensorRT
- Native TensorRT FP16 execution on Jetson without requiring CUDA-enabled OpenCV
- COCO-17 decoding, person-box NMS, letterbox coordinate restoration, and synthesized neck
- Fresh pose inference for every decoded frame, with no stale-result reuse
- Canonical named-joint pose data independent of the current model
- Monotonic frame IDs and source-relative timestamps for motion calculations
- Optional depth and 3D fields reserved in the pose interface
- Per-stage latency for preprocessing, inference, postprocessing, rendering, and the
  complete processor pipeline
- Real-time squat phase, joint-angle, speed, and repetition analysis in a side panel
- Rear-view overhead-press phase, lockout, velocity, repetition, and JSON analysis
- Input-named JSON summaries with per-repetition metrics
- CPU profiles for macOS and native TensorRT FP16 profiles for Jetson Orin Nano
- Automated tests for rendering, resizing, timing, pose mapping and decoding, profiling,
  and squat/OHP state semantics

Both inference paths publish the same structured `Pose` objects, so analytics and rendering
do not depend on whether keypoints came from OpenPose or YOLO. Measurements remain 2D
image-space estimates rather than anatomical 3D angles or metric velocities.

## Demo

The executable performs pose estimation, draws connected joints, and optionally runs an
exercise analyzer. It can display a live OpenCV window or save an annotated video. At the
end of a run it reports average throughput and stage latency:

```text
[Average Benchmarks] result="output/squat.json" frames=875 FPS: 19.2
| pose_inference: 30.95 ms
| pose_estimator: 37.21 ms
| pipeline: 42.59 ms
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
 OpenPose / YOLO preprocess
 OpenCV DNN / native TensorRT
 model-specific decode + NMS
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
through `FrameContext`. `OpenPoseCocoAdapter` maps OpenPose channels, while
`YoloPoseDecoder` maps multi-person COCO-17 output and synthesizes the missing neck joint.
Both produce the same named canonical joints before rendering or analytics.

On the measured Jetson path, the ARM CPU handles capture/decode, resize, YOLO letterbox
preprocessing, pose/NMS decoding, analytics, rendering, display, and video encoding.
`TensorRtRunner` uploads the FP32 input tensor, executes the internally FP16 engine on the
Ampere GPU/Tensor Cores, downloads the FP32 output, and synchronizes before CPU decoding.

- macOS: OpenCV DNN CPU with Caffe or YOLO11n-pose ONNX
- Jetson production path: native TensorRT with a fixed 640×640 YOLO11n-pose FP16 engine
- Legacy Jetson Caffe path: OpenCV DNN CUDA FP16, only when OpenCV was built with CUDA

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

### Jetson platform

The current measurements were collected on:

- NVIDIA Jetson Orin Nano Developer 8 GB unified memory
- 15 W power mode
- Jetson Linux R36.4.7, CUDA 12 runtime, TensorRT 10.3, OpenCV 4.8.0
- Release build with native TensorRT enabled
- YOLO11n-pose, fixed 640×640 TensorRT engine with internal FP16 execution
- Portrait 1080×1920, 30 FPS MOV input

The installed OpenCV did not include CUDA support. This does not affect the native
TensorRT YOLO path;

### Application benchmark

Both measured configurations had live display and MP4 saving enabled. These are complete
application demonstrations.

| Measurement | OHP | Squat |
| --- | ---: | ---: |
| Processed frames | 589 | 875 |
| Detected repetitions | 6 | 5 |
| Valid analysis frames | 589 | 875 |
| Observed throughput | 16.8 FPS | 19.2 FPS |
| Wall time per frame from FPS | 59.5 ms | 52.1 ms |
| Complete processor pipeline | 40.12 ms | 42.59 ms |
| Pipeline-only capacity | 24.9 FPS | 23.5 FPS |
| Work outside pipeline | 19.4 ms | 9.5 ms |
| TensorRT inference | 29.97 ms | 30.95 ms |
| Inference share of pipeline | 74.7% | 72.7% |
| Complete pose estimator | 35.47 ms | 37.21 ms |
| Pose-estimator share of pipeline | 88.4% | 87.4% |


TensorRT inference is the dominant cost at roughly 30 ms and 73–75% of pipeline time.
CPU preprocessing is the next visible cost at approximately 5.4–6.2 ms. Exercise
analytics costs about 0.01 ms, while skeleton and side-panel rendering remain below
2.7 ms combined.

The gap between pipeline latency and observed FPS comes from work outside the pipeline
timer: video decode, display/event processing, MP4 encoding and writing, per-frame logging,
and loop overhead. A clean performance-contract run should disable display and saving:

```bash
./build/video_engine \
  --config configs/yolo11_squat_jetson.yaml \
  --no-display \
  --no-save
```


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
