# Real-Time Video, Pose, and Exercise Analytics Engine

A modular C++17/OpenCV video-processing engine designed to grow from classical computer
vision into reliable real-time human-pose and exercise analytics on macOS and NVIDIA
Jetson Orin Nano.

The project currently supports motion detection, centroid-based object tracking, OpenPose
COCO skeleton inference, per-stage performance profiling, reliable per-frame measurement
time, and a model-independent pose data contract. Its next milestone is a pure
mathematical squat analyzer, followed by a faster multi-person ONNX/TensorRT pose model,
persistent primary-person tracking, and Orin-specific optimization.

This repository is both an engineering project and a documented learning journey. It
focuses on the parts that make real-time analytics trustworthy: explicit CPU/GPU
boundaries, reproducible timing, fresh measurements instead of cached display results,
stable data contracts, tests, and measured optimization.

---

## Project at a Glance

| Area | Current status |
| --- | --- |
| Video input | Webcam and recorded video |
| Classical CV | Motion detection, merged regions, centroid tracking, trajectory trails |
| Pose inference | OpenPose COCO 18-joint Caffe model through OpenCV DNN |
| macOS execution | OpenCV DNN on CPU |
| Jetson execution | CUDA FP16 profile prepared for Jetson Orin Nano |
| Pose timing | Preprocess, inference, postprocess, processor, and total pipeline latency |
| Measurement timeline | Monotonic frame ID and source-relative timestamp |
| Pose freshness | Fresh inference on every decoded frame; no cached pose reuse |
| Canonical pose schema | Named joints, float 2D positions, optional depth/3D, person ID, bbox |
| Model adapters | Explicit OpenPose COCO channel-to-joint adapter |
| Tests | Motion, tracking, rendering, timeline, and pose-schema tests |
| Exercise analytics | Interface and squat semantics designed; implementation is next |
| Multi-person pose tracking | Planned; current OpenPose decoder produces one simple skeleton |
| Orin performance target | At least 20 FPS and less than 100 ms end-to-end latency |

The CUDA/TensorRT performance target has not yet been validated on the physical Orin Nano.
The current measured macOS result is a CPU baseline, not a Jetson performance claim.

---

## Why This Project Exists

Many pose demos stop after drawing a skeleton. Exercise analytics needs a stricter
foundation:

- Every result must belong to a known input frame.
- Velocity must use source time, not processing FPS.
- Failed inference must remain invalid instead of silently reusing an old pose.
- Model-specific joint indices must not leak into analytics code.
- Tracking must preserve the selected person when distractors enter.
- Performance optimization must be based on measured stage latency.
- Two-dimensional and future depth/three-dimensional measurements need one extensible
  interface.

The engine is being built around those constraints so later squat analysis, trained
exercise models, and Jetson optimization can share the same reliable core.

---

## Key Engineering Achievements

### 1. Modular processor pipeline

Video logic is organized as small `IFrameProcessor` components:

```text
VideoSource
    |
    v
ResizeProcessor
    |
    +--> MotionDetector --> ObjectTracker --> OverlayRenderer
    |
    +--> PoseEstimator ---------------------> SkeletonRenderer
```

Each processor reads and writes a shared `FrameContext`. Pipelines are assembled in
`buildPipeline()` without placing all algorithms inside the main loop.

### 2. Detailed performance profiling

The engine reports both processor-level and pose-internal latency:

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

This makes the dominant cost visible. On the current macOS CPU build, inference takes
roughly 250–330 ms after warm-up while resize, postprocessing, and rendering are small.
That evidence prioritizes model/backend work over premature GPU rewrites of inexpensive
stages.

`pose_inference` is wall-clock time around `net.forward()`. With CUDA it may include
deferred upload, synchronization, or output download; use NVIDIA profiling tools when pure
kernel timing is required.

### 3. Reliable per-frame measurement timeline

The original display-oriented frame-reuse path was removed. Every successfully decoded
frame now starts a new measurement:

```text
read source frame
    |
    v
FrameTimeline::beginFrame()
    |
    +--> assign increasing frame_id
    +--> assign source_time_seconds
    +--> clear previous poses
    +--> reset inference and validity flags
    |
    v
run fresh pose inference
```

Live camera sources use a monotonic clock. Recorded videos use media timestamps with a
frame-index/FPS fallback. Duplicate or decreasing source times are rejected.

Every frame exposes:

```cpp
std::uint64_t frame_id;
double source_time_seconds;
bool pose_inference_ran;
bool pose_measurement_valid;
```

This contract prevents false zero-velocity samples and false velocity spikes caused by
reusing an old pose.

### 4. Model-independent pose schema

Raw OpenPose channel numbers are converted into a canonical, OpenCV-independent data
model:

```cpp
enum class JointId {
    Nose,
    Neck,
    RightShoulder,
    RightElbow,
    RightWrist,
    LeftShoulder,
    LeftElbow,
    LeftWrist,
    RightHip,
    RightKnee,
    RightAnkle,
    LeftHip,
    LeftKnee,
    LeftAnkle,
    RightEye,
    LeftEye,
    RightEar,
    LeftEar,
};
```

Each `Pose` carries:

- A person ID placeholder for future tracking
- Frame ID and source timestamp
- Frame dimensions
- A derived person bounding box
- A fixed array of named joints
- Floating-point 2D pixel coordinates
- Confidence and validity per joint
- Optional depth
- Optional 3D position

The current `OpenPoseCocoAdapter` maps all 18 Caffe output channels into this schema.
Future YOLO-pose, RTMPose, MoveNet, or TensorRT outputs can implement their own adapters
without changing rendering, tracking, or exercise-analysis consumers.

### 5. Cross-platform inference profiles

The project keeps one application architecture while selecting platform-specific
inference:

```text
macOS
OpenCV DNN backend + CPU target

Jetson Orin Nano
OpenCV DNN CUDA backend + CUDA FP16 target

Planned
shared ONNX model semantics + TensorRT FP16 execution on Orin
```

### 6. Testable design

The current test suite covers:

| Test | What it protects |
| --- | --- |
| `motion_detector_test` | Motion contour detection and primary-region behavior |
| `object_tracker_test` | Unique centroid tracker IDs and aging behavior |
| `skeleton_renderer_test` | Rendering from canonical named-joint pose data |
| `frame_timeline_test` | Increasing IDs/time and no stale pose state |
| `pose_schema_test` | Complete unique joint mapping, float precision, optional depth |

---

## Complete Pose Inference Workflow

### Step 1: Capture

`WebcamSource` or `VideoFileSource` returns a CPU-accessible `cv::Mat`.

### Step 2: Start a fresh measurement

`FrameTimeline` assigns the frame identity and source-relative time, clears previous pose
results, and resets measurement flags.

### Step 3: Resize the processed frame

`ResizeProcessor` uses CPU `cv::resize()` to produce the configured processing resolution,
normally 640×480.

### Step 4: Build the DNN input

`PoseEstimator` calls `cv::dnn::blobFromImage()` to:

- Resize to the model input resolution
- Scale pixels by `1 / 255`
- Build a floating-point NCHW blob
- Pass the blob to the network

This preprocessing currently runs on CPU.

### Step 5: Run inference

```cpp
cv::Mat output = net_.forward();
```

The selected backend determines execution:

```text
opencv + cpu       -> CPU inference
cuda + cuda_fp16   -> NVIDIA CUDA FP16 inference
```

### Step 6: Decode model output

The current Caffe model produces OpenPose COCO heatmaps. `cv::minMaxLoc()` finds one peak
for each of the first 18 channels.

### Step 7: Convert to canonical joints

`OpenPoseCocoAdapter` maps model channels into `JointId`. The estimator publishes a
canonical `Pose` with float coordinates, confidence, validity, time, frame ID, and bbox.

### Step 8: Render or analyze

`SkeletonRenderer` is currently the first consumer. Future consumers will include:

```text
PrimarySubjectTracker
PoseAnalyticsProcessor
SquatAnalyzer
MetricsRenderer
RepEventSink
```

### Current CPU/GPU boundary

```text
Capture                     CPU-facing cv::Mat
Frame resize                CPU
Blob preprocessing          CPU
Input transfer              CPU -> GPU when CUDA is selected
DNN forward                 CPU or CUDA FP16
Output transfer             GPU -> CPU when CUDA is selected
Heatmap decoding            CPU
Canonical pose creation     CPU
Skeleton rendering          CPU
Display/video output        CPU-facing APIs
```

The first optimization target is inference because it dominates the measured baseline.
Hardware decode, GPU preprocessing, and zero-copy are later steps that should only be
added when profiling shows a material benefit.

---

## Available Pipelines

### Motion detection

```text
ResizeProcessor
→ MotionDetector
→ OverlayRenderer
```

The motion detector uses frame differencing, thresholding, morphology, contour filtering,
and region merging. Configuration controls sensitivity, minimum area, merge padding, and
maximum detections.

### Motion tracking

```text
ResizeProcessor
→ MotionDetector
→ ObjectTracker
→ OverlayRenderer
```

The tracker associates motion boxes using centroid distance, assigns IDs, expires missing
tracks, and stores trajectory trails.

This is not yet the planned pose/person tracker. The future primary-subject tracker will
operate on multi-person pose detections and preserve one chosen identity through
distractors and brief occlusion.

### Pose estimation

```text
ResizeProcessor
→ PoseEstimator
→ SkeletonRenderer
```

The current estimator produces one simple skeleton by selecting the strongest peak in each
OpenPose heatmap. It does not yet decode part-affinity fields into multiple people.

---

## Exercise Analytics Design

The first exercise module will analyze a fixed side or approximately 45-degree side view
using monocular 2D keypoints. Optional depth and 3D fields are already reserved in the
pose schema.

### Interface boundary

The planned design separates pure mathematics from the video framework:

```text
Pure analysis layer
IPoseAnalyzer
└── SquatAnalyzer

Video adapter layer
PoseAnalyticsProcessor
├── reads canonical Pose
├── invokes IPoseAnalyzer
└── writes per-frame metrics and rep events
```

The pure analyzer will not depend on OpenCV, cameras, or a particular pose model. A future
trained exercise classifier can implement the same `IPoseAnalyzer` interface.

### Planned squat state machine

```text
Standing:
    knee angle > 160°

Descending:
    knee angle decreases
    hip moves downward

Bottom:
    knee angle < 100°

Ascending:
    knee angle increases
    hip moves upward

Rep complete:
    return to knee angle > 160°
```

Only the complete ordered cycle counts as a repetition. Thresholds will be configurable.

### Planned per-rep outputs

- Rep count
- Current squat phase
- Descent time
- Ascent time
- Total rep duration
- Minimum knee angle
- Hip and knee angle trajectories
- Average normalized speed
- Peak normalized speed
- Confidence/validity state

### 2D and 3D interpretation

The initial analyzer will compute projected 2D joint angles:

```text
u = joint_before - center_joint
v = joint_after  - center_joint

angle = acos(dot(u, v) / (norm(u) * norm(v)))
```

Uncalibrated velocity is measured in pixels per second:

```text
velocity = (current_position - previous_position) / delta_time
```

Body-scale normalization can improve comparisons in a fixed camera setup. True metric
speed or reliable 3D anatomical angles require calibration, a known scale, stereo/depth
input, or a 3D pose model.

---

## Primary-Subject Tracking Design

When a future multi-person pose model is integrated, the tracker will:

1. Initially select the largest person near the frame center.
2. Lock the selected identity.
3. Avoid switching when another person enters.
4. Associate future observations using predicted centroid, bbox overlap, keypoint
   similarity, size, and center prior.
5. Tolerate a configurable number of missing frames.
6. Reacquire only after the original target is considered lost.

Planned tests include crossings, temporary occlusion, a larger distractor entering, target
exit/re-entry, and low-confidence pose frames.

---

## Platform Support

| Platform | Config | Backend | Target | Model input | Fresh inference |
| --- | --- | --- | --- | --- | --- |
| macOS | `configs/pose.yaml` | OpenCV DNN | CPU | 256×256 | Every frame |
| Jetson Orin Nano | `configs/pose_jetson.yaml` | OpenCV DNN CUDA | CUDA FP16 | 192×192 | Every frame |

Build separately on each platform. A macOS executable cannot run on Jetson Linux even
when both machines use ARM processors.

### Jetson Orin Nano requirements

The target uses JetPack 6 and CUDA compute capability 8.7. OpenCV must be built with:

```text
WITH_CUDA=ON
WITH_CUDNN=ON
WITH_CUBLAS=ON
OPENCV_DNN_CUDA=ON
CUDA_ARCH_BIN=8.7
```

Installing CUDA alone does not enable OpenCV's CUDA DNN backend. Verify the OpenCV build:

```bash
opencv_version --verbose
```

The repository has been compiled and tested on the current macOS environment. The Orin
configuration is prepared but must still be validated and benchmarked on the physical
device.

---

## Requirements

- CMake 3.16 or newer
- C++17 compiler
- OpenCV components:
  - `core`
  - `imgproc`
  - `highgui`
  - `videoio`
  - `dnn`
- OpenPose COCO model files for the current pose pipeline
- CUDA/cuDNN-enabled OpenCV for Jetson GPU execution

---

## Model Setup

Place the current OpenPose COCO model files under `models/`:

```text
models/
├── pose_deploy_linevec.prototxt
└── pose_iter_440000.caffemodel
```

The model is not replaced by the canonical schema. The schema separates model output from
downstream consumers so the model can be replaced later.

---

## Build

Use a Release build for meaningful performance measurements:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## Run

Run commands from the project root so model and configuration paths resolve correctly.

### Default pose mode

```bash
./build/video_engine
```

### macOS CPU pose inference

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose.yaml
```

### Jetson Orin Nano CUDA FP16 pose inference

```bash
./build/video_engine \
  --pipeline pose \
  --config configs/pose_jetson.yaml
```

### Pose inference on a recorded video

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose.yaml
```

### Headless benchmark

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose_jetson.yaml \
  --no-display \
  --no-save
```

### Motion detection

```bash
./build/video_engine \
  --source webcam \
  --pipeline motion \
  --config configs/motion.yaml
```

### Motion tracking

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline tracking \
  --config configs/tracking.yaml
```

---

## Command-Line Options

| Option | Purpose |
| --- | --- |
| `--config <path>` | Select the YAML-style configuration file |
| `--source webcam` | Use the default camera |
| `--source <path>` | Read a video file |
| `--pipeline motion` | Run motion detection |
| `--pipeline tracking` | Run motion detection and centroid tracking |
| `--pipeline pose` | Run pose estimation |
| `--inference-platform manual` | Keep backend/target from the configuration |
| `--inference-platform cpu` | Select OpenCV DNN on CPU |
| `--inference-platform jetson` | Select CUDA FP16 |
| `--inference-platform jetson-orin-nano` | Explicit alias for the Jetson profile |
| `--display` | Enable the OpenCV display window |
| `--no-display` | Disable the window for SSH or benchmarking |
| `--save <prefix>` | Enable output video with the supplied path prefix |
| `--no-save` | Disable output video |

The selected configuration file is loaded before supported command-line overrides are
applied.

---

## Configuration Files

### `configs/pose.yaml`

Compatibility-oriented CPU configuration:

```yaml
source: webcam
width: 640
height: 480
display: true
save_output: true
pose:
  pose_input_width: 256
  pose_input_height: 256
  pose_confidence: 0.12
  pose_backend: opencv
  pose_target: cpu
```

### `configs/pose_jetson.yaml`

Jetson Orin Nano CUDA FP16 configuration:

```yaml
source: webcam
width: 640
height: 480
display: true
save_output: false
pose:
  pose_input_width: 192
  pose_input_height: 192
  pose_confidence: 0.12
  inference_platform: jetson
```

### Motion and tracking

- `configs/motion.yaml` controls threshold, minimum contour area, region merge padding, and
  maximum detections.
- `configs/tracking.yaml` adds centroid match distance, maximum track age, and trail length.

The parser intentionally supports a small YAML-like subset. Section labels such as `pose:`
are ignored; scalar key/value lines are read.

---

## Performance Methodology

For reproducible comparisons:

1. Build in Release mode.
2. Use the same recorded video.
3. Use the same model input size when comparing CPU and CUDA directly.
4. Disable display and saving.
5. Ignore initial warm-up frames.
6. Record `pose_inference`, `pose_estimator`, `pipeline`, and FPS.
7. On Orin, also record `tegrastats`, power mode, temperature, and clocks.

Baseline command:

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --pipeline pose \
  --config configs/pose.yaml \
  --no-display \
  --no-save
```

The `pipeline` metric excludes source reading, display, and video writing. If total
end-to-end latency differs from pipeline latency, profile those external stages separately.

---

## Repository Structure

```text
.
├── CMakeLists.txt
├── README.md
├── configs/
│   ├── motion.yaml
│   ├── pose.yaml
│   ├── pose_jetson.yaml
│   └── tracking.yaml
├── docs/
│   ├── PLATFORM_RUN_GUIDE.md
│   ├── POSE_ANALYTICS_DECISION_MAP.md
│   ├── POSE_INFERENCE_GUIDE.md
│   └── VIDEO_ENGINE_GUIDE.md
├── include/
│   ├── core/
│   │   ├── FrameContext.hpp
│   │   ├── FrameTimeline.hpp
│   │   ├── IFrameProcessor.hpp
│   │   ├── IVideoSource.hpp
│   │   └── Pipeline.hpp
│   ├── pose/
│   │   ├── OpenPoseCocoAdapter.hpp
│   │   └── PoseTypes.hpp
│   ├── processors/
│   ├── sources/
│   └── utils/
├── lessons/
│   ├── 0001-cpu-gpu-dataflow.html
│   ├── 0002-pose-results-to-motion-metrics.html
│   └── 0003-reliable-pose-contract.html
├── models/
├── reference/
├── src/
│   └── main.cpp
├── tests/
└── video_source/
```

---

## Current Limitations

- The OpenPose Caffe model is large and slow on CPU.
- The current decoder extracts one strongest point per heatmap and does not perform true
  multi-person part-affinity-field decoding.
- Persistent pose/person tracking is not implemented yet.
- The squat analyzer and analytics processor are designed but not implemented yet.
- Two-dimensional angles are camera-view projections, not guaranteed anatomical 3D angles.
- Metric speed requires calibration, a known scale, depth, or 3D pose.
- Video capture currently returns CPU-accessible `cv::Mat`; hardware decode and zero-copy
  are not implemented.
- The Orin Nano CUDA profile has not yet been benchmarked on the physical device.

---

## Roadmap

The canonical roadmap is maintained in
[Pose Analytics and Orin Optimization Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md).

### Completed decisions and foundations

- Input geometry: monocular camera/video, side-view squat, optional future depth
- Primary-subject selection and reacquisition policy
- Orin target: 20 FPS and less than 100 ms latency
- Squat repetition semantics and configurable thresholds
- Pure mathematical analyzer interface boundary
- Shared ONNX semantics across macOS and Orin
- Reliable per-frame timeline with no cached pose reuse
- Canonical pose and optional depth/3D schema

### Next implementation: pure mathematical squat analyzer

Implement:

- Confidence gating
- 2D joint-angle math
- Body-scale-normalized motion
- Velocity from source time
- Trajectory smoothing
- Hysteretic squat phases
- Rep-completion events
- Invalid-observation behavior
- Synthetic sequence tests

### Faster pose model research

Compare current licensable multi-person ONNX candidates such as:

- Small YOLO-pose variants
- RTMPose-based pipelines
- MoveNet MultiPose
- Other exportable models with documented TensorRT support

Evaluation criteria:

- Keypoint accuracy for side-view squats
- Multi-person output
- Model license
- ONNX export reliability
- OpenCV DNN compatibility on macOS
- TensorRT FP16 compatibility on Orin
- Pre/postprocessing complexity
- Measured latency rather than headline FPS

### Persistent primary-person tracking

Implement and test identity association using:

- Predicted centroid
- Bounding-box overlap
- Keypoint similarity
- Person size
- Center prior
- Lost-frame timeout

### Orin Nano optimization

After correctness:

- TensorRT FP16 and eligible INT8
- Tensor Core utilization
- Power mode and thermal behavior
- Hardware decode
- CPU/GPU transfer profiling
- Bounded asynchronous capture/inference/output
- GPU preprocessing or NVMM zero-copy only when measured

---

## Documentation

### Engineering guides

- [Video Engine Guide](docs/VIDEO_ENGINE_GUIDE.md) — complete architecture and extension
  reference
- [Pose Inference Guide](docs/POSE_INFERENCE_GUIDE.md) — inference stages, timing, settings,
  and CPU/GPU boundaries
- [macOS and Jetson Orin Nano Run Guide](docs/PLATFORM_RUN_GUIDE.md) — platform setup and
  commands
- [Pose Analytics Decision Map](docs/POSE_ANALYTICS_DECISION_MAP.md) — resolved decisions,
  dependencies, and remaining work

### Learning sequence

1. [CPU/GPU Data Flow](lessons/0001-cpu-gpu-dataflow.html)
2. [Pose Results to Motion Metrics](lessons/0002-pose-results-to-motion-metrics.html)
3. [Reliable Per-Frame Pose Contract](lessons/0003-reliable-pose-contract.html)

### Quick references

- [Jetson Optimization Cheat Sheet](reference/jetson-pipeline-cheatsheet.html)
- [Pose Analytics Math Cheat Sheet](reference/pose-analytics-cheatsheet.html)

---

## Portfolio Summary

This project demonstrates:

- Modern C++17 interface-driven design
- OpenCV video capture, image processing, DNN inference, rendering, and encoding
- Configurable processor pipelines
- Classical motion detection and tracking
- Human-pose estimation and canonical keypoint modeling
- CPU/CUDA deployment boundaries
- Jetson Orin Nano and TensorRT-oriented architecture
- Source-time-aware measurement design
- Performance instrumentation and evidence-driven optimization
- Extensible interfaces for deterministic and learned exercise analytics
- Tests that protect time, identity, and data-contract assumptions
- Technical documentation suitable for handoff, learning, and deployment

The central engineering lesson is simple: a skeleton overlay is not yet an analytics
system. Reliable analytics begins with fresh measurements, stable time, explicit validity,
model-independent data, controlled identity, and measurable performance.
