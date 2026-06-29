# Pose Inference Pipeline and Performance Guide

This document describes the complete pose-estimation path: where each operation runs,
how latency is measured, which settings control performance, and how the macOS CPU and
Jetson Orin Nano CUDA configurations differ.

Two model contracts are supported:

- OpenPose COCO Caffe: one 18-joint pose from heatmap maxima.
- YOLO11n-pose ONNX: multi-person COCO-17 poses with CPU NMS and a synthesized neck
  joint, selected with `pose_format: yolo11_pose`.

For both contracts, OpenCV image operations and output decoding use the CPU. The
YOLO Jetson profile uses a native TensorRT FP16 runner; the legacy Caffe Jetson profile
uses OpenCV's CUDA FP16 DNN target.

## 1. Current Pose Pipeline

The pose pipeline is assembled in `buildPipeline()`:

```text
VideoSource
    |
    v
ResizeProcessor
    |
    v
PoseEstimator
    |
    v
SkeletonRenderer
    |
    v
Display and/or VideoWriter
```

Each frame is passed through the processors in order using a shared `FrameContext`.

## 2. Complete Frame and Inference Data Flow

### 2.1 Capture

`WebcamSource` or `VideoFileSource` reads a frame with `cv::VideoCapture`:

```cpp
capture_.read(frame);
```

The result is a CPU-accessible `cv::Mat`. The current source classes do not explicitly
request Jetson hardware decoding or NVMM zero-copy memory.

### 2.2 Frame resize

`ResizeProcessor` uses:

```cpp
cv::resize(ctx.raw_frame, ctx.processed_frame, cv::Size(width_, height_));
```

This operation currently runs on the CPU. The default processed frame is 640×480.

### 2.3 Pose preprocessing

`PoseEstimator` converts the processed BGR image into a neural-network blob:

```cpp
cv::dnn::blobFromImage(
    ctx.processed_frame,
    1.0 / 255.0,
    cv::Size(input_width_, input_height_),
    cv::Scalar(0, 0, 0),
    false,
    false);
```

This step:

1. Resizes the image to the configured pose input size.
2. Converts pixel values from 0–255 to approximately 0–1.
3. Creates an NCHW floating-point blob.
4. Passes the blob to the network with `net_.setInput()`.

The current preprocessing code runs on the CPU.

### 2.4 Network inference

The expensive operation for OpenCV DNN models is:

```cpp
cv::Mat output = net_.forward();
```

For YOLO TensorRT, the equivalent stage uploads the CPU-created NCHW blob, calls
`enqueueV3()`, and downloads the fixed `[1,56,8400]` result. CUDA synchronization is
included in `pose_inference`, so the reported latency is not an asynchronous illusion.

Its execution device depends on the selected backend and target:

```text
opencv + cpu       -> CPU inference
cuda + cuda_fp16   -> NVIDIA CUDA FP16 inference
```

With the CUDA backend, OpenCV transfers the input into GPU-accessible memory and returns a
CPU-accessible output. Some upload, synchronization, or download work may be included in
the measured `net.forward()` wall-clock time.

### 2.5 Pose postprocessing

The model output is expected to have OpenPose COCO heatmap layout:

```text
N × C × H × W
```

The estimator reads the first 18 keypoint channels. For each channel, `cv::minMaxLoc()`
finds the highest-confidence heatmap location. That location is scaled back into processed
frame coordinates and stored in:

```cpp
ctx.poses
```

The model-specific channel numbers are converted through
`OpenPoseCocoAdapter.hpp` into the canonical `JointId` enum. Downstream code does not need
to know that, for example, OpenPose channel 9 represents the right knee.

Each canonical `Pose` contains:

```text
person_id
frame_id and source_time_seconds
frame dimensions
float 2D pixel coordinates
confidence and validity per named joint
derived person bounding box
optional depth and optional 3D position per joint
```

The pose types are declared in `include/pose/PoseTypes.hpp` and do not depend on OpenCV.
This lets the mathematical `SquatAnalyzer` and future model adapters share one interface.
When `--exercise squat` is enabled, `PoseAnalyticsProcessor` consumes the canonical pose
and publishes per-frame phase, angle, speed, and repetition data before
`PoseAnalyticsRenderer` adds the live side panel. See
[Squat Analytics Guide](SQUAT_ANALYTICS_GUIDE.md).

This postprocessing currently runs on the CPU.

The implementation extracts one maximum per keypoint channel. It does not decode OpenPose
part-affinity fields into separate people, so it should be treated as a simple
single-skeleton estimator.

### 2.6 Skeleton rendering

`SkeletonRenderer` clones the processed frame and draws:

```text
keypoints -> cv::circle()
skeleton links -> cv::line()
```

Rendering currently runs on the CPU.

### 2.7 Display and output

After the pipeline:

- `cv::imshow()` displays the frame when enabled.
- `cv::VideoWriter::write()` saves the frame when enabled.

These operations are outside the measured `pipeline` timer.

## 3. CPU and GPU Boundary

The current Jetson path is not a fully GPU-resident video pipeline:

```text
Capture                    CPU cv::Mat
Resize                     CPU
Blob preprocessing         CPU
Input transfer             CPU -> GPU
DNN forward                GPU with CUDA profile
Output transfer            GPU -> CPU
Heatmap decoding           CPU
Skeleton drawing           CPU
Display/video writing      CPU-facing APIs
```

This design is intentional for the first optimization stage. Network inference dominates
the current CPU runtime, so moving only DNN inference to CUDA offers the highest-value
change without replacing the entire video architecture.

## 4. Latency Metrics

The runtime reports these stages:

| Metric | Included work |
| --- | --- |
| `resize` | `ResizeProcessor::process()` |
| `pose_preprocess` | `blobFromImage()` and `net.setInput()` |
| `pose_inference` | Wall-clock time around `net.forward()` |
| `pose_postprocess` | Heatmap validation, maxima search, and keypoint creation |
| `pose_estimator` | Complete `PoseEstimator::process()` call |
| `skeleton_renderer` | Frame clone and skeleton drawing |
| `pipeline` | All configured processors |
| `FPS` | Average completed frames per second since startup |

Example from the current macOS CPU build:

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

The first frame is often slower because model and backend initialization may be lazy.
Compare warmed-up frames rather than using frame 1.

### What `pose_inference` does not prove

`pose_inference` is host wall-clock time, not a pure GPU-kernel measurement. On CUDA it may
include deferred upload, synchronization, and output download. Use Nsight Systems or CUDA
profiling tools when kernel-level timing is required.

## 5. Pose Configuration

### 5.1 macOS CPU configuration

`configs/pose.yaml`:

```yaml
pose_input_width: 256
pose_input_height: 256
pose_confidence: 0.12
pose_backend: opencv
pose_target: cpu
```

This configuration runs a fresh model inference on every frame using the OpenCV CPU backend.

### 5.2 Jetson Orin Nano configuration

`configs/pose_jetson.yaml`:

```yaml
pose_input_width: 192
pose_input_height: 192
pose_confidence: 0.12
inference_platform: jetson
```

The `jetson` profile resolves to:

```text
pose_backend = cuda
pose_target  = cuda_fp16
```

Orin Nano uses CUDA compute capability 8.7. OpenCV must be built with CUDA, cuDNN, and
`OPENCV_DNN_CUDA=ON`.

`configs/yolo11_pose_jetson.yaml` instead uses:

```yaml
pose_model: models/yolo11n-pose-fp16.engine
pose_format: yolo11_pose
pose_input_width: 640
pose_input_height: 640
pose_confidence: 0.50
inference_platform: tensorrt
```

Build that engine on the target board with `scripts/build_yolo11_tensorrt.sh`.

## 6. Setting Reference

| Setting | Effect | Performance tradeoff |
| --- | --- | --- |
| `width`, `height` | Size of the processed/displayed frame | Affects CPU resize, drawing, display, and output |
| `resize_mode` | `fit` preserves aspect ratio; `stretch` forces dimensions | Stretching can corrupt pose geometry |
| `pose_input_width`, `pose_input_height` | DNN input resolution | Usually the strongest configuration-level speed/accuracy tradeoff |
| `pose_confidence` | Minimum visible-keypoint score | Mostly affects rendering, not network cost |
| `pose_format` | `openpose_coco` or `yolo11_pose` output contract | Selects preprocessing and decoding |
| `pose_detection_confidence` | YOLO person-box threshold | Lower values retain more candidates |
| `pose_nms_threshold` | YOLO box overlap threshold | Controls duplicate-person suppression |
| `pose_max_people` | Maximum YOLO poses after NMS | Bounds downstream CPU work |
| `pose_backend` | OpenCV DNN implementation or native TensorRT | Must be available in the build |
| `pose_target` | CPU, CUDA, CUDA FP16, and other targets | Selects execution hardware/precision |
| `inference_platform` | `cpu`, `jetson`, or `tensorrt` profile | Overrides backend and target after config loading |
| `display` | Enable the OpenCV window | Disable for clean benchmarks |
| `save_output` | Enable video encoding and writing | Disable for clean benchmarks |

The YOLO TensorRT engine in this repository has a fixed 640×640 network input. Reducing
only `width` and `height` may save a little CPU resize/render work, but it does not reduce
TensorRT inference latency because the result is scaled back to 640×640. A true inference
speed change requires exporting and building a separate lower-resolution model engine,
then measuring its accuracy loss.

## 7. Input Resolution and Measurement Cadence

### Input resolution

Reducing the model input from 256×256 to 192×192 reduces input pixels to:

```text
(192 × 192) / (256 × 256) = 56.25%
```

This does not guarantee a 43.75% latency reduction, but convolutional work commonly drops
substantially. Lower resolution can make small or partially occluded body parts less
accurate.

Suggested test points:

```text
192×192  performance-oriented
224×224  middle ground
256×256  current macOS default
368×368  accuracy-oriented and expensive
```

### Measurement cadence

The project deliberately runs a fresh pose inference on every successfully decoded frame.
It does not reuse a previous pose to increase display FPS. Each frame begins with cleared
pose state and receives a monotonic frame ID and source timestamp. If inference produces no
usable keypoints, `pose_measurement_valid` remains false instead of publishing stale data.

## 8. Running Benchmarks

Use a fixed video instead of a webcam so every test receives the same frames.

### macOS CPU

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose.yaml \
  --no-display \
  --no-save
```

### Jetson Orin Nano CUDA FP16

```bash
./build/video_engine \
  --source video_source/squat.mov \
  --config configs/pose_jetson.yaml \
  --no-display \
  --no-save
```

For a controlled CPU-versus-CUDA comparison, use the same input dimensions on both
platforms. For a practical deployment comparison, use each platform's recommended
configuration and clearly record the settings.

Collect at least:

```text
OpenCV version and build options
model and model input size
pose_inference latency after warm-up
pipeline latency
average FPS
Orin power mode and tegrastats output
```

## 9. How to Interpret Results

### `pose_inference` dominates

This is expected with the current OpenPose model. Prioritize:

1. CUDA FP16 on Orin Nano.
2. Lower pose input resolution.
3. A lighter TensorRT pose model.

### `pose_preprocess` becomes significant

This can happen after inference becomes much faster. Consider GPU preprocessing or direct
TensorRT bindings only after measurement shows that CPU preprocessing and transfer are
material bottlenecks.

### `resize` or rendering becomes significant

Consider `cv::cuda::GpuMat` operations or a GPU-resident pipeline only if their measured
cost justifies the added memory-management complexity.

### FPS is lower than expected from pipeline latency

The pipeline timer excludes capture, display, and video writing. Disable display and saving
to isolate processing, then profile those external stages separately if necessary.

## 10. Recommended Optimization Order

1. Measure with a fixed video, Release build, `--no-display`, and `--no-save`.
2. Verify OpenCV DNN CUDA FP16 on Orin Nano.
3. Tune model input size while preserving per-frame inference.
4. Replace the heavy Caffe model if inference still dominates.
5. Add hardware decode or GPU preprocessing only after profiling.
6. Add bounded asynchronous capture/inference/output queues if throughput is limited by
   serial execution.
7. Consider NVMM/CUDA zero-copy only when memory transfers are a measured bottleneck.

The goal is not to move every operation to a tensor. The useful target is to keep large
image data on the GPU when it saves measurable time and return only small keypoint results
to the CPU.

## 11. Replacing the Model

The current parser requires OpenPose COCO heatmaps with at least 18 keypoint channels.
MoveNet, YOLO Pose, `trt_pose`, and other architectures have different output formats and
cannot be selected by changing only the model path.

For a TensorRT model, add a separate processor:

```text
TensorRTPoseEstimator
```

It should:

1. Accept `ctx.processed_frame`.
2. Run model-specific preprocessing and TensorRT inference.
3. Decode the model's output format.
4. Convert keypoints into the project's `Pose` and `PoseKeypoint` types.
5. Store results in `ctx.poses`.

`SkeletonRenderer` can then remain unchanged.

## 12. Related Documentation

- [Platform Run Guide](PLATFORM_RUN_GUIDE.md)
- [Video Engine Guide](VIDEO_ENGINE_GUIDE.md)
- [Jetson Pipeline Optimization Cheat Sheet](../reference/jetson-pipeline-cheatsheet.html)
