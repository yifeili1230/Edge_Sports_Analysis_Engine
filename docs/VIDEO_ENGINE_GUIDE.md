# Video Engine Guide

The executable has one purpose: estimate human pose from webcam or video frames and
draw joints connected by skeleton lines. Optional pose analytics can consume those
joints without changing the inference or rendering contract.

## Runtime flow

```text
Webcam or video file
        |
        v
FrameTimeline
        |
        v
ResizeProcessor
        |
        v
PoseEstimator
        |
        v
Canonical Pose objects
        |
        +--> optional PoseAnalyticsProcessor
        |
        v
SkeletonRenderer
        |
        +--> optional PoseAnalyticsRenderer
        |
        v
Display and/or annotated video
```

`FrameContext` carries the current frame, timestamps, canonical poses, optional analytics
results, and profiling data between processors. It does not contain motion boxes,
centroid tracks, or trails.

## Configuration

The program starts with defaults from `AppConfig`, pre-scans `--config`, loads the
selected YAML-style file, applies command-line overrides, validates the result, and then
resolves the inference platform.

The configuration reader is intentionally flat. It recognizes individual `key: value`
lines; headings such as `pose:`, `squat:`, and `ohp:` are organizational only.

Available profiles:

- `configs/pose.yaml`: OpenPose Caffe on OpenCV CPU.
- `configs/pose_jetson.yaml`: OpenPose Caffe on OpenCV CUDA FP16.
- `configs/caffe_pose_jetson.yaml`: portrait-safe OpenPose Caffe pose profile.
- `configs/caffe_squat_jetson.yaml`: OpenPose Caffe with squat analytics.
- `configs/caffe_ohp_jetson.yaml`: OpenPose Caffe with OHP analytics.
- `configs/squat.yaml`: CPU pose inference with squat analytics.
- `configs/ohp.yaml`: CPU pose inference with overhead-press analytics.
- `configs/yolo11_pose_jetson.yaml`: YOLO11 pose with native TensorRT FP16.
- `configs/yolo11_squat_jetson.yaml`: TensorRT pose with squat analytics.
- `configs/yolo11_ohp_jetson.yaml`: TensorRT pose with overhead-press analytics.

General settings control the source, processing dimensions, aspect-ratio behavior,
display, output video, and `film_side`. `FilmSide` represents eight camera directions
and is part of `IPoseAnalyzer`, so every exercise analyzer can consume the same typed
view. Pose settings select
the model contract, input dimensions, confidence thresholds, maximum people, and
inference backend. Exercise settings control state thresholds, timing, smoothing, and
normalized travel requirements.

## Pose model contracts

### OpenPose COCO

`pose_format: openpose_coco` loads Caffe weights plus a Prototxt network definition.
The output heatmaps are converted into the engine's canonical named-joint schema.

### YOLO11 pose

`pose_format: yolo11_pose` loads a device-built TensorRT engine. Detection confidence,
NMS threshold, and maximum-person settings select the retained COCO-17 poses. The model
adapter maps them into the same canonical schema and synthesizes the neck joint from the
shoulders.

Because downstream code consumes canonical poses, skeleton rendering and exercise
logic do not need model-specific channel indices.

## Running

CPU pose from the webcam:

```bash
./build/video_engine --config configs/pose.yaml
```

Pose from a video without a window or output encoding:

```bash
./build/video_engine \
  --config configs/pose.yaml \
  --source video_source/squat.mov \
  --no-display \
  --no-save
```

Squat analytics:

```bash
./build/video_engine --config configs/squat.yaml
```

Overhead-press analytics:

```bash
./build/video_engine --config configs/ohp.yaml
```

When an analytics profile saves video, the encoded frame includes both the annotated
source image and the same 300-pixel analysis panel shown in the live window.

TensorRT pose on Jetson:

```bash
./build/video_engine --config configs/yolo11_pose_jetson.yaml
```

Supported command-line options:

| Option | Effect |
| --- | --- |
| `--config <path>` | Select a configuration profile |
| `--source webcam\|<video>` | Override the input source |
| `--inference-platform <profile>` | Select `manual`, `cpu`, `jetson`, or `tensorrt` |
| `--exercise none\|squat\|ohp` | Toggle pose analytics |
| `--film-side <view>` | Select one of eight camera views around the subject |
| `--analysis-output <dir>` | Select the JSON summary directory |
| `--display`, `--no-display` | Toggle the GUI |
| `--save <prefix>`, `--no-save` | Toggle annotated video output |
| `-h`, `--help` | Print usage |

The canonical film-side values are `front_view`, `front_left_view`,
`front_right_view`, `left_side_view`, `right_side_view`, `rear_left_view`,
`rear_view`, and `rear_right_view`.

## Output and timing

When saving is enabled, the program chooses the next unused numbered MP4 path for the
configured prefix. Squat mode also writes an input-named JSON session summary.

Runtime logs report preprocessing, inference, decoding, rendering, and aggregate
processor-chain latency. Compare inference settings with the same source, model input
size, and disabled display/output encoding.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The tests cover pose schema mapping, skeleton rendering, frame timing, resizing, YOLO
decoding, and squat/OHP state logic.

## Extension points

New pose models should implement decoding into the canonical `Pose` representation.
New exercise logic should implement `IPoseAnalyzer` and run through
`PoseAnalyticsProcessor`. Rendering remains downstream so analytics can use structured
joint data instead of pixels.
