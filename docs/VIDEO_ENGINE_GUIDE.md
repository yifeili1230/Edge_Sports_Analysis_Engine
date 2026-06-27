# Video Engine Guide

This document explains how the video engine works, what each part of the code does, how to run it, and how to debug or modify it. It is written for someone who may not have prior experience with OpenCV, C++ video pipelines, or AI pose estimation.

For focused deployment commands and inference performance details, see
[Platform Run Guide](PLATFORM_RUN_GUIDE.md) and
[Pose Inference Guide](POSE_INFERENCE_GUIDE.md).

## 1. What This Project Does

This project is a small C++17/OpenCV video processing engine.

It can read frames from:

- A webcam
- A video file

Then it can process those frames with one of three pipelines:

- `motion`: detects moving regions and draws boxes.
- `tracking`: detects moving regions, assigns object IDs, and draws motion trails.
- `pose`: detects human body keypoints and draws a skeleton.

The program is built around a simple idea:

```text
video source -> frame context -> pipeline processors -> display/save/log
```

Each input frame is placed into a shared `FrameContext`. A pipeline then runs several processors in order. Each processor reads from and writes to the same `FrameContext`.

## 2. Project Layout

The important folders are:

```text
openCV/
  CMakeLists.txt
  README.md
  configs/
    motion.yaml
    tracking.yaml
    pose.yaml
  docs/
    VIDEO_ENGINE_GUIDE.md
  include/
    core/
    processors/
    sources/
    utils/
  models/
    pose_deploy_linevec.prototxt
    pose_iter_440000.caffemodel
  src/
    main.cpp
  tests/
```

The main executable is built from:

```text
src/main.cpp
```

Most of the actual logic lives in header files under:

```text
include/
```

This is common in small C++ demo projects: the main program wires components together, while the component logic is implemented in reusable classes.

## 3. The Main Program

The entry point is:

```cpp
int main(int argc, char** argv)
```

It performs these high-level steps:

```text
1. Parse command-line arguments and config file.
2. Create the video source.
3. Create the selected processing pipeline.
4. Read frames in a loop.
5. Run the pipeline on each frame.
6. Display and optionally save the processed frame.
7. Log FPS and pipeline latency.
8. Exit when the input ends or ESC is pressed.
```

In code, the main flow is:

```cpp
auto config = parseArgs(argc, argv);
auto source = createSource(config);
auto pipeline = buildPipeline(config);

while (true) {
    cv::Mat frame;
    if (!source->read(frame)) {
        break;
    }

    ctx.raw_frame = frame;
    pipeline->run(ctx);
    cv::imshow("Video Engine", ctx.processed_frame);
}
```

The real code has more details, such as clearing old frame data, measuring latency, and writing output video.

## 4. Configuration Flow

The program uses `AppConfig` in `src/main.cpp` to store runtime options.

Important config fields include:

```cpp
std::string source;
std::string pipeline;
int width;
int height;
int threshold;
int min_area;
std::string pose_model;
std::string pose_config;
int pose_input_width;
int pose_input_height;
float pose_confidence;
std::string inference_platform;
std::string pose_backend;
std::string pose_target;
```

The config is built in the following stages:

```text
1. Start with default values in `AppConfig`.
2. Pre-scan `--config` to select the configuration file.
3. Read the selected YAML-style configuration file.
4. Apply supported command-line overrides.
```

For example:

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml
```

This means:

- `--pipeline pose`: use pose mode.
- `--config configs/pose.yaml`: load pose-specific settings.

The config file parser is intentionally simple. It reads lines like:

```yaml
width: 640
height: 480
pose_backend: opencv
```

It ignores section names such as:

```yaml
pose:
```

This works because the parser only cares about key/value lines with actual values after `:`.

## 5. Video Sources

The function `createSource()` chooses where frames come from.

```cpp
std::unique_ptr<video_engine::IVideoSource> createSource(const AppConfig& config)
```

If the config says:

```yaml
source: webcam
```

then the program creates:

```cpp
WebcamSource(0)
```

This opens camera index `0`, usually the built-in or default webcam.

If the config says:

```yaml
source: videos/demo.mp4
```

then the program creates:

```cpp
VideoFileSource("videos/demo.mp4")
```

The common interface is:

```cpp
class IVideoSource {
public:
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual double fps() const = 0;
};
```

This lets the rest of the program treat webcam input and video-file input the same way.

## 6. FrameContext

`FrameContext` is the shared data object passed through the pipeline.

It is defined in:

```text
include/core/FrameContext.hpp
```

It contains:

```cpp
cv::Mat raw_frame;
cv::Mat processed_frame;
std::vector<Detection> detections;
std::vector<Pose> poses;
std::vector<std::vector<cv::Point>> trails;
std::vector<std::string> stage_names;
std::vector<double> stage_latencies_ms;
```

Think of `FrameContext` as a backpack carried by each video frame.

Each processor can:

- Read data from the backpack.
- Add new data to the backpack.
- Modify existing data.

For example:

- `ResizeProcessor` reads `raw_frame` and writes `processed_frame`.
- `MotionDetector` reads `processed_frame` and writes `detections`.
- `ObjectTracker` reads `detections` and writes updated IDs and `trails`.
- `PoseEstimator` reads `processed_frame` and writes `poses`.
- `SkeletonRenderer` reads `poses` and draws onto `processed_frame`.

## 7. Pipeline System

The pipeline is defined in:

```text
include/core/Pipeline.hpp
```

It stores a list of processors:

```cpp
std::vector<std::unique_ptr<IFrameProcessor>> processors_;
```

Each processor implements this interface:

```cpp
class IFrameProcessor {
public:
    virtual void process(FrameContext& ctx) = 0;
    virtual std::string name() const = 0;
};
```

When the pipeline runs:

```cpp
void run(FrameContext& ctx) {
    for (const auto& processor : processors_) {
        processor->process(ctx);
    }
}
```

This means processors run in the exact order they were added.

Order matters.

For example, pose mode is built as:

```text
ResizeProcessor -> PoseEstimator -> SkeletonRenderer
```

If `SkeletonRenderer` ran before `PoseEstimator`, there would be no pose keypoints to draw.

## 8. Available Pipelines

The selected pipeline is created in:

```cpp
buildPipeline(const AppConfig& config)
```

### 8.1 Motion Pipeline

The motion pipeline is:

```text
ResizeProcessor -> MotionDetector -> OverlayRenderer
```

Purpose:

```text
Input frame
-> resize
-> detect moving blobs
-> draw bounding boxes
```

Run it with:

```bash
./build/video_engine --pipeline motion --config configs/motion.yaml
```

### 8.2 Tracking Pipeline

The tracking pipeline is:

```text
ResizeProcessor -> MotionDetector -> ObjectTracker -> OverlayRenderer
```

Purpose:

```text
Input frame
-> resize
-> detect moving blobs
-> assign stable-ish object IDs
-> draw boxes, IDs, and trails
```

Run it with:

```bash
./build/video_engine --pipeline tracking --config configs/tracking.yaml
```

### 8.3 Pose Pipeline

The pose pipeline is:

```text
ResizeProcessor -> PoseEstimator -> SkeletonRenderer
```

Purpose:

```text
Input frame
-> resize
-> run human-pose model
-> extract body keypoints
-> draw skeleton
```

Run it with:

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml
```

## 9. Processor Details

### 9.1 ResizeProcessor

File:

```text
include/processors/ResizeProcessor.hpp
```

Purpose:

```text
Normalize every input frame to the configured width and height.
```

It reads:

```cpp
ctx.raw_frame
```

It writes:

```cpp
ctx.processed_frame
```

This is usually the first processor in every pipeline.

### 9.2 MotionDetector

File:

```text
include/processors/MotionDetector.hpp
```

Purpose:

```text
Detect areas that changed between the previous frame and the current frame.
```

Basic algorithm:

```text
1. Convert current frame to grayscale.
2. Blur it to reduce noise.
3. Compare it with the previous grayscale frame.
4. Threshold the difference image.
5. Erode and dilate to clean noise.
6. Find contours.
7. Convert contours into bounding boxes.
8. Merge nearby boxes.
9. Keep the largest N boxes.
```

Important config:

```yaml
threshold: 25
min_area: 500
merge_padding: 24
max_detections: 0
```

Meaning:

- `threshold`: how different pixels must be before they count as motion.
- `min_area`: removes tiny noisy regions.
- `merge_padding`: merges nearby boxes into a larger box.
- `max_detections`: limits the number of output boxes. `0` means keep all.

### 9.3 ObjectTracker

File:

```text
include/processors/ObjectTracker.hpp
```

Purpose:

```text
Give motion boxes track IDs and keep short motion trails.
```

This is not a deep-learning tracker. It is a simple centroid tracker.

Basic algorithm:

```text
1. Compute the center point of each detection box.
2. Compare new centers with active track centers.
3. Match closest pairs within max_distance.
4. Reuse the matched track ID.
5. Create new IDs for unmatched detections.
6. Age unmatched tracks.
7. Delete tracks older than max_age.
8. Store recent center points as trails.
```

Important config:

```yaml
max_distance: 80
max_age: 8
trail_length: 25
```

Meaning:

- `max_distance`: how far a detection may move and still be treated as the same object.
- `max_age`: how many missed frames before a track is deleted.
- `trail_length`: how many old center points to draw.

### 9.4 OverlayRenderer

File:

```text
include/processors/OverlayRenderer.hpp
```

Purpose:

```text
Draw regular motion/tracking visualization.
```

It draws:

- Green detection boxes
- Track IDs
- Blue trails

It is used in:

```text
motion
tracking
```

It is not used in:

```text
pose
```

Pose mode uses `SkeletonRenderer` instead.

### 9.5 PoseEstimator

File:

```text
include/processors/PoseEstimator.hpp
```

Purpose:

```text
Run the human-pose neural network and extract body keypoints.
```

Current model:

```text
OpenPose COCO 18-keypoint Caffe model
```

Files:

```text
models/pose_deploy_linevec.prototxt
models/pose_iter_440000.caffemodel
```

The `.prototxt` file describes the neural network structure.

The `.caffemodel` file stores trained weights.

The core inference call is:

```cpp
cv::Mat output = net_.forward();
```

The estimator records `pose_preprocess`, `pose_inference`, and `pose_postprocess`
latencies. See [Pose Inference Guide](POSE_INFERENCE_GUIDE.md) for the complete CPU/GPU
data flow and timing definitions.

The output is expected to be OpenPose COCO heatmaps:

```text
N x C x H x W
```

The code expects at least 18 keypoint channels. It then finds the maximum value in each keypoint heatmap.

This produces one point per body part, such as:

- Head/nose
- Neck
- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

Important config:

```yaml
pose_model: models/pose_iter_440000.caffemodel
pose_config: models/pose_deploy_linevec.prototxt
pose_input_width: 256
pose_input_height: 256
pose_confidence: 0.12
pose_backend: opencv
pose_target: cpu
```

Meaning:

- `pose_model`: neural network weights.
- `pose_config`: neural network definition.
- `pose_input_width`: model input width.
- `pose_input_height`: model input height.
- `pose_confidence`: minimum keypoint confidence required for rendering.
- `pose_backend`: OpenCV DNN backend.
- `pose_target`: target hardware for OpenCV DNN.

### 9.6 SkeletonRenderer

File:

```text
include/processors/SkeletonRenderer.hpp
```

Purpose:

```text
Draw the keypoints and connect them into a human skeleton.
```

It reads:

```cpp
ctx.poses
```

It draws onto:

```cpp
ctx.processed_frame
```

The skeleton connections are listed in:

```cpp
kSkeletonPairs
```

For example:

```cpp
{1, 2}   // neck to right shoulder
{2, 3}   // right shoulder to right elbow
{3, 4}   // right elbow to right wrist
{8, 9}   // right hip to right knee
{9, 10}  // right knee to right ankle
```

## 10. Pose Model Notes

The current pose estimator is designed for an OpenPose COCO-style model.

That matters because the code assumes:

```text
18 keypoint heatmap channels
OpenPose COCO keypoint order
One best point per keypoint channel
```

If you replace the model with another architecture, you may need to change:

```text
include/processors/PoseEstimator.hpp
include/processors/SkeletonRenderer.hpp
```

For example:

### If you switch to MoveNet

MoveNet does not output OpenPose heatmaps in the same format.

You would probably replace:

```cpp
PoseEstimator
```

with a new processor such as:

```cpp
MoveNetPoseEstimator
```

That new processor would parse MoveNet output and fill:

```cpp
ctx.poses
```

Then `SkeletonRenderer` may still be reusable if the keypoint order is adapted.

### If you switch to YOLO pose

YOLO pose usually outputs:

```text
bounding boxes + keypoints + confidence values
```

That output format is very different from OpenPose heatmaps.

You would need a YOLO-specific estimator that:

```text
1. Parses detections.
2. Runs non-maximum suppression.
3. Extracts keypoints for each person.
4. Fills ctx.poses.
```

### If you switch to TensorRT

On Jetson Orin Nano, the recommended high-performance path is:

```text
ONNX model -> TensorRT engine -> custom TensorRT inference processor
```

In that case, create a new processor:

```text
include/processors/TensorRTPoseEstimator.hpp
```

Then change `buildPipeline()` in `src/main.cpp`:

```cpp
pipeline->addProcessor(std::make_unique<video_engine::TensorRTPoseEstimator>(...));
pipeline->addProcessor(std::make_unique<video_engine::SkeletonRenderer>());
```

OpenCV can still handle:

- Camera input
- Image resizing
- Drawing
- Display
- Video output

TensorRT would replace the expensive neural network inference part and can use Orin
Nano's Ampere Tensor Cores for supported FP16/INT8 operations.

## 11. Inference Backend and Target

OpenCV DNN lets you choose a backend and target.

In `configs/pose.yaml`:

```yaml
pose_backend: opencv
pose_target: cpu
```

This means:

```text
Use OpenCV's default DNN backend on CPU.
```

Other possible values supported by the code include:

```yaml
pose_backend: openvino
pose_target: cpu
```

```yaml
pose_backend: opencv
pose_target: opencl
```

```yaml
pose_backend: cuda
pose_target: cuda_fp16
```

The Orin profile provides the same selection:

```yaml
inference_platform: jetson
```

See [Platform Run Guide](PLATFORM_RUN_GUIDE.md) for macOS and Orin Nano commands.

Important: the code accepting a backend name does not mean your installed OpenCV build supports that backend.

For example:

- CUDA requires OpenCV built with CUDA DNN support.
- OpenVINO requires OpenCV built with OpenVINO support.
- NPU targets require platform support and compatible OpenCV/OpenVINO builds.

If a backend is unavailable, OpenCV may fail at runtime or fall back depending on the backend.

## 12. Build Instructions

From the project directory:

```bash
cd /Users/yifei_li/Desktop/openCV/openCV
```

Configure in Release mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Build:

```bash
cmake --build build -j8
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Why Release mode matters:

```text
Debug builds can be much slower.
Pose inference is already expensive.
Always benchmark FPS and latency with Release builds.
```

## 13. How To Run Each Mode

### Motion mode

```bash
./build/video_engine --pipeline motion --config configs/motion.yaml
```

Expected result:

```text
Moving areas are drawn as boxes.
```

### Tracking mode

```bash
./build/video_engine --pipeline tracking --config configs/tracking.yaml
```

Expected result:

```text
Moving areas are drawn as boxes with IDs and trails.
```

### Pose mode

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml
```

Expected result:

```text
Human body keypoints and skeleton lines are drawn.
```

### Run on a video file

Edit the config:

```yaml
source: videos/demo.mp4
```

Or pass it from CLI:

```bash
./build/video_engine --source videos/demo.mp4 --pipeline pose --config configs/pose.yaml
```

Note: the program loads the config file first, then applies command-line arguments. This means command-line arguments override config-file values.

## 14. Saving Output Video

Processed output is saved by default because the config files use:

```yaml
save_output: true
save_path: output/output
```

`save_path` is treated as a filename prefix. The program chooses the first available numbered MP4 path with a `while` loop:

```text
output/output0.mp4
output/output1.mp4
output/output2.mp4
...
```

For example, if `output/output0.mp4` already exists, the next run saves to:

```text
output/output1.mp4
```

You can override the prefix from the command line:

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml --save output/pose_debug
```

That will create:

```text
output/pose_debug0.mp4
```

The output video dimensions are:

```yaml
width: 640
height: 480
```

## 15. Reading Runtime Logs

The program logs lines like:

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

Meaning:

- `Frame 3`: the processed frame index.
- `FPS: 3.4`: average frames per second since startup.
- `pose_inference`: wall-clock time around `net_.forward()`.
- `pose_estimator`: preprocessing, inference, and postprocessing combined.
- `pipeline`: all processors combined; capture, display, and video writing are excluded.

For pose mode, most of the latency usually comes from:

```cpp
net_.forward()
```

inside:

```text
PoseEstimator.hpp
```

The sample above was measured on the current macOS CPU setup and is not an Orin Nano
performance claim. See [Pose Inference Guide](POSE_INFERENCE_GUIDE.md) for benchmarking
instructions.

## 16. Debugging Strategy

### 16.1 First check that the program builds

```bash
cmake --build build -j8
```

If this fails, the problem is compile-time.

Common causes:

- Missing OpenCV headers
- Wrong include path
- Missing OpenCV library component
- C++ syntax error

### 16.2 Then run unit tests

```bash
ctest --test-dir build --output-on-failure
```

Current tests cover:

- Motion detection
- Object tracking ID assignment
- Skeleton rendering

### 16.3 Then run a simple pipeline

Start with motion mode:

```bash
./build/video_engine --pipeline motion --config configs/motion.yaml
```

If motion mode works, your camera/display path is probably fine.

Then try pose mode:

```bash
./build/video_engine --pipeline pose --config configs/pose.yaml
```

If motion works but pose fails, the issue is likely:

- Missing model file
- Bad model path
- Unsupported DNN backend
- Pose model output format mismatch

### 16.4 Check model paths

Pose mode requires:

```text
models/pose_deploy_linevec.prototxt
models/pose_iter_440000.caffemodel
```

Check them:

```bash
ls -lh models/pose_deploy_linevec.prototxt models/pose_iter_440000.caffemodel
```

Expected:

```text
pose_deploy_linevec.prototxt   around 45 KB
pose_iter_440000.caffemodel    around 200 MB
```

### 16.5 Check OpenCV can load the model

You can test with Python:

```bash
python3 -c "import cv2; net=cv2.dnn.readNet('models/pose_iter_440000.caffemodel','models/pose_deploy_linevec.prototxt'); print('loaded', not net.empty())"
```

Expected:

```text
loaded True
```

### 16.6 Debug include errors in VSCode

If the program builds but VSCode shows red include errors, the editor does not know the same include paths as CMake.

This project uses:

```text
build/compile_commands.json
```

VSCode should point to it:

```json
"C_Cpp.default.compileCommands": "${workspaceFolder}/openCV/build/compile_commands.json"
```

If the red squiggles remain:

```text
Cmd + Shift + P
-> C/C++: Reset IntelliSense Database
-> Developer: Reload Window
```

## 17. Tuning Motion Mode

Use:

```text
configs/motion.yaml
```

Important parameters:

```yaml
threshold: 25
min_area: 500
merge_padding: 24
max_detections: 0
```

If there are too many small noisy boxes:

```yaml
min_area: 1000
```

If the same person is split into many boxes:

```yaml
merge_padding: 40
```

If you only want the main moving target:

```yaml
max_detections: 1
```

If you want all moving targets:

```yaml
max_detections: 0
```

## 18. Tuning Tracking Mode

Use:

```text
configs/tracking.yaml
```

Important parameters:

```yaml
max_distance: 80
max_age: 8
trail_length: 25
```

If IDs change too easily:

```yaml
max_distance: 120
max_age: 12
```

If different objects get confused:

```yaml
max_distance: 50
```

If trails are too long:

```yaml
trail_length: 10
```

## 19. Tuning Pose Mode

Use:

```text
configs/pose.yaml
```

For the full setting reference, platform comparison, and benchmark method, see
[Pose Inference Guide](POSE_INFERENCE_GUIDE.md).

Important parameters:

```yaml
pose_input_width: 256
pose_input_height: 256
pose_confidence: 0.12
pose_backend: opencv
pose_target: cpu
```

### Higher accuracy

Try:

```yaml
pose_input_width: 368
pose_input_height: 368
```

This is slower but can produce better keypoint placement.

### Higher FPS

Try:

```yaml
pose_input_width: 192
pose_input_height: 192
```

This is faster but less accurate.

The project always performs fresh inference for each successfully decoded frame. It does
not reuse old poses to increase display FPS because stale measurements corrupt downstream
velocity and repetition analysis.

## 20. How To Add a New Processor

Create a new header:

```text
include/processors/MyProcessor.hpp
```

Implement:

```cpp
#pragma once

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class MyProcessor : public IFrameProcessor {
public:
    void process(FrameContext& ctx) override {
        // Read or write ctx here.
    }

    std::string name() const override {
        return "my_processor";
    }
};

}  // namespace video_engine
```

Then add it in `buildPipeline()`:

```cpp
pipeline->addProcessor(std::make_unique<video_engine::MyProcessor>());
```

Where you add it matters.

For example:

- Add it before `PoseEstimator` if it prepares image input.
- Add it after `PoseEstimator` if it uses pose keypoints.
- Add it after `SkeletonRenderer` if it modifies the final visualization.

## 21. How To Replace the Pose Model

There are two levels of replacement.

### 21.1 Replace with another OpenPose COCO Caffe model

If the new model has the same output format, only update:

```yaml
pose_model: path/to/new.caffemodel
pose_config: path/to/new.prototxt
```

No C++ changes should be required.

### 21.2 Replace with a different pose architecture

If the output format changes, create a new estimator.

Examples:

```text
MoveNetPoseEstimator
YoloPoseEstimator
MediaPipePoseEstimator
TensorRTPoseEstimator
```

The new estimator should still write results into:

```cpp
ctx.poses
```

That lets `SkeletonRenderer` keep working, as long as the keypoint order matches or is converted.

## 22. Common Problems

### Problem: OpenCV window does not open

Possible causes:

- Running in a terminal without GUI support
- Camera permission denied
- `display: false` in config

Try:

```yaml
display: true
```

### Problem: Camera cannot open

Possible causes:

- macOS camera permission not granted
- Another app is using the camera
- Camera index is not 0

The current `WebcamSource` always uses camera index `0`.

To support another index, extend the config and pass it into:

```cpp
WebcamSource(device_index)
```

### Problem: Pose mode is very slow

Likely cause:

```text
OpenPose COCO is a heavy model, and CPU inference is expensive.
```

Try:

```yaml
pose_input_width: 192
pose_input_height: 192
```

For real edge AI performance, replace the model and inference backend:

```text
Jetson Orin Nano: TensorRT + lightweight ONNX pose model
Intel edge: OpenVINO model
Apple Silicon: CoreML model
```

### Problem: Skeleton appears but keypoints are wrong

Possible causes:

- Person is too small in the frame
- Lighting is poor
- Model input resolution is too low
- OpenPose single-person heatmap parsing is mixing multiple people

Try:

```yaml
pose_input_width: 368
pose_input_height: 368
pose_confidence: 0.2
```

### Problem: Multiple people create weird skeletons

Current limitation:

```text
PoseEstimator takes the maximum response for each keypoint channel globally.
```

This is simple and works best for one person.

For true multi-person pose, you need:

- Full OpenPose PAF parsing
- Or a model that directly outputs per-person keypoints, such as YOLO pose

## 23. Recommended Learning Path

If you want to understand the project deeply, read in this order:

```text
1. src/main.cpp
2. include/core/FrameContext.hpp
3. include/core/Pipeline.hpp
4. include/core/IFrameProcessor.hpp
5. include/processors/ResizeProcessor.hpp
6. include/processors/MotionDetector.hpp
7. include/processors/ObjectTracker.hpp
8. include/processors/PoseEstimator.hpp
9. include/processors/SkeletonRenderer.hpp
10. configs/*.yaml
```

The most important concept is:

```text
Every processor reads and writes FrameContext.
```

Once that is clear, the rest of the project becomes much easier to modify.
