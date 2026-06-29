# Mission: Jetson Orin Nano Pose and Exercise Analytics

## Why
Deploy the current C++/OpenCV pose project on Jetson Orin Nano, understand the cost of
CPU, GPU, Tensor Core, and memory-transfer stages, and build a reliable exercise-analysis
pipeline rather than a display-only demo.

## Success looks like
- Explain the CPU/GPU path of every frame and measure each important stage.
- Run a fresh pose inference on every frame with stable frame IDs and source timestamps.
- Use one canonical pose schema across macOS, OpenCV, ONNX, and TensorRT backends.
- Lock onto a primary person and keep that identity when distractors enter.
- Count side-view squat repetitions and report joint angles, phase timing, and normalized
  motion speed through a model-independent analysis interface.
- Meet the Orin Nano target of at least 20 FPS and less than 100 ms end-to-end latency.

## Constraints
- Target hardware is Jetson Orin Nano with an Ampere GPU and Tensor Cores.
- CUDA compute capability is 8.7; JetPack 6 is the preferred software stack.
- Current implementation uses C++17, OpenCV DNN, and an OpenPose COCO Caffe model.
- The first analytics release uses a fixed monocular side or 45-degree side view.
- Preserve the existing `Pipeline` and `FrameContext` architecture where practical.

## Out of scope
- Training a custom exercise model in the first release.
- Implementing a complete DeepStream production system.
- Implementing stereo/depth capture now; the data contract only reserves optional depth
  and 3D fields.



The project is running a YOLO11n pose + TensorRT pipeline on a Jetson Orin Nano Developer Kit “Super.” Neural inference runs on the NVIDIA GPU/Tensor Cores; nearly everything else runs on the ARM CPU.
Hardware and software
Detected locally:
NVIDIA Jetson Orin Nano Developer Kit Super
8 GB unified memory
6× ARM Cortex‑A78AE CPU cores, up to 1.728 GHz
Ampere GPU, CUDA compute capability 8.7
15 W power mode
Jetson Linux R36.4.7
TensorRT 10.3, CUDA 12 runtime
OpenCV 4.8.0
Release build with native TensorRT enabled
YOLO11n-pose 640×640 TensorRT engine, internally FP16
The installed OpenCV has no CUDA support. That is fine for these YOLO benchmarks because native TensorRT is used directly. The legacy Caffe Jetson profiles cannot use GPU acceleration with this OpenCV build.
Frame lifecycle
Both input videos are portrait 1080×1920, 30 FPS:
MOV decode
  → 360×640 processing frame
  → 640×640 YOLO letterbox
  → TensorRT inference
  → YOLO keypoint/NMS decoding
  → exercise analysis
  → skeleton drawing
  → 300-pixel analytics panel
  → 660×640 display/MP4 frame
The processor order is assembled in [main.cpp (line 324)](/home/yifei/openCV/src/main.cpp:324).
Reported stage	What it does	Hardware
resize	CPU-decodes frame is resized from 1080×1920 to 360×640, preserving aspect ratio	CPU
pose_preprocess	Letterboxes to 640×640 with gray padding, converts BGR→RGB, scales pixels to 0–1, creates FP32 NCHW tensor	CPU
pose_inference	Uploads FP32 input, executes FP16 TensorRT engine, downloads FP32 [1,56,8400] result, synchronizes and clones output	CPU + GPU
pose_postprocess	Filters detections, converts boxes/keypoints out of letterbox coordinates, runs NMS, maps COCO-17 joints, synthesizes neck	CPU
pose_estimator	Aggregate of the preceding three stages plus minimal bookkeeping	Mixed
pose_analytics	Calculates angles, smoothed normalized motion speed, exercise phase and repetitions	CPU
skeleton_renderer	Clones the frame and draws valid joints and 14 skeletal connections	CPU
pose_analytics_renderer	Creates the 300-pixel panel and draws phase, count and metrics	CPU
pipeline	Complete processor chain above	Mixed
FPS	Whole frame loop, including work not counted in pipeline	Whole system

The TensorRT upload, execution, download, and synchronization are visible in [TensorRtRunner.hpp (line 107)](/home/yifei/openCV/include/inference/TensorRtRunner.hpp:107). Thus, pose_inference is host-observed inference latency, not pure GPU kernel time.
Benchmark interpretation
Measurement	OHP	Squat
Frames	589	875
Throughput	16.8 FPS	19.2 FPS
Wall time/frame from FPS	59.5 ms	52.1 ms
Pipeline	40.12 ms	42.59 ms
Pipeline-only capacity	24.9 FPS	23.5 FPS
Work outside pipeline	19.4 ms	9.5 ms
TensorRT inference	29.97 ms	30.95 ms
Inference share of pipeline	74.7%	72.7%
Complete pose estimator	35.47 ms	37.21 ms
Pose-estimator share	88.4%	87.4%

The two runs are quite consistent: inference differs by only 0.98 ms, while the complete pipeline differs by 2.47 ms. The network is unequivocally the main bottleneck.
Preprocessing is now significant too: 5.4–6.2 ms, about 14% of the pipeline. Analytics itself is effectively free at 0.01 ms. Rendering, including the side panel, remains under 2.7 ms total.
The large difference between pipeline and reciprocal FPS comes from operations outside its timer in [main.cpp (line 557)](/home/yifei/openCV/src/main.cpp:557):
source video decoding
window display and event processing
MP4 encoding and writing
per-frame console logging
general loop overhead
Both supplied TensorRT configs enable display and MP4 saving ([OHP config (line 6)](/home/yifei/openCV/configs/yolo11_ohp_jetson.yaml:6), [squat config (line 6)](/home/yifei/openCV/configs/yolo11_squat_jetson.yaml:6)). Therefore, these are application demonstrations, not clean inference benchmarks.
Exercise behavior
For OHP, the rear-view analyzer combines both arms when available, measures shoulder–elbow–wrist angle and torso-normalized wrist motion, then follows:
Rack → Pressing → Lockout → Lowering → Rack
It found 6 repetitions, with all 589 frames valid.
For squat, the configured right-side chain uses shoulder, hip, knee and ankle. It calculates 2D knee/hip/back/ankle angles and normalized vertical speed, following:
Standing → Descending → Bottom → Ascending → Standing
It found 5 repetitions, with all 875 frames valid. Results are in [ohp.json](/home/yifei/openCV/output/ohp.json) and [squat.json](/home/yifei/openCV/output/squat.json).
These measurements are 2D image-space estimates—not anatomical 3D angles or metric velocities.
Assessment
The project’s stated latency goal is met comfortably: pipeline latency is about 40–43 ms, and total observed frame time is 52–60 ms, both below 100 ms.
The 20 FPS target is not demonstrated by these runs:
Squat is 4% short.
OHP is 16% short.
However, because these runs encode MP4 and likely display every frame, they should not be used to declare the compute pipeline below target. The pipeline itself already supports roughly 23–25 FPS. A proper target measurement should run:
./build/video_engine --config configs/yolo11_squat_jetson.yaml --no-display --no-save
Remaining project limitations:
Analytics always consumes poses.front(); persistent subject tracking is not implemented.
Capture/decode, preprocessing and rendering are CPU-based—no NVMM, hardware decode, GPU preprocessing, or zero-copy path.
Benchmark averages include the cold first frame and provide no median, p95 or variance.
GPU utilization, clocks, temperature and power were not recorded alongside these runs.
The board was in 15 W mode and clocks were not confirmed locked.
All eight current unit tests pass. The implementation is functionally healthy; the next useful performance work is a clean headless benchmark, followed by profiling TensorRT kernels and the 5–6 ms CPU preprocessing path.

