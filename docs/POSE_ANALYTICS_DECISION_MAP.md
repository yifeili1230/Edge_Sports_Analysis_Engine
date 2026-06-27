# Pose Analytics and Orin Optimization Decision Map

Goal: replace cached pose reuse with precise per-frame measurements, adopt a faster
cross-platform multi-person pose model, lock onto one primary subject, and provide an
extensible mathematical exercise-analysis layer beginning with side-view squats.

## #1: Input Geometry

Blocked by: none
Type: Grilling

### Question

What camera geometry must the first analytics release support?

### Answer

Resolved. Support ordinary monocular cameras and recorded video without changing the
current source layer. Optimize squat analysis for a fixed side or approximately 45-degree
side view. Use 2D measurements now, but keep optional depth/3D fields in the canonical pose
interface so stereo or depth cameras can be added later.

## #2: Primary Subject Policy

Blocked by: #1
Type: Grilling

### Question

How is the primary person selected and preserved when other people enter the frame?

### Answer

Resolved. Initially select the largest person near the frame center. Lock that identity
across frames and do not switch when another person enters. Reacquire only after the target
has been missing for a configured number of frames, using predicted position and pose/bbox
similarity.

## #3: Performance Contract

Blocked by: #1
Type: Grilling

### Question

What performance must Jetson Orin Nano achieve?

### Answer

Resolved. At 640×480 input, run pose inference on every frame with no cached-result reuse.
Target at least 20 FPS and less than 100 ms end-to-end latency. Benchmark with display and
video saving disabled, then measure their costs separately.

## #4: Squat Rep Semantics

Blocked by: #1
Type: Grilling

### Question

What constitutes one squat repetition in the first mathematical analyzer?

### Answer

Resolved. Use a hysteretic state machine: standing above 160° knee angle, descending with
decreasing angle and downward hip motion, bottom below 100°, ascending with increasing
angle and upward hip motion, then standing above 160°. Count only the complete ordered
cycle. Report descent time, ascent time, minimum knee angle, and average/peak normalized
speed. Thresholds must be configurable.

## #5: Analytics Interface Boundary

Blocked by: #1, #4
Type: Grilling

### Question

How can mathematical analyzers later be replaced or complemented by trained models?

### Answer

Resolved. Keep a pure, OpenCV-independent `IPoseAnalyzer` interface. Implement
`SquatAnalyzer` as pure mathematics over canonical keypoints, confidence, identity, and
time. Add a `PoseAnalyticsProcessor` adapter for `FrameContext`. Expose both per-frame
analysis results and completed-rep events. Future trained analyzers implement the same
interface.

## #6: Cross-Platform Model Contract

Blocked by: #3, #5
Type: Grilling

### Question

May macOS and Orin use different models?

### Answer

Resolved. Prefer one ONNX model and one keypoint/output contract on both platforms:
OpenCV DNN CPU on macOS and TensorRT FP16 on Orin Nano. Platform-specific inference
adapters are allowed; tracking and analytics must receive the same canonical pose schema.

## #7: Reliable Per-Frame Measurement Timeline

Blocked by: #1, #3
Type: Prototype

### Question

What exact frame/time contract prevents cached poses and timing artifacts from contaminating
velocity and rep analysis?

### Answer

Resolved. `PoseEstimator` now runs on every frame; cached poses, inference intervals, and
`pose_cache` were removed. `FrameTimeline` assigns a strictly increasing frame ID and
source-relative timestamp, then clears prior pose state. Live sources use a steady clock;
recorded video uses media time with frame/FPS fallback. Each frame exposes
`pose_inference_ran` and `pose_measurement_valid`. The contract is covered by
`tests/test_frame_timeline.cpp`; all tests pass, and a recorded-video smoke test showed
`pose_inference` on every frame with no stale-pose path.

## #8: Canonical Pose and Optional Depth Schema

Blocked by: #7
Type: Prototype

### Question

What model-independent data types allow 2D math now, multi-person tracking next, and depth
later?

### Answer

Resolved. `include/pose/PoseTypes.hpp` defines a fixed named `JointId` schema with float 2D
pixels, confidence/validity, optional depth and 3D coordinates, person ID, bounding box,
frame dimensions, frame ID, and source time. It has no OpenCV dependency.
`OpenPoseCocoAdapter.hpp` explicitly maps all 18 model channels into that schema.
`PoseEstimator` publishes canonical poses and `SkeletonRenderer` consumes named joints.
`tests/test_pose_schema.cpp` proves complete unique mapping, float precision, and optional
depth behavior; all tests and a real-model video smoke test pass.

## #9: Pure Mathematical Squat Analyzer

Blocked by: #4, #5, #8
Type: Prototype

### Question

Can deterministic, unit-tested math robustly segment reps and report useful side-view
metrics before any learned exercise model is introduced?

### Answer

Open. Implement confidence gating, 2D joint angles, body-scale-normalized position and
velocity, smoothing, hysteretic phase transitions, rep events, and invalid-data handling.
Test with synthetic keypoint sequences before connecting video inference.

## #10: Faster Accurate Multi-Person Pose Model

Blocked by: #3, #6, #8
Type: Research

### Question

Which licensable ONNX pose model best satisfies shared macOS/Orin output semantics,
multi-person detection, subject tracking, side-view squat accuracy, and the 20 FPS target?

### Answer

Open. Compare small YOLO-pose, RTMPose-based, MoveNet MultiPose, and other current candidates
using authoritative model documentation, licenses, export compatibility, keypoint schemas,
and Orin TensorRT evidence. Do not choose from headline FPS alone.

## #11: Cross-Platform Inference Adapters and Benchmark

Blocked by: #10
Type: Prototype

### Question

Does the selected model produce equivalent canonical poses through OpenCV DNN on macOS and
TensorRT FP16 on Orin Nano?

### Answer

Open. Implement model preprocessing/decoding once where possible, platform inference
adapters where necessary, and golden-output tests. Benchmark warm inference, preprocessing,
postprocessing, memory transfers, and total latency with identical video and settings.

## #12: Persistent Primary-Subject Tracking

Blocked by: #2, #8, #11
Type: Prototype

### Question

Which association and reacquisition strategy preserves the original subject through
occlusion and distractors without identity switching?

### Answer

Open. Start with bbox IoU, predicted centroid, keypoint similarity, size, and center prior;
use a persistent ID and configurable lost-frame timeout. Test crossings, temporary
occlusion, a larger distractor entering, and target exit/re-entry before considering a
learned re-identification model.

## #13: Tracked Squat Pipeline Integration

Blocked by: #9, #12
Type: Prototype

### Question

Does the full pipeline analyze only the locked subject and preserve correct rep state
through low-confidence frames and brief occlusion?

### Answer

Open. Integrate pose inference → subject tracking → canonical pose → squat analysis →
overlay/event sinks. Validate that distractors cannot increment reps or replace the target,
and that invalid observations pause rather than corrupt state.

## #14: Orin Nano Dedicated Optimization

Blocked by: #11, #13
Type: Research

### Question

Which Orin-specific optimizations are required after model and analytics correctness are
established?

### Answer

Open. Measure TensorRT FP16 and eligible INT8 paths, Tensor Core use, power mode, thermals,
capture/decode, CPU↔GPU transfer, and bounded asynchronous stages. Add GStreamer/NVMM,
GPU preprocessing, or zero-copy only when profiling shows a material bottleneck.

## #15: Optional Depth/3D Adapter

Blocked by: #8, #13
Type: Prototype

### Question

Can a future stereo/depth source enrich the same analytics contract without changing the
2D squat analyzer or inference pipeline?

### Answer

Deferred. Define the optional schema in #8, but do not implement a depth source in the
first release. Later add calibrated depth/3D joint adapters and separate 3D metric
implementations behind the same analyzer interfaces.
