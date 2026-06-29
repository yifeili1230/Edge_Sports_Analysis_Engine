# Squat Analytics Guide

This guide describes the deterministic side-view squat analyzer, its real-time display,
configuration, data flow, and JSON session output.

## 1. Run Squat Analysis

From the project root:

```bash
./build/video_engine \
  --source video_source/IMG_1389.mov \
  --config configs/squat.yaml \
  --exercise squat
```

`configs/squat.yaml` already selects the pose pipeline and squat analyzer, so the shorter
equivalent is:

```bash
./build/video_engine \
  --source video_source/IMG_1389.mov \
  --config configs/squat.yaml
```

Press `ESC` in the video window to stop normally. `Ctrl+C` is also handled as a graceful
stop: the current inference finishes, the video writer is finalized, and the JSON summary
is written.

## 2. Runtime Data Flow

```text
VideoSource
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
Canonical Pose
    |
    v
PoseAnalyticsProcessor
    |
    +--> SquatAnalyzer
    |      confidence gating
    |      side selection
    |      2D angles
    |      smoothing
    |      normalized speed
    |      phase state machine
    |      completed-rep event
    |
    v
SkeletonRenderer
    |
    v
PoseAnalyticsRenderer
    |
    +--> video on the left
    +--> live metrics panel on the right
```

`SquatAnalyzer` implements `IPoseAnalyzer` and has no OpenCV dependency. The processor
adapts `FrameContext` to that pure interface. A future learned squat analyzer can
implement the same interface without changing video capture or pose inference.

## 3. Live Side Panel

The analyzer preserves the complete processed video and adds a 300-pixel panel to its
right. For a 640×480 processing frame, the displayed and saved result is 940×480.

The panel updates every processed frame:

- Current phase
- Completed repetition count
- Smoothed knee angle
- Hip angle, when the shoulder is valid
- Body-scale-normalized vertical hip speed
- Left or right observed body side
- Metrics for a newly completed repetition

When required joints have low confidence, the panel reports that the analysis state is
paused. A missing observation does not advance the phase or count a repetition.

## 4. Joint Selection and Measurements

The analyzer evaluates the left and right chains:

```text
shoulder -> hip -> knee -> ankle
```

It selects the side whose hip, knee, and ankle have the stronger minimum confidence.

The knee angle is:

```text
angle(hip, knee, ankle)
```

The hip angle is:

```text
angle(shoulder, hip, knee)
```

Angles are 2D image-plane projections. They are useful for a fixed side or approximately
45-degree view, but they are not anatomical 3D angles. The canonical pose schema already
reserves optional depth and 3D coordinates for a future calibrated implementation.

Vertical hip speed is calculated from source timestamps, not processing FPS:

```text
(current smoothed hip Y - previous smoothed hip Y)
--------------------------------------------------
     hip-to-knee body scale × source time delta
```

Positive values indicate downward image motion; negative values indicate upward motion.
The body-scale normalization makes the value less sensitive to image resolution and
distance from the camera. It is not meters per second.

## 5. Repetition State Machine

One repetition must pass through the complete ordered sequence:

```text
Standing -> Descending -> Bottom -> Ascending -> Standing
```

Default thresholds:

| Transition | Condition |
| --- | --- |
| Establish standing | Knee angle ≥ 160° |
| Start descending | Knee angle < 155° with downward motion or decreasing angle |
| Reach bottom | Knee angle ≤ 100° |
| Start ascending | Knee angle ≥ 105° with upward motion or increasing angle |
| Complete repetition | Knee angle ≥ 160° |

The separate entry and exit thresholds provide hysteresis so small fluctuations around a
single boundary do not repeatedly change the phase. A movement that never reaches the
bottom threshold is not counted.

## 6. Configuration

The settings live in `configs/squat.yaml`:

```yaml
exercise: squat
analysis_output_dir: output
film_side: right_side_view
squat:
  squat_standing_angle: 160
  squat_descent_start_angle: 155
  squat_bottom_angle: 100
  squat_bottom_exit_angle: 105
  squat_minimum_speed: 0.02
  squat_smoothing_alpha: 0.35
```

The pose confidence setting is also the minimum confidence used by the analyzer:

```yaml
pose_confidence: 0.12
```

The analyzer also supports stability controls:

```yaml
squat_transition_confirmation_frames: 2
squat_bottom_confirmation_frames: 1
squat_descent_commit_angle: 145
squat_minimum_phase_duration: 0.10
squat_minimum_rep_duration: 0.50
squat_maximum_angular_velocity: 720
squat_maximum_normalized_speed: 4.0
squat_direction_confirmation_speed: 0.50
```

Phase changes must persist across multiple frames, satisfy minimum timing, and agree
with knee direction. The configured film side is locked for a repetition. Impossible
one-frame angle jumps are rejected and extreme hip-speed samples are ignored, preventing
small pose jitter from beginning or ending a phase early.

Supported film sides are `front_view`, `front_left_view`, `front_right_view`,
`left_side_view`, `right_side_view`, `rear_left_view`, `rear_view`, and
`rear_right_view`. Diagonal and side views select their matching body chain. Straight
front and rear views choose the valid chain with the higher joint confidence.

Use `resize_mode: fit` for analytics. Non-uniform resizing changes the angles between
image-space limb vectors even when the source pose did not change.

The ready-to-run Jetson profile combines these settings with YOLO11 TensorRT:

```bash
./build/video_engine --config configs/yolo11_squat_jetson.yaml
```

That profile uses a 110° bottom and 115° bottom-exit threshold. On the supplied portrait
video, two valid repetitions bottom near 105° in YOLO coordinates and would be rejected
by the legacy Caffe profile's stricter 100°/105° thresholds. Thresholds are deliberately
model/profile-specific rather than silently changing the Caffe baseline.

Phase changes are logged as compact `[Squat]` lines. For frame-by-frame diagnosis:

```bash
VIDEO_ENGINE_TRACE_ANALYTICS=1 ./build/video_engine \
  --config configs/yolo11_squat_jetson.yaml
```

If a descent begins late because hip/knee/ankle observations are invalid, the analyzer
pauses rather than fabricating a start time. Improving that case requires better pose
visibility, camera placement, or temporal keypoint tracking.

Tune thresholds against representative videos. Camera placement, clothing, body
proportions, and model accuracy can all change the observed 2D angles.

## 7. Session Summary

The summary is written under `analysis_output_dir` using the input video's stem:

```text
video_source/IMG_1389.mov -> output/IMG_1389.json
video_source/squat.mov    -> output/squat.json
webcam                    -> output/webcam.json
```

The same path is replaced on the next run for that input. Example:

```json
{
  "source": "video_source/IMG_1389.mov",
  "exercise": "squat",
  "processed_frames": 875,
  "valid_analysis_frames": 851,
  "invalid_analysis_frames": 24,
  "total_reps": 1,
  "reps": [
    {
      "rep": 1,
      "start_time_seconds": 5.200,
      "end_time_seconds": 8.700,
      "descent_time_seconds": 1.700,
      "ascent_time_seconds": 1.800,
      "minimum_knee_angle_degrees": 91.400,
      "average_normalized_speed_per_second": 0.430,
      "peak_normalized_speed_per_second": 0.810
    }
  ]
}
```

The numbers above illustrate the schema and are not benchmark results.

## 8. Output Video

Video saving is independent from JSON summary saving:

```bash
./build/video_engine \
  --source video_source/IMG_1389.mov \
  --config configs/squat.yaml \
  --save output/IMG_1389_annotated
```

This produces a numbered MP4 such as:

```text
output/IMG_1389_annotated0.mp4
```

Use `--no-save` to skip annotated video encoding. The JSON summary is still written when
squat analysis is enabled.

## 9. Current Limitations

- The current OpenPose Caffe decoder emits one simple skeleton and is not a true
  multi-person decoder.
- Persistent primary-person tracking is not implemented yet.
- The analyzer can therefore follow incorrect joints if another person enters.
- Measurements are 2D projections and normalized image motion, not metric 3D motion.
- Thresholds have synthetic-sequence coverage but still require validation and tuning
  against a labeled real-video set.
- Current macOS CPU inference is much slower than the source video frame rate, although
  calculations still use source timestamps.

## 10. Tests

`tests/test_squat_analyzer.cpp` verifies:

- 2D angle calculation
- One complete phase cycle produces exactly one repetition
- Descent, ascent, minimum-angle, and speed metrics are populated
- Low-confidence observations do not corrupt state
- A shallow movement that never reaches the bottom is not counted
- Input video names map to the expected JSON summary path

Run:

```bash
ctest --test-dir build --output-on-failure
```
