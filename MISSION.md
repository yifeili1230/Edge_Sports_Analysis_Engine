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



