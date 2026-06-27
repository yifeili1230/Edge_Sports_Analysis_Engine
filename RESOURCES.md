# Jetson Orin Nano Pose Analytics Resources

## Knowledge

- [NVIDIA: Jetson Orin technical specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
  CUDA cores, Tensor Cores, AI performance, memory, and power specifications for Orin
  Nano variants. Use for hardware capability boundaries.
- [NVIDIA: Jetson Orin Nano Developer Kit guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/)
  Official entry point for the developer kit, JetPack, CUDA, and board setup.
- [NVIDIA: JetPack SDK setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_jetpack.html)
  Installation and version-check workflow for the JetPack software stack.
- [OpenCV: DNN module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
  Backend, target, network-loading, and inference APIs used by `PoseEstimator`.
- [OpenCV: Configuration options](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)
  CUDA, cuDNN, and DNN build settings needed for GPU-enabled OpenCV.
- [NVIDIA-AI-IOT: trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
  Jetson-oriented TensorRT pose models. Use when evaluating a replacement for OpenPose.
- [jetson-inference: poseNet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/posenet.md)
  C++ TensorRT pose-estimation reference for Jetson deployments.
- [OpenCV: KalmanFilter](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html)
  Prediction and measurement-correction API for future keypoint trajectory smoothing.
- [OpenCV: Camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
  Camera-model requirements for converting image measurements toward real-world geometry.

## Wisdom (Communities)

- [NVIDIA Jetson Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)
  Practical help for JetPack, CUDA, TensorRT, GStreamer, cameras, and version compatibility.
- [OpenCV Forum](https://forum.opencv.org/)
  Useful for version-specific DNN CUDA behavior and OpenCV build problems.
