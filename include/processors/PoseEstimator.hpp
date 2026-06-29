#pragma once

#include <algorithm>  // Normalizes backend/target strings and replaces separators.
#include <cctype>     // Converts backend/target names to lowercase safely.
#include <limits>     // Initializes pose bounding-box extrema.
#include <memory>     // Owns the optional native TensorRT runner.
#include <stdexcept>  // Reports model loading and backend configuration errors.
#include <string>     // Stores model paths and backend/target names.

#include <opencv2/core.hpp>    // Provides cv::Mat, cv::Point, and scalar types.
#include <opencv2/dnn.hpp>     // Provides OpenCV DNN model loading and inference.
#include <opencv2/imgproc.hpp> // Provides blob/image preprocessing helpers.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.
#include "inference/TensorRtRunner.hpp" // Runs a fixed-shape TensorRT engine on Jetson.
#include "pose/OpenPoseCocoAdapter.hpp" // Maps OpenPose output channels to canonical joints.
#include "pose/YoloPoseDecoder.hpp"   // Decodes YOLO11 COCO-17 pose predictions.
#include "utils/Profiler.hpp"        // Measures preprocessing, inference, and postprocessing latency.

namespace video_engine {

class PoseEstimator : public IFrameProcessor {
public:
    explicit PoseEstimator(std::string model_weights, std::string model_config = "",
                           int input_width = 368, int input_height = 368,
                           float confidence_threshold = 0.12F,
                           std::string backend = "opencv", std::string target = "cpu",
                           std::string model_format = "openpose_coco",
                           float detection_confidence = 0.25F,
                           float nms_threshold = 0.45F, int max_poses = 10)
        : input_width_(input_width),
          input_height_(input_height),
          confidence_threshold_(confidence_threshold),
          detection_confidence_(detection_confidence),
          nms_threshold_(nms_threshold),
          max_poses_(max_poses) {
        if (model_weights.empty()) {
            throw std::runtime_error("Pose model path is empty. Set pose_model in configs/pose.yaml.");
        }

        normalize(model_format);
        if (model_format == "openpose_coco" || model_format == "openpose") {
            model_format_ = ModelFormat::OpenPoseCoco;
        } else if (model_format == "yolo11_pose" || model_format == "yolo_coco17") {
            model_format_ = ModelFormat::YoloCoco17;
        } else {
            throw std::runtime_error("Unsupported pose model format: " + model_format);
        }

        normalize(backend);
        if (backend == "tensorrt") {
            if (model_format_ != ModelFormat::YoloCoco17) {
                throw std::runtime_error(
                    "Native TensorRT currently supports the YOLO pose format.");
            }
            tensor_rt_ = std::make_unique<TensorRtRunner>(model_weights);
        } else {
            net_ = model_format_ == ModelFormat::YoloCoco17 || model_config.empty()
                       ? cv::dnn::readNet(model_weights)
                       : cv::dnn::readNet(model_weights, model_config);
            if (net_.empty()) {
                throw std::runtime_error("Failed to load pose model: " + model_weights);
            }
            net_.setPreferableBackend(parseBackend(backend));
            net_.setPreferableTarget(parseTarget(target));
        }
    }

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        const auto preprocess_start = Profiler::now();
        cv::Mat inference_image = ctx.processed_frame;
        YoloLetterboxTransform letterbox;
        bool swap_red_blue = false;
        if (model_format_ == ModelFormat::YoloCoco17) {
            inference_image = makeLetterbox(ctx.processed_frame, letterbox);
            swap_red_blue = true;
        }
        cv::Mat blob = cv::dnn::blobFromImage(
            inference_image, 1.0 / 255.0,
            cv::Size(input_width_, input_height_), cv::Scalar(0, 0, 0),
            swap_red_blue, false);
        net_.setInput(blob);
        const auto preprocess_end = Profiler::now();
        recordLatency(ctx, "pose_preprocess", preprocess_start, preprocess_end);

        const auto inference_start = Profiler::now();
        ctx.pose_inference_ran = true;
        cv::Mat output =
            tensor_rt_ ? tensor_rt_->forward(blob) : net_.forward();
        const auto inference_end = Profiler::now();
        recordLatency(ctx, "pose_inference", inference_start, inference_end);

        const auto postprocess_start = Profiler::now();
        if (model_format_ == ModelFormat::YoloCoco17) {
            ctx.poses = YoloPoseDecoder::decode(
                output, letterbox, ctx.frame_id, ctx.source_time_seconds,
                confidence_threshold_, detection_confidence_, nms_threshold_,
                max_poses_);
            ctx.pose_measurement_valid =
                std::any_of(ctx.poses.begin(), ctx.poses.end(),
                            [](const Pose& pose) { return pose.valid; });
        } else {
            decodeOpenPose(ctx, output);
        }
        const auto postprocess_end = Profiler::now();
        recordLatency(ctx, "pose_postprocess", postprocess_start, postprocess_end);
    }

    std::string name() const override {
        return "pose_estimator";
    }

private:
    enum class ModelFormat {
        OpenPoseCoco,
        YoloCoco17,
    };

    cv::Mat makeLetterbox(const cv::Mat& frame,
                          YoloLetterboxTransform& transform) const {
        transform.scale = std::min(
            input_width_ / static_cast<float>(frame.cols),
            input_height_ / static_cast<float>(frame.rows));
        const int resized_width =
            std::max(1, cvRound(frame.cols * transform.scale));
        const int resized_height =
            std::max(1, cvRound(frame.rows * transform.scale));
        const int left = (input_width_ - resized_width) / 2;
        const int top = (input_height_ - resized_height) / 2;
        transform.pad_x = static_cast<float>(left);
        transform.pad_y = static_cast<float>(top);
        transform.original_width = frame.cols;
        transform.original_height = frame.rows;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(resized_width, resized_height));
        cv::Mat letterboxed(input_height_, input_width_, frame.type(),
                            cv::Scalar(114, 114, 114));
        resized.copyTo(letterboxed(
            cv::Rect(left, top, resized_width, resized_height)));
        return letterboxed;
    }

    void decodeOpenPose(FrameContext& ctx, cv::Mat& output) const {
        if (output.dims != 4 ||
            output.size[1] < static_cast<int>(kOpenPoseCocoChannelToJoint.size())) {
            throw std::runtime_error(
                "Pose model output is not OpenPose COCO 18-keypoint format.");
        }

        const int heatmap_height = output.size[2];
        const int heatmap_width = output.size[3];
        const int frame_width = ctx.processed_frame.cols;
        const int frame_height = ctx.processed_frame.rows;

        Pose pose;
        pose.frame_id = ctx.frame_id;
        pose.source_time_seconds = ctx.source_time_seconds;
        pose.frame_size_pixels =
            FrameSize2D{static_cast<float>(frame_width), static_cast<float>(frame_height)};

        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        for (std::size_t part = 0; part < kOpenPoseCocoChannelToJoint.size(); ++part) {
            cv::Mat heatmap(heatmap_height, heatmap_width, CV_32F, output.ptr(0, part));

            double confidence = 0.0;
            cv::Point max_location;
            cv::minMaxLoc(heatmap, nullptr, &confidence, nullptr, &max_location);

            auto& keypoint = pose.joint(kOpenPoseCocoChannelToJoint[part]);
            keypoint.confidence = static_cast<float>(confidence);
            keypoint.valid = confidence >= confidence_threshold_;
            keypoint.position_2d_pixels = Point2D{
                (frame_width * max_location.x) / static_cast<float>(heatmap_width),
                (frame_height * max_location.y) / static_cast<float>(heatmap_height),
            };

            if (keypoint.valid) {
                min_x = std::min(min_x, keypoint.position_2d_pixels.x);
                min_y = std::min(min_y, keypoint.position_2d_pixels.y);
                max_x = std::max(max_x, keypoint.position_2d_pixels.x);
                max_y = std::max(max_y, keypoint.position_2d_pixels.y);
            }
        }

        pose.valid = std::any_of(
            pose.joints.begin(), pose.joints.end(),
            [](const PoseKeypoint& keypoint) { return keypoint.valid; });
        if (pose.valid) {
            pose.bounding_box = BoundingBox2D{
                min_x,
                min_y,
                max_x - min_x,
                max_y - min_y,
                true,
            };
        }

        ctx.poses.clear();
        ctx.poses.push_back(pose);
        ctx.pose_measurement_valid = pose.valid;
    }

    void recordLatency(FrameContext& ctx, const std::string& stage,
                       std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) const {
        ctx.stage_names.push_back(stage);
        ctx.stage_latencies_ms.push_back(Profiler::elapsedMs(start, end));
    }

    int parseBackend(std::string backend) const {
        normalize(backend);
        if (backend == "default") {
            return cv::dnn::DNN_BACKEND_DEFAULT;
        }
        if (backend == "opencv") {
            return cv::dnn::DNN_BACKEND_OPENCV;
        }
        if (backend == "openvino" || backend == "inference_engine") {
            return cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
        }
        if (backend == "cuda") {
            return cv::dnn::DNN_BACKEND_CUDA;
        }
        if (backend == "vkcom" || backend == "vulkan") {
            return cv::dnn::DNN_BACKEND_VKCOM;
        }
        throw std::runtime_error("Unsupported pose backend: " + backend);
    }

    int parseTarget(std::string target) const {
        normalize(target);
        if (target == "cpu") {
            return cv::dnn::DNN_TARGET_CPU;
        }
        if (target == "cpu_fp16" || target == "fp16") {
            return cv::dnn::DNN_TARGET_CPU_FP16;
        }
        if (target == "opencl" || target == "gpu") {
            return cv::dnn::DNN_TARGET_OPENCL;
        }
        if (target == "opencl_fp16" || target == "gpu_fp16") {
            return cv::dnn::DNN_TARGET_OPENCL_FP16;
        }
        if (target == "vulkan") {
            return cv::dnn::DNN_TARGET_VULKAN;
        }
        if (target == "cuda") {
            return cv::dnn::DNN_TARGET_CUDA;
        }
        if (target == "cuda_fp16") {
            return cv::dnn::DNN_TARGET_CUDA_FP16;
        }
        if (target == "myriad" || target == "vpu") {
            return cv::dnn::DNN_TARGET_MYRIAD;
        }
        if (target == "hddl") {
            return cv::dnn::DNN_TARGET_HDDL;
        }
        if (target == "npu") {
            return cv::dnn::DNN_TARGET_NPU;
        }
        throw std::runtime_error("Unsupported pose target: " + target);
    }

    void normalize(std::string& value) const {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        std::replace(value.begin(), value.end(), '-', '_');
    }

    int input_width_;
    int input_height_;
    float confidence_threshold_;
    float detection_confidence_;
    float nms_threshold_;
    int max_poses_;
    ModelFormat model_format_ = ModelFormat::OpenPoseCoco;
    cv::dnn::Net net_;
    std::unique_ptr<TensorRtRunner> tensor_rt_;
};

}  // namespace video_engine
