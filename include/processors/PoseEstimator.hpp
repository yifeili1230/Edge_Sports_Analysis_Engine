#pragma once

#include <algorithm>  // Normalizes backend/target strings and replaces separators.
#include <cctype>     // Converts backend/target names to lowercase safely.
#include <limits>     // Initializes pose bounding-box extrema.
#include <stdexcept>  // Reports model loading and backend configuration errors.
#include <string>     // Stores model paths and backend/target names.

#include <opencv2/core.hpp>    // Provides cv::Mat, cv::Point, and scalar types.
#include <opencv2/dnn.hpp>     // Provides OpenCV DNN model loading and inference.
#include <opencv2/imgproc.hpp> // Provides blob/image preprocessing helpers.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.
#include "pose/OpenPoseCocoAdapter.hpp" // Maps OpenPose output channels to canonical joints.
#include "utils/Profiler.hpp"        // Measures preprocessing, inference, and postprocessing latency.

namespace video_engine {

class PoseEstimator : public IFrameProcessor {
public:
    explicit PoseEstimator(std::string model_weights, std::string model_config = "",
                           int input_width = 368, int input_height = 368,
                           float confidence_threshold = 0.12F,
                           std::string backend = "opencv", std::string target = "cpu")
        : input_width_(input_width),
          input_height_(input_height),
          confidence_threshold_(confidence_threshold) {
        if (model_weights.empty()) {
            throw std::runtime_error("Pose model path is empty. Set pose_model in configs/pose.yaml.");
        }

        net_ = model_config.empty() ? cv::dnn::readNet(model_weights)
                                    : cv::dnn::readNet(model_weights, model_config);
        if (net_.empty()) {
            throw std::runtime_error("Failed to load pose model: " + model_weights);
        }

        net_.setPreferableBackend(parseBackend(backend));
        net_.setPreferableTarget(parseTarget(target));
    }

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        const auto preprocess_start = Profiler::now();
        cv::Mat blob = cv::dnn::blobFromImage(ctx.processed_frame, 1.0 / 255.0,
                                              cv::Size(input_width_, input_height_),
                                              cv::Scalar(0, 0, 0), false, false);
        net_.setInput(blob);
        const auto preprocess_end = Profiler::now();
        recordLatency(ctx, "pose_preprocess", preprocess_start, preprocess_end);

        const auto inference_start = Profiler::now();
        ctx.pose_inference_ran = true;
        cv::Mat output = net_.forward();
        const auto inference_end = Profiler::now();
        recordLatency(ctx, "pose_inference", inference_start, inference_end);

        if (output.dims != 4 ||
            output.size[1] < static_cast<int>(kOpenPoseCocoChannelToJoint.size())) {
            throw std::runtime_error("Pose model output is not OpenPose COCO 18-keypoint format.");
        }

        const auto postprocess_start = Profiler::now();
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
        const auto postprocess_end = Profiler::now();
        recordLatency(ctx, "pose_postprocess", postprocess_start, postprocess_end);
    }

    std::string name() const override {
        return "pose_estimator";
    }

private:
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
    cv::dnn::Net net_;
};

}  // namespace video_engine
