#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "pose/PoseTypes.hpp"

namespace video_engine {

struct YoloLetterboxTransform {
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
    int original_width = 0;
    int original_height = 0;
};

class YoloPoseDecoder {
public:
    static std::vector<Pose> decode(
        const cv::Mat& output, const YoloLetterboxTransform& transform,
        std::uint64_t frame_id, double source_time_seconds,
        float keypoint_confidence, float detection_confidence,
        float nms_threshold, int max_poses) {
        if (output.empty() || output.type() != CV_32F) {
            throw std::runtime_error("YOLO pose output must be a non-empty FP32 tensor.");
        }
        if (transform.scale <= 0.0F || transform.original_width <= 0 ||
            transform.original_height <= 0) {
            throw std::runtime_error("YOLO pose letterbox transform is invalid.");
        }

        TensorView tensor(output);
        if (tensor.features < kFeatureCount) {
            throw std::runtime_error(
                "YOLO11 pose output must contain 56 features per proposal.");
        }

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> scores;
        std::vector<Pose> candidates;
        for (int proposal = 0; proposal < tensor.proposals; ++proposal) {
            const float score = tensor.at(kBoxFeatureCount, proposal);
            if (score < detection_confidence) {
                continue;
            }

            const float center_x = unletterboxX(
                tensor.at(0, proposal), transform);
            const float center_y = unletterboxY(
                tensor.at(1, proposal), transform);
            const float width = tensor.at(2, proposal) / transform.scale;
            const float height = tensor.at(3, proposal) / transform.scale;
            const float left = clamp(center_x - width * 0.5F, 0.0F,
                                     static_cast<float>(transform.original_width - 1));
            const float top = clamp(center_y - height * 0.5F, 0.0F,
                                    static_cast<float>(transform.original_height - 1));
            const float right = clamp(center_x + width * 0.5F, left,
                                      static_cast<float>(transform.original_width - 1));
            const float bottom = clamp(center_y + height * 0.5F, top,
                                       static_cast<float>(transform.original_height - 1));

            Pose pose;
            pose.person_id = static_cast<std::int64_t>(candidates.size());
            pose.frame_id = frame_id;
            pose.source_time_seconds = source_time_seconds;
            pose.frame_size_pixels = FrameSize2D{
                static_cast<float>(transform.original_width),
                static_cast<float>(transform.original_height),
            };
            pose.bounding_box = BoundingBox2D{
                left, top, right - left, bottom - top, true,
            };

            for (std::size_t keypoint = 0; keypoint < kCocoKeypointMap.size();
                 ++keypoint) {
                const int offset =
                    kBoxFeatureCount + kClassCount +
                    static_cast<int>(keypoint) * kValuesPerKeypoint;
                auto& destination = pose.joint(kCocoKeypointMap[keypoint]);
                destination.position_2d_pixels = Point2D{
                    clamp(unletterboxX(tensor.at(offset, proposal), transform),
                          0.0F, static_cast<float>(transform.original_width - 1)),
                    clamp(unletterboxY(tensor.at(offset + 1, proposal), transform),
                          0.0F, static_cast<float>(transform.original_height - 1)),
                };
                destination.confidence = tensor.at(offset + 2, proposal);
                destination.valid =
                    destination.confidence >= keypoint_confidence;
            }
            synthesizeNeck(pose, keypoint_confidence);
            pose.valid = std::any_of(
                pose.joints.begin(), pose.joints.end(),
                [](const PoseKeypoint& point) { return point.valid; });

            nms_boxes.emplace_back(
                cvRound(left), cvRound(top),
                std::max(1, cvRound(right - left)),
                std::max(1, cvRound(bottom - top)));
            scores.push_back(score);
            candidates.push_back(pose);
        }

        std::vector<int> kept;
        cv::dnn::NMSBoxes(nms_boxes, scores, detection_confidence, nms_threshold,
                          kept, 1.0F, max_poses > 0 ? max_poses : 0);

        std::vector<Pose> poses;
        poses.reserve(kept.size());
        for (const int index : kept) {
            auto pose = candidates.at(static_cast<std::size_t>(index));
            pose.person_id = static_cast<std::int64_t>(poses.size());
            poses.push_back(std::move(pose));
        }
        return poses;
    }

private:
    static constexpr int kBoxFeatureCount = 4;
    static constexpr int kClassCount = 1;
    static constexpr int kValuesPerKeypoint = 3;
    static constexpr int kFeatureCount =
        kBoxFeatureCount + kClassCount + 17 * kValuesPerKeypoint;

    inline static constexpr std::array<JointId, 17> kCocoKeypointMap = {
        JointId::Nose,
        JointId::LeftEye,
        JointId::RightEye,
        JointId::LeftEar,
        JointId::RightEar,
        JointId::LeftShoulder,
        JointId::RightShoulder,
        JointId::LeftElbow,
        JointId::RightElbow,
        JointId::LeftWrist,
        JointId::RightWrist,
        JointId::LeftHip,
        JointId::RightHip,
        JointId::LeftKnee,
        JointId::RightKnee,
        JointId::LeftAnkle,
        JointId::RightAnkle,
    };

    struct TensorView {
        explicit TensorView(const cv::Mat& output) : data(output.ptr<float>()) {
            int first = 0;
            int second = 0;
            if (output.dims == 3 && output.size[0] == 1) {
                first = output.size[1];
                second = output.size[2];
            } else if (output.dims == 2) {
                first = output.rows;
                second = output.cols;
            } else {
                throw std::runtime_error(
                    "YOLO pose output must have shape [1,56,N] or [1,N,56].");
            }

            if (first >= kFeatureCount && first <= 128) {
                features = first;
                proposals = second;
                feature_major = true;
            } else if (second >= kFeatureCount && second <= 128) {
                features = second;
                proposals = first;
                feature_major = false;
            } else {
                throw std::runtime_error(
                    "YOLO pose output has an unsupported tensor layout.");
            }
        }

        float at(int feature, int proposal) const {
            return feature_major
                       ? data[feature * proposals + proposal]
                       : data[proposal * features + feature];
        }

        const float* data;
        int features = 0;
        int proposals = 0;
        bool feature_major = true;
    };

    static float clamp(float value, float minimum, float maximum) {
        return std::max(minimum, std::min(value, maximum));
    }

    static float unletterboxX(float value,
                              const YoloLetterboxTransform& transform) {
        return (value - transform.pad_x) / transform.scale;
    }

    static float unletterboxY(float value,
                              const YoloLetterboxTransform& transform) {
        return (value - transform.pad_y) / transform.scale;
    }

    static void synthesizeNeck(Pose& pose, float confidence_threshold) {
        const auto& left = pose.joint(JointId::LeftShoulder);
        const auto& right = pose.joint(JointId::RightShoulder);
        auto& neck = pose.joint(JointId::Neck);
        neck.confidence = std::min(left.confidence, right.confidence);
        neck.valid = left.valid && right.valid &&
                     neck.confidence >= confidence_threshold;
        if (neck.valid) {
            neck.position_2d_pixels = Point2D{
                (left.position_2d_pixels.x + right.position_2d_pixels.x) * 0.5F,
                (left.position_2d_pixels.y + right.position_2d_pixels.y) * 0.5F,
            };
        }
    }
};

}  // namespace video_engine
