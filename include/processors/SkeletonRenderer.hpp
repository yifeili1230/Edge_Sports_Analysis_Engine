#pragma once

#include <array>  // Stores the fixed skeleton edge list.

#include <opencv2/imgproc.hpp>  // Provides drawing APIs for skeleton lines and keypoints.

#include "core/IFrameProcessor.hpp" // Defines the frame processor interface.
#include "pose/PoseTypes.hpp"       // Provides canonical named joints.

namespace video_engine {

class SkeletonRenderer : public IFrameProcessor {
public:
    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        cv::Mat overlay = ctx.processed_frame.clone();
        for (const auto& pose : ctx.poses) {
            if (!pose.valid) {
                continue;
            }

            for (const auto& [first, second] : kSkeletonPairs) {
                const auto& a = pose.joint(first);
                const auto& b = pose.joint(second);
                if (a.valid && b.valid) {
                    cv::line(overlay, toCvPoint(a.position_2d_pixels),
                             toCvPoint(b.position_2d_pixels),
                             cv::Scalar(45, 210, 255), 3, cv::LINE_AA);
                }
            }

            for (std::size_t i = 0; i < pose.joints.size(); ++i) {
                const auto& keypoint = pose.joints[i];
                if (!keypoint.valid) {
                    continue;
                }
                const bool is_nose = i == jointIndex(JointId::Nose);
                const cv::Scalar color =
                    is_nose ? cv::Scalar(0, 80, 255) : cv::Scalar(40, 255, 90);
                const cv::Point point = toCvPoint(keypoint.position_2d_pixels);
                cv::circle(overlay, point, is_nose ? 6 : 5, color, cv::FILLED, cv::LINE_AA);
                cv::circle(overlay, point, is_nose ? 8 : 7, cv::Scalar(20, 20, 20), 1,
                           cv::LINE_AA);
            }
        }

        ctx.processed_frame = overlay;
    }

    std::string name() const override {
        return "skeleton_renderer";
    }

private:
    static cv::Point toCvPoint(const Point2D& point) {
        return cv::Point(cvRound(point.x), cvRound(point.y));
    }

    static constexpr std::array<std::pair<JointId, JointId>, 14> kSkeletonPairs = {{
        {JointId::Neck, JointId::RightShoulder},
        {JointId::RightShoulder, JointId::RightElbow},
        {JointId::RightElbow, JointId::RightWrist},
        {JointId::Neck, JointId::LeftShoulder},
        {JointId::LeftShoulder, JointId::LeftElbow},
        {JointId::LeftElbow, JointId::LeftWrist},
        {JointId::Neck, JointId::RightHip},
        {JointId::RightHip, JointId::RightKnee},
        {JointId::RightKnee, JointId::RightAnkle},
        {JointId::Neck, JointId::LeftHip},
        {JointId::LeftHip, JointId::LeftKnee},
        {JointId::LeftKnee, JointId::LeftAnkle},
        {JointId::Nose, JointId::Neck},
        {JointId::RightHip, JointId::LeftHip},
    }};
};

}  // namespace video_engine
